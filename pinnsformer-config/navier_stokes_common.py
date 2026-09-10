"""Shared Navier-Stokes PINNsFormer definitions for diagnostics and ConFIG tests.

Model/data/PDE definitions follow the user's previously executed notebook and the
AdityaLab/pinnsformer Navier-Stokes demo. Training/evaluation policy lives in the
notebooks so experimental differences remain explicit.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn as nn


def make_time_sequence(src, num_step=5, step=1e-4):
    """Same logic as AdityaLab/pinnsformer util.py::make_time_sequence."""
    dim = num_step
    src = np.repeat(np.expand_dims(src, axis=1), dim, axis=1)
    for i in range(num_step):
        src[:, i, -1] += step * i
    return src


def get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def get_n_params(model):
    return sum(p.numel() for p in model.parameters())


class WaveAct(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(d_model, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.linear(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()

    def forward(self, x):
        x2 = self.act1(x)
        x = x + self.attn(x2, x2, x2)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()

    def forward(self, x, e_outputs):
        x2 = self.act1(x)
        x = x + self.attn(x2, e_outputs, e_outputs)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, n_layers, heads):
        super().__init__()
        self.n_layers = n_layers
        self.layers = get_clones(EncoderLayer(d_model, heads), n_layers)
        self.act = WaveAct()

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.layers[i](x)
        return self.act(x)


class Decoder(nn.Module):
    def __init__(self, d_model, n_layers, heads):
        super().__init__()
        self.n_layers = n_layers
        self.layers = get_clones(DecoderLayer(d_model, heads), n_layers)
        self.act = WaveAct()

    def forward(self, x, e_outputs):
        for i in range(self.n_layers):
            x = self.layers[i](x, e_outputs)
        return self.act(x)


class PINNsformer(nn.Module):
    def __init__(self, d_out, d_model, d_hidden, N, heads):
        super().__init__()
        self.linear_emb = nn.Linear(3, d_model)
        self.encoder = Encoder(d_model, N, heads)
        self.decoder = Decoder(d_model, N, heads)
        self.linear_out = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x, y, t):
        src = torch.cat((x, y, t), dim=-1)
        src = self.linear_emb(src)
        e_outputs = self.encoder(src)
        d_output = self.decoder(src, e_outputs)
        return self.linear_out(d_output)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def load_training_data(data_path, device, *, seed=0, n_train=800, num_step=5, time_step=1e-2):
    """Load and sample the same Navier-Stokes training data structure as the uploaded notebook."""
    data = scipy.io.loadmat(Path(data_path))
    U_star = data['U_star']
    P_star = data['p_star']
    t_star = data['t']
    X_star = data['X_star']

    n_space = X_star.shape[0]
    n_time = t_star.shape[0]
    XX = np.tile(X_star[:, 0:1], (1, n_time))
    YY = np.tile(X_star[:, 1:2], (1, n_time))
    TT = np.tile(t_star, (1, n_space)).T
    UU = U_star[:, 0, :]
    VV = U_star[:, 1, :]

    x = XX.flatten()[:, None]
    y = YY.flatten()[:, None]
    t = TT.flatten()[:, None]
    u = UU.flatten()[:, None]
    v = VV.flatten()[:, None]

    rng = np.random.RandomState(seed)
    idx = rng.choice(n_space * n_time, n_train, replace=False)
    x_np = np.expand_dims(np.tile(x[idx, :], num_step), -1)
    y_np = np.expand_dims(np.tile(y[idx, :], num_step), -1)
    t_np = make_time_sequence(t[idx, :], num_step=num_step, step=time_step)

    batch = {
        'x': torch.tensor(x_np, dtype=torch.float32, requires_grad=True, device=device),
        'y': torch.tensor(y_np, dtype=torch.float32, requires_grad=True, device=device),
        't': torch.tensor(t_np, dtype=torch.float32, requires_grad=True, device=device),
        'u': torch.tensor(u[idx, :], dtype=torch.float32, device=device),
        'v': torch.tensor(v[idx, :], dtype=torch.float32, device=device),
        'idx': idx,
    }
    reference = {'U_star': U_star, 'P_star': P_star, 't_star': t_star, 'X_star': X_star, 'TT': TT}
    return batch, reference


def compute_ns_losses(model, x_in, y_in, t_in, u_obs, v_obs):
    psi_and_p = model(x_in, y_in, t_in)
    psi = psi_and_p[:, :, 0:1]
    pressure = psi_and_p[:, :, 1:2]

    # Training convention in the uploaded notebook: u=dpsi/dy, v=-dpsi/dx.
    u_pred = torch.autograd.grad(psi, y_in, grad_outputs=torch.ones_like(psi), retain_graph=True, create_graph=True)[0]
    v_pred = -torch.autograd.grad(psi, x_in, grad_outputs=torch.ones_like(psi), retain_graph=True, create_graph=True)[0]

    u_t = torch.autograd.grad(u_pred, t_in, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0]
    u_x = torch.autograd.grad(u_pred, x_in, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0]
    u_y = torch.autograd.grad(u_pred, y_in, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_in, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_in, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]

    v_t = torch.autograd.grad(v_pred, t_in, grad_outputs=torch.ones_like(v_pred), retain_graph=True, create_graph=True)[0]
    v_x = torch.autograd.grad(v_pred, x_in, grad_outputs=torch.ones_like(v_pred), retain_graph=True, create_graph=True)[0]
    v_y = torch.autograd.grad(v_pred, y_in, grad_outputs=torch.ones_like(v_pred), retain_graph=True, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x_in, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y_in, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0]

    p_x = torch.autograd.grad(pressure, x_in, grad_outputs=torch.ones_like(pressure), retain_graph=True, create_graph=True)[0]
    p_y = torch.autograd.grad(pressure, y_in, grad_outputs=torch.ones_like(pressure), retain_graph=True, create_graph=True)[0]

    f_u = u_t + (u_pred * u_x + v_pred * u_y) + p_x - 0.01 * (u_xx + u_yy)
    f_v = v_t + (u_pred * v_x + v_pred * v_y) + p_y - 0.01 * (v_xx + v_yy)

    loss_u_data = torch.mean((u_pred[:, 0] - u_obs) ** 2)
    loss_v_data = torch.mean((v_pred[:, 0] - v_obs) ** 2)
    loss_fu = torch.mean(f_u ** 2)
    loss_fv = torch.mean(f_v ** 2)
    loss_data = loss_u_data + loss_v_data
    loss_physics = loss_fu + loss_fv

    return {
        'total': loss_data + loss_physics,
        'data': loss_data,
        'physics': loss_physics,
        'u_data': loss_u_data,
        'v_data': loss_v_data,
        'f_u': loss_fu,
        'f_v': loss_fv,
    }


def evaluation_tensors(reference, device, *, snap=100, num_step=5, time_step=1e-2):
    X_star = reference['X_star']
    TT = reference['TT']
    U_star = reference['U_star']
    P_star = reference['P_star']
    snap_arr = np.array([snap])

    x_np = np.expand_dims(np.tile(X_star[:, 0:1], num_step), -1)
    y_np = np.expand_dims(np.tile(X_star[:, 1:2], num_step), -1)
    t_np = make_time_sequence(TT[:, snap_arr], num_step=num_step, step=time_step)

    tensors = {
        'x': torch.tensor(x_np, dtype=torch.float32, requires_grad=True, device=device),
        'y': torch.tensor(y_np, dtype=torch.float32, requires_grad=True, device=device),
        't': torch.tensor(t_np, dtype=torch.float32, requires_grad=True, device=device),
    }
    truth = {
        'u': U_star[:, 0, snap_arr],
        'v': U_star[:, 1, snap_arr],
        'p': P_star[:, snap_arr],
    }
    return tensors, truth
