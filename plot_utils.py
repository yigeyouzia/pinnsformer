"""训练损失可视化工具

Usage: 在 notebook 训练 cell 末尾加一行即可：

    from plot_utils import plot_loss
    plot_loss(loss_track, save_path='./loss_curve.png')

参数:
    loss_track : list of scalar 或 list of [loss_res, loss_bc, loss_ic, ...]
    save_path  : 保存路径，默认 './loss_curve.png'
    title      : 图标题，默认 'Training Loss'
    labels     : 自定义标签列表；不传则自动推断
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_loss(loss_track, save_path='./loss_curve.png', title='Training Loss',
              labels=None):
    loss_track = np.array(loss_track)

    # 处理 1D 标量（如 Navier-Stokes），reshape 为 2D
    if loss_track.ndim == 1:
        loss_track = loss_track[:, None]

    epochs = np.arange(1, len(loss_track) + 1)
    n = loss_track.shape[1]

    # 自动推断标签
    if labels is None:
        if n == 1:
            labels = ['Total Loss']
        elif n == 3:
            labels = ['Loss_Res', 'Loss_BC', 'Loss_IC']
        elif n == 4:
            labels = ['Loss_Res', 'Loss_BC', 'Loss_IC', 'Loss_IC2']
        else:
            labels = [f'Loss_{i+1}' for i in range(n)]

    plt.figure(figsize=(6, 4))
    for i in range(n):
        plt.plot(epochs, loss_track[:, i], label=labels[i], linewidth=1)

    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f'Loss curve saved to {save_path}')
