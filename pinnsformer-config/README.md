---
title: PINNsFormer + ConFIG 诊断与适配代码
summary: 基于 PINNsFormer 官方 Navier–Stokes demo 和用户已运行 notebook，先诊断 data/physics 参数梯度冲突，再提供数据、网络和 PDE 设置一致的 Adam simple-sum / ConFIG 2-loss 公平实验。
type: repo-note
status: to-run
created: 2026-09-10
updated: 2026-09-10
papers:
  - PINNsFormer: A Transformer-Based Framework for Physics-Informed Neural Networks
  - ConFIG: Towards Conflict-Free Training of Physics Informed Neural Networks
github_repositories:
  - https://github.com/AdityaLab/pinnsformer
  - https://github.com/tum-pbs/ConFIG
data:
  - PINNsFormer 官方 demo/navier_stokes/cylinder_nektar_wake.mat；公开 demo 数据
---

# PINNsFormer + ConFIG

## 1. 文件

| 文件 | 作用 | 当前状态 |
|---|---|---|
| `gradient_diagnostics.py` | 梯度向量、norm、pairwise cosine、conflict rate | 静态检查完成，待服务器真实运行 |
| `navier_stokes_common.py` | 共享 PINNsFormer 结构、数据采样、Navier–Stokes loss 与评估张量 | 静态检查完成，待服务器真实运行 |
| `navier_stokes_gradient_diagnostic.ipynb` | 加载既有 `ns_pinnsformer.pt`，诊断 2-loss/4-loss 梯度冲突 | 待运行 |
| `navier_stokes_adam_baseline.ipynb` | `Adam(lr=1e-4)` + simple-sum 的公平对照组 | 待运行 |
| `navier_stokes_config.ipynb` | 同设置下 2-loss `L_data/L_physics` + ConFIG + Adam | 待运行 |

## 2. 与上传 notebook 保持不变的设置

- seed = 0；
- `N_TRAIN = 800`；
- pseudo sequence 长度 5，时间步 `1e-2`；
- `PINNsformer(d_out=2, d_hidden=512, d_model=32, N=1, heads=2)`；
- 数据抽样逻辑和 `cylinder_nektar_wake.mat`；
- stream-function 训练定义 $u=\psi_y,\ v=-\psi_x$；
- 两个 Navier–Stokes momentum residual 与黏性系数 0.01；
- 外层 optimizer steps = 1000。

上传 notebook 原始训练使用 LBFGS strong-Wolfe，而 ConFIG 组合后的梯度一般不再是 `L_total` 的普通梯度，因此不建议直接交给 LBFGS line search。为了把 ConFIG 本身作为唯一变量，本目录同时提供两份**匹配的一阶优化实验**：

```text
PINNsFormer + Adam(lr=1e-4) + simple sum
                 VS
PINNsFormer + Adam(lr=1e-4) + ConFIG
```

两者的 seed、数据点、网络、PDE、sequence 和 1000 optimizer steps 相同，只有多损失梯度合成方式不同。`lr=1e-4` 是相对于原 LBFGS notebook 新增的超参数，因此原始 LBFGS 结果只作为“原方法参考”，不能直接用来隔离 ConFIG 的收益。

另外，LBFGS 的一次外层 `step(closure)` 可能多次调用 closure，因此“LBFGS 1000 外层循环”和“Adam 1000 optimizer steps”也不是同等函数评估预算。

## 3. 运行前环境

默认路径与服务器一致：

```text
PINNsFormer: /home/simplexity/cyt/pinnsformer-main
ConFIG:      /home/simplexity/cyt/ConFIG-main
```

可用环境变量覆盖：

```bash
export PINNSFORMER_ROOT=/path/to/pinnsformer-main
export CONFIG_ROOT=/path/to/ConFIG-main
```

ConFIG 环境至少需要 `torch numpy scipy matplotlib tqdm conflictfree`。如果沿用此前 `config_pinn`，缺少 SciPy 时再安装 `scipy`，不要无理由改动其它已固定依赖。

## 4. 推荐运行顺序

```text
1. navier_stokes_gradient_diagnostic.ipynb
   先确认 data ↔ physics 的 gradient norm / cosine / conflict
        ↓
2. navier_stokes_adam_baseline.ipynb
   得到同 optimizer、同 LR 的 simple-sum baseline
        ↓
3. navier_stokes_config.ipynb
   只把梯度合成改成 ConFIG
        ↓
4. 对比 relative L2 / loss / conflict / wall time / peak memory
        ↓
5. 再决定是否做 4-loss ConFIG / M-ConFIG
```

## 5. 一个已知的上游/上传代码一致性问题

上传 notebook 的训练阶段使用 $u=\psi_y,\ v=-\psi_x$，但测试阶段写成 $u=\psi_x,\ v=-\psi_y$。上游 Navier–Stokes PINNsFormer notebook 也存在同样的训练/测试不一致。这里不静默覆盖原始事实：诊断代码按训练定义；两个新训练 notebook 的 evaluation 也按训练定义，并在 notebook 中明确注明。

因此，用户之前得到的 velocity 测试误差不能直接作为新实验公平 baseline；应先用保存的原始 checkpoint 按训练一致定义重新评估。

## 6. 当前结论边界

此目录代码**尚未在 Linux 服务器真实运行**。目前只能说适配逻辑和 Python/Notebook 结构已准备；不能提前声称 ConFIG 能提高 PINNsFormer 精度、降低显存或降低 wall time。
