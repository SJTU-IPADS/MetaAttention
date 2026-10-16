# Scan Stage DAG 与稳定 LogAddExp algebra

状态：accepted。

Schema v5 增加显式 `ScanStageIR`：每个 `ScanStage` 使用独立验证的 affine `AlgorithmIR` 或有限的 `LogAddExpScanIR`，以 typed `StageBinding` 将上游 Readout 轨迹绑定为下游内部 Input。这样可表达 Preconditioned KDA/GDN 的非线性系数变换，而不将整个程序错误地证明为联合 affine map；单 token 的 [Transition DAG](0004-ordered-affine-state-transitions.md) 保持原有语义。

先采用 Readout 作为阶段导出，而不是增加任意 callback 或开放下游 State feedback。Readout 已有 prior/current 时序、类型与 dtype 规则，因此上游状态变换和下游 scan 可以复用现有分析。Public State 名称全局唯一，顺序按 stage 声明而非执行拓扑；内部绑定不成为调用者输入。首版绑定要求源/目标相同 logical roles 和 dtype，支持 role 物理排列变化；head sharing 在下游 HeadMapping 中处理，使 auxiliary State 保持原生 key-head 形状。

稳定 LightNet 使用专门的有限 algebra：`z' = logaddexp(z, logits)`，identity 为 `-inf`，readouts 为 normalized key 和 log gate。Identity 属于 scan 合同而不是允许任意无限 Constant；`logaddexp(-inf,-inf)` 的 VJP 定义为零。Cold prior 的 gate 为 0，finite prior 使用 `z_prior-z_current`，与 FLA operator 保持一致。这个节点不开放一般 Nonlinear State Feedback。

组合执行物化 upstream trajectories，并按最后消费者释放 Python 引用；梯度由可微 child scans 与表达式链路组合。可选 `checkpoint=True` 通过 non-reentrant Torch checkpoint 重算阶段内部值，避免保留所有 scan 层的保存张量。资源估计仍采用保守的非 checkpoint 预算，包含轨迹、表达式、State 和反向 scan 各级；checkpoint 不降低合法性检查的预算。

`Abs` 与显式 `Cast` 支持 FP32/BF16 转换。Cast 的位置属于数学精度合同，不能仅靠修改 leaf dtype 或隐式标量 promotion 替代。Preconditioned examples 先 Cast q/k/v 为 FP32 再运算；Abs 零点采用零 subgradient。

这是阶段组合、缓存和梯度的能力交付。Preconditioned matrix stages 仍可使用 reference affine scan，长序列内存与延迟需单独优化；稳定 log scan 也使用 PyTorch segmented parallel prefix。首版不保证 FLA 性能相当。测量合同见 [阶段基准](../stateful-stage-benchmarks.md)。
