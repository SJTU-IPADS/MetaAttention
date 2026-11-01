# PR 矩阵测试结果分析

本报告分析唯一保留的 `benchmark_results/pr_matrix_full`。三个 Composition 作业的 8 GiB 补测已纳入矩阵，原始失败原因保留在 consolidation.json 和对应日志中。

## 结论

收集已完成，验证尚未全部通过。专用路径有明显性能优势，但最终状态正确性、表达式泛化、反向 workspace 和编译环境仍有未解决问题。8 GiB 补测仅覆盖三个 Composition 作业。

## 范围与统计口径

- 合并后：900 个作业均 COLLECTED；其中三个 Composition 作业采用 8 GiB 补测。COLLECTED 只代表收集完成，不代表数值或性能通过。
- 自动化测试：1278 条，1263 PASS、14 SKIPPED、1 XFAIL，无 FAIL/ERROR。XFAIL 为 scalar decay + value-axis scale 的最终 State 不准确。
- 基准：合并后 7080 条（含 12 条 8 GiB 补测）；FLA 对比 6000、泛化 972、Composition 合计 24、stages 48、plans 36。
- 测量设备 NVIDIA H20；原矩阵续跑跨 GPU 4 / GPU 3，补测在 GPU 3。不是 H100 实机测量，也不是相对旧提交的回归对比。
- 性能指标为 warmed public-call CUDA event；比值 FLA / MetaAttention，>1 表示 MetaAttention 更快。只聚合 status=OK 的对比；包含 recurrent/prefill、多形状、多契约，聚合中位数不代表单一工作负载吞吐。
- 配置：主矩阵 workspace 1 GiB，补测 Composition 8 GiB；torch 2.11.0+cu128、TileLang 0.1.15、Triton 3.6.0、FLA 0.5.2。

## FLA 对比：正确性和性能

| 状态 | 记录数 |
| --- | ---: |
| OK | 4197 |
| FLA_INVALID | 331 |
| NOT_COMPARABLE | 481 |
| META_INVALID | 340 |
| META_RUNTIME_REJECTED | 182 |
| FLA_FAILED | 375 |
| META_COMPILE_REJECTED | 59 |
| META_FAILED | 35 |

有效对比 4197/6000（69.95%）；MetaAttention 较快 2106/4197（50.18%）。全体有效记录的比值中位数 1.002×、几何均值 1.012×，不足以声称整体领先。

| 算法 | 有效/240 | Meta 更快 | CUDA 比值中位数 |
| --- | ---: | ---: | ---: |
| abc | 42 | 23 | 1.056× |
| based | 42 | 4 | 0.647× |
| comba | 155 | 60 | 0.467× |
| delta-product | 84 | 41 | 0.997× |
| delta-rule | 240 | 99 | 0.625× |
| dplr | 27 | 8 | 0.202× |
| gdn | 240 | 92 | 0.673× |
| gdn2 | 240 | 40 | 0.633× |
| gla | 240 | 149 | 1.248× |
| gsa | 149 | 94 | 1.525× |
| hgrn | 240 | 112 | 0.968× |
| iplr | 18 | 0 | 0.342× |
| kda | 240 | 60 | 0.698× |
| lightnet | 240 | 18 | 0.265× |
| linear | 222 | 222 | 3.954× |
| normalized-linear | 240 | 240 | 2.345× |
| oja | 106 | 17 | 0.345× |
| preconditioned-gdn | 240 | 24 | 0.479× |
| preconditioned-kda | 120 | 26 | 0.190× |
| rebased | 19 | 4 | 0.913× |
| retention | 170 | 169 | 2.763× |
| retnet | 173 | 173 | 2.782× |
| rwkv6 | 230 | 163 | 1.606× |
| rwkv7 | 240 | 33 | 0.701× |
| simple-gla | 240 | 235 | 3.900× |

Linear、Normalized Linear、Simple GLA、Retention/RetNet 的优势最明确；GDN、GDN2、KDA、Delta Rule、LightNet、RWKV7 等有效样本多数偏慢。DPLR/IPLR 等有效覆盖很少，排名受环境失败和样本选择影响。

按长度聚合的有效比值中位数：

| T | 有效记录 | FLA / Meta 中位数 |
| --- | ---: | ---: |
| 1 | 456 | 0.978× |
| 63 | 628 | 0.981× |
| 64 | 638 | 0.996× |
| 65 | 640 | 0.917× |
| 130 | 630 | 0.789× |
| 2048 | 609 | 1.223× |
| 8192 | 596 | 1.364× |

- 340 条 META_INVALID 全部来自 Based（144）和 Rebased（196）。Based 的 72 条 final-state 和 72 条 continuation 均不通过；Rebased 的 final-state 78、continuation 72、output-only 46 条不通过。不能只用输出看起来正常来认定状态契约正确。
- 331 条 FLA_INVALID 表示在本基准输入、契约和容差下对独立 FP32 oracle 不通过；不能计入性能胜负，也不据此泛称 FLA 算法错误。
- 422 条记录的错误明确含 No space left on device，涉及 Triton/TileLang cache 和 /tmp。375 条 FLA_FAILED 全部属于此类。
- 其余失败还含 nvcc 无法预处理 host compiler、nvcc segmentation fault、4 条 CUDA_ERROR_INVALID_IMAGE。META_COMPILE_REJECTED / META_RUNTIME_REJECTED 标签不等同于合法的编译器资源拒绝，必须结合错误文本。
- 481 条 NOT_COMPARABLE 含缺少公开 recurrent API 或契约不一致，应作为可比覆盖缺口，不能判胜负。

## 泛化测试

- 972 条：820 OK、16 INCORRECT、136 REJECTED。OK 表示正确性通过且完成测量，不等于 stable。
- 107 条 stable 全部为 level=0；864 条非零级别变体中，没有一条同时满足既定性能条件。默认条件为 variant/control <=1.25 且配对比值 relative IQR <=0.10。
- level=1 已可见 lowerer 切换，以下是正确且完成测量子集的 CUDA 延迟比中位数；变体增加了工作量，此比值衡量相对原始算法的性能包络，不是等工作量优化退化的证明。

| 变体族 | level=1 延迟 / control | 所选 lowerer |
| --- | ---: | --- |
| combined | 12.65× | tilelang-factored-recurrence |
| expression | 4.61× | tilelang-elementwise |
| propagation | 3.76× | tilelang-elementwise |
| rank | 6.19× | tilelang-factored-recurrence |
| readout | 7.16× | tilelang-elementwise |
| write | 7.57× | tilelang-elementwise |

- 16 条错误：expression level 5（8）、level 6（1）；propagation level 1（7）。前者示例为少数元素略超容差，后者示例超过 22% 元素不匹配，不能统一归为浮点抖动。
- 136 条拒绝均在 variant 阶段触发 runtime workload exceeds compiled resource envelope；未包含在本次 8 GiB Composition 补测内。

## Composition：补测结果与限制

下面仅列 8 GiB 补测。相同 Algorithm IR 的优化 lowerer 对比 generic dense lowerer；每项 10 批、每批 5 次计时，warmup=3，轮换执行顺序。这里的加速比不代表相对 FLA 的加速。

| B/T | 类型 | 优化 ms | Generic ms | 加速比 | 优化 MiB | Generic MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1/2048 | axis | 0.357 | 153.922 | 431.18× | 14.12 | 4996.02 |
| 1/2048 | injections | 0.379 | 2.645 | 6.98× | 16.12 | 1414.10 |
| 1/2048 | mixed | 1.800 | 155.430 | 86.34× | 47.33 | 5382.02 |
| 1/2048 | rank-one | 1.625 | 152.755 | 94.01× | 42.28 | 4614.02 |
| 2/512 | axis | 0.257 | 64.373 | 250.07× | 7.25 | 2501.01 |
| 2/512 | injections | 0.327 | 1.545 | 4.72× | 8.25 | 707.07 |
| 2/512 | mixed | 1.096 | 65.176 | 59.49× | 46.35 | 2694.03 |
| 2/512 | rank-one | 0.856 | 63.778 | 74.50× | 38.30 | 2307.03 |
| 2/2048 | axis | 0.589 | 314.668 | 533.80× | 28.25 | 10004.05 |
| 2/2048 | injections | 0.614 | 4.844 | 7.89× | 32.25 | 2828.27 |
| 2/2048 | mixed | 2.435 | 317.120 | 130.24× | 98.69 | 10776.11 |
| 2/2048 | rank-one | 1.938 | 312.069 | 161.01× | 84.59 | 9228.11 |

- B=2,T=2048：优化路径峰值增量仅 28.25–98.69 MiB；generic 为 2.76–10.52 GiB。说明 typed lowerer 避免 dense 展开对资源和延迟非常关键。
- workspace_bytes 是计划资源估算的上限，不是 PyTorch 全部峰值分配的硬上限；generic 峰值超过 8 GiB 不代表配置未生效。
- 所有 24 条 Composition 性能记录均缺少脚本内数值正确性断言；原 runner 明确标记 NOT_CHECKED_BY_RUNNER。不能据此声称这 24 个配置数值已验证。
- 8 GiB 补测的 12 条记录，优化路径的批间 CUDA IQR/median 约 0.7%–4.3%，generic 约 0.0%–0.9%；本次测量较一致，但不能替代独立重跑。
- 输入生成器使用 seed=1701+length；补测启动器未复现原启动器的全局 torch.manual_seed(42)，随机初始 State 不保证一致。因此不能将前后差异解释成仅调大 workspace 的性能影响。

## Stages 与 plans

- 三种 staged 前向共 48 条 Meta forward 均通过；相对 FLA recurrent 的 FLA/Meta 中位比值：LightNet 0.439×、preconditioned GDN 0.290×、preconditioned KDA 0.335×，表明 Meta staged public call 仍有性能差距。
- Meta backward：23 OK、1 NUMERICAL_MISMATCH、24 META_RUNTIME_REJECTED。LightNet 有一条 logits 梯度最大绝对误差 0.02734375，未通过既定容差。
- 24 条反向拒绝均为 workspace 资源限制；显式给出所需大小的记录最高约 10.34 GiB，所以单纯统一调到 8 GiB 未必足够。
- FLA preconditioned KDA chunk forward/backward 各 16 条 NUMERICAL_MISMATCH；不作有效性能对比。LightNet/GDN 部分 FLA backward 只有 MEASURED_UNVALIDATED。
- Mamba2 与 pressure 的 36 条 plans 记录均 OK，meta_correctness 均 VALID。

## 建议的处理顺序

1. 优先修复 Based/Rebased 的 State 和 continuation 正确性，以及 propagation level=1、LightNet 梯度问题；expression level 5/6 的边界误差单独分析。
2. 修复缓存和临时目录空间问题后，仅重跑受影响记录；对 nvcc host preprocessing、segfault 和 invalid image 保留分类，不能假设都是同一原因。
3. 为 Composition 增加独立 oracle / State / continuation 校验后，再把大加速比作为正确性已验证的结论。
4. 分析泛化 level=1 的 lowerer 切换、额外工作量和物化开销；针对 GDN/KDA/RWKV7/LightNet 做 public-call 路径性能分析。
5. 另行规划泛化和 backward 的资源补测，逐项记录实际需要、上限与分配峰值；当前报告不启动这些工作。

## 来源

原始文件均位于 SSH 远端 `benchmark_results/pr_matrix_full/`（gitignored）。

- matrix.json / tests.json / state.json / manifest.json
- compositions_workspace8g/run.json、status.json、consolidation.json 及对应作业的 records.json
- testing/benchmark_stateful_compositions.py（无正确性断言）、testing/benchmark_stateful_generalization.py（stable 定义）
- analysis_summary.json（本报告的机器可读统计）
