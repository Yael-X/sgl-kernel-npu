# 🎯 一、面试结构建议（130W级别）

建议 3~4 轮：

| 轮次          | 主题                      | 时长     | 目标                  |
| ------------- | ------------------------- | -------- | --------------------- |
| 技术深挖轮    | MoE通信 + RDMA + NPU      | 60–90min | 是否真做过内核级优化  |
| 系统架构轮    | DeepEP / zbccl / 融合算子 | 60min    | 架构能力              |
| 网络+AI融合轮 | 算网协同 + QoS            | 45min    | 跨栈能力              |
| 技术领导力轮  | 开源 + 影响力             | 45min    | 是否具备Principal潜力 |

------

# 🧠 二、Warm-up（快速验证真实性）

这些问题用来快速判断是不是“亲手做的”。

------

## Q1：你在 DeepEP-Ascend 里**最核心的性能瓶颈**是哪一个？

追问：

- 是 dispatch 还是 combine？
- 是 RDMA 延迟还是 HBM copy？
- 是 load imbalance 还是 queue backpressure？
- 你是怎么量化的？

### ✅ 真专家回答特征：

会说：

- token skew
- rank tail latency
- NIC WQE 深度
- CQ polling
- credit starvation
- AIV pipeline bubble

### ❌ 假专家：

> “主要是通信慢，我们做了优化”

------

# 🔬 三、DeepEP / MoE 通信专项深挖（核心轮）

------

## Q2：MoE Dispatch 的延迟构成分解一下

要求拆到：

```
token reorder
→ routing map
→ buffer pack
→ NIC send
→ remote unpack
→ expert queue
```

追问：

- 哪一步最容易被忽略？
- 哪一步最容易 cache miss？
- 哪一步最容易导致 tail？

------

## Q3：你为什么设计 RingBuffer + ZeroBuffer 两套？

这是**杀手题**。

追问：

- 两者 latency/throughput tradeoff？
- 在 EP32 / EP64 时谁更优？
- 在 burst token 场景谁更稳？
- backpressure 怎么传播？

------

## Q4：AIV 直驱 RDMA 的 pipeline 是什么？

要他说清：

```
AI Core → DMA → NIC queue → RDMA write → remote HBM
```

追问：

- 谁负责 doorbell？
- 谁管理 WQE？
- completion 怎么处理？
- 如果 CQ 满了会怎样？

------

## Q5：为什么 MoE combine 比 dispatch 更难优化？

正确方向：

- gather vs scatter
- remote offset unknown
- write amplification
- 原子 offset 分配

------

# ⚙️ 四、融合算子设计（FusedDeepMoe）

------

## Q6：Dispatch + GEMM + Combine 融合的最大挑战是什么？

追问：

- 内存布局如何兼容？
- expert GEMM shape 不一致怎么办？
- kernel launch 怎么减少？
- UB / shared buffer 如何规划？

------

## Q7：融合算子如何避免：

- register spill
- UB overflow
- pipeline stall

------

## Q8：为什么融合后 TPOT 能降 6ms？具体来自哪？

必须能拆到：

```
kernel launch ↓
HBM round-trip ↓
sync barrier ↓
NIC overlap ↑
```

------

# 🌐 五、RDMA + 网络深挖（必须问）

他网络背景很强，这轮可以拉开档次。

------

## Q9：RoCE 下 PFC 打开 vs 不开，对 MoE 有什么影响？

追问：

- head-of-line blocking
- PFC storm
- ECN vs PFC 取舍
- token skew 时谁更容易触发 pause？

------

## Q10：你怎么判断通信瓶颈在：

- NIC
- Switch buffer
- Host DMA
- PCIe
- HBM

------

## Q11：ECN 标记比例应该怎么调？

看他是否知道：

- incast
- RTT scale
- WRED curve

------

# 🧩 六、Zero-Buffer & zbccl 深挖（高级题）

------

## Q12：Zero-buffer 最大的风险是什么？

期待答案：

- remote HBM contention
- cache coherence
- remote page fault
- ordering issue
- memory fencing

------

## Q13：GVA 映射如何保证一致性？

追问：

- IPC handle 生命周期
- rank crash 怎么办？
- stale mapping 怎么清理？

------

## Q14：PADO 主动卸载机制：

请画流程（让他现场讲）

看点：

- 元数据预规划
- load threshold
- steal 策略
- fairness

------

# 🧠 七、系统设计题（必须有）

------

## Q15：让你设计一个 **MoE 推理通信系统（Ascend 版 NCCL）**

必须覆盖：

- 拓扑感知
- token skew
- rank imbalance
- credit flow control
- overlap

------

## Q16：如果 EP 从 16 → 128，你的设计哪块会崩？

看是否提到：

- metadata explosion
- routing table size
- CQ depth
- NIC QP 数量
- dispatch fanout

------

# 📊 八、性能工程能力

------

## Q17：你做性能分析的完整方法论？

必须包含：

```
trace
counter
timeline
micro-benchmark
roofline
```

------

## Q18：Prefill 和 Decode 通信模式差异？

必须说出：

- burst vs sparse
- steady vs spiky
- combine dominance vs dispatch dominance

------

# 👥 九、技术领导力（130W 必问）

------

## Q19：DeepEP 从 0 到社区接入，你推动了哪些关键决策？

看：

- API 稳定性
- 兼容 NVIDIA
- 生态策略
- 客户落地

------

## Q20：你如何决定一个优化“值得做”？

看是否提：

- perf gain / eng cost
- maintenance burden
- ecosystem impact

------

# 🚩 十、红旗信号（直接降级）

如果出现：

- 说不清 latency breakdown
- 说不清 RDMA pipeline
- 不知道 CQ / WQE
- 不知道 tail latency
- 不知道 token skew
- 说不清融合算子内存布局
- 所有优化都说“调参”

👉 直接不是 130W 级别

------

# ✅ 十一、加分信号（可直接上 Principal）

如果他能：

- 画出 dispatch pipeline
- 解释 NIC credit flow
- 讲清 GVA zero-buffer hazard
- 解释 tail latency 数学模型
- 给出调度公式
- 有 perf 数学模型

👉 顶级基础设施人才