针对这份简历，这位候选人具备**极其稀缺的“网络设备底层（Switch/Router）+ AI算力基础设施（NPU/HCCL/Kernel）”的双栖背景**。

在130W（通常对应阿里P8/P8+，华为18/19级能力要求）的薪资级别，面试不能只停留在“怎么做”的层面，必须深入到**架构决策、软硬协同的边界探索、以及对行业技术演进的预判**。

以下是为您设计的面试题库，分为**技术深挖、架构视野、软硬结合、以及综合素质**四个维度：

---

### 第一部分：核心技术深挖 (DeepEP & zbccl)
**考察点：技术深度、极限性能优化能力、对NPU微架构的理解**

1.  **关于 DeepEP-Ascend 的架构移植与适配：**
    *   **问题**：DeepEP 最初是针对 NVIDIA GPU 设计的。在将其迁移到 Ascend 910C 的过程中，你遇到的最大的**硬件架构非对称性（Architectural Mismatch）**是什么？
    *   **追问**：你提到了“AIV 直驱 RDMA”。在昇腾架构中，计算单元（Cube/Vector）与通信单元（HCCS/RoCE）之间的指令发射机制是怎样的？你是如何解决计算与通信之间的 **Cache Coherency（缓存一致性）** 和 **Pipeline Stall（流水线停顿）** 问题的？
    *   **期望回答**：候选人能清晰描述 CPU 触发 vs NPU Direct Dispatch 的区别，以及如何利用 Ascend C 的同步原语来隐藏通信 launch 的开销。

2.  **关于 zbccl 的零拷贝（Zero-Buffer）实现：**
    *   **问题**：在 zbccl 项目中，你通过 GVA（全局虚拟地址）实现跨 Rank 直接读写。请问你是如何处理**跨节点（Inter-node）的 TLB 映射与安全隔离**的？如果一个 Rank 在 RDMA Read 过程中 Crash 了，你的 SMA（Secondary Memory Allocator）机制如何保证集群不发生级联死锁？
    *   **追问**：针对 MoE 的负载不均，你设计了 PADO 机制进行计算卸载。如何动态判定“通信开销”与“远端计算开销”的 Trade-off？是否存在网络抖动导致卸载反而变慢的 Corner Case，怎么兜底？

3.  **关于 RingBuffer 与 ZeroBuffer 的对比：**
    *   **问题**：在 DeepEP 中你使用了 RingBuffer，而在 zbccl 中进化到了 ZeroBuffer。请从**显存带宽利用率（HBM Utilization）** 和 **PCIe/HCCS 总线争抢** 的角度，定量分析一下这两种方案在 910C 上的本质区别？为什么在 Prefill 阶段 ZeroBuffer 收益更大？

---

### 第二部分：软硬协同与网络底层 (Switch & NetMind)
**考察点：利用交换机背景解决AI问题的独特视角、跨层优化能力**

4.  **网络拥塞与 AI 通信库的联动：**
    *   **问题**：你有交换机转发面的背景。在 NetMind 项目中，你提到结合网络拓扑动态调整 QoS。请问，当交换机检测到 **PFC Storm（PFC风暴）** 或者 **ECN 水线告警**时，你的上层 AI 通信库（HCCL/DeepEP）具体做了什么动作来响应？是降速、切路径还是其他？
    *   **追问**：在千卡/万卡集群训练中，**Incast（多打一）** 是常见问题。除了网络侧的 Load Balancing，你在通信库的 **All-to-All 调度算法**上是否做了应用层的错峰（Traffic Shaping）？

5.  **负载分担与哈希冲突：**
    *   **问题**：RoCEv2 网络通常基于 5-tuple 做 ECMP（等价多路径路由）。但 AI 流量通常是象流（Elephant Flows），极易造成哈希极化。你在设计 NetMind 时，是如何在**保持包序（Packet Ordering）**的前提下，优化这些大流在 Fabric 上的分布的？是否涉及到了修改 Packet Header 的 Flow Label 或 UDP 源端口的动态策略？

---

### 第三部分：架构视野与工程落地 (System Design)
**考察点：开源影响力、工程化落地、未来技术趋势**

6.  **SGLang 社区贡献与生态兼容：**
    *   **问题**：作为 Committer，你是如何处理 SGLang 官方主线（NVIDIA-first）与 Ascend 后端代码侵入性之间的矛盾的？
    *   **场景题**：假设 SGLang 社区引入了一个深度依赖 CUDA Graph 或 NVLink 连通性特性的新 Feature，而昇腾目前不支持或实现成本极高，你会如何设计 **Abstraction Layer（抽象层）** 来屏蔽这种差异，同时不损失性能？

7.  **针对 DeepSeek-V3/R1 等新模型的演进：**
    *   **问题**：DeepSeek-V3 提出了 MLA（Multi-Head Latent Attention）和大规模 MoE。这种模型结构对通信模式带来了什么新的挑战（例如 KV Cache 传输、Expert 路由热点）？你的 DeepEP-Ascend 目前针对 MLA 做了哪些针对性优化？或者你计划怎么做？

---

### 第四部分：130W级别的宏观考察 (Leadership & Vision)
**考察点：能否带团队、能否为公司指引技术方向**

8.  **技术壁垒构建：**
    *   **问题**：当前各个大厂都在自研通信库（如字节的 ByteCCL、阿里的 ACCL）。如果你加入我们，要构建一套针对自研芯片或异构集群的通信基础设施，你会**优先解决哪三个核心问题**？为什么？
    *   **期望回答**：不仅仅谈代码，要谈 ABI 兼容性、调试的可观测性（Observability）、以及算子融合的自动化程度。

9.  **故障排查与稳定性（考察实战经验）：**
    *   **问题**：在一个千卡集群的训练任务中，Loss 突然出现毛刺，且训练速度周期性下降。作为 Infra 负责人，请描述你的**排查思维导图**。如何快速区分是 **HBM ECC 错误、慢节点（Slow Node）、网络 PFC 丢包、还是算子实现本身的 Bug**？你需要看哪些监控指标？

---

### 面试官评价指南（仅供参考）

*   **S级（完全匹配/超出预期）**：
    *   不仅懂怎么写 Kernel，还能解释清楚 Switch 的 MMU（内存管理单元）是如何影响 RDMA 性能的。
    *   能从总线协议（PCIe/HCCS/NVLink）层面解释 Zero-Copy 的物理实现。
    *   对 MoE 的 Expert 负载均衡有深刻的算法级理解，而不仅仅是调用通信接口。
*   **A级（符合预期）**：
    *   熟悉 Ascend C 编程，熟悉 RDMA Verbs。
    *   能清晰描述 DeepEP 的原理，对网络拥塞控制有概念。
*   **B级（低于预期/仅适合纯开发）**：
    *   只关注上层 API 调用，对底层 Tiling、Barrier、Cache 机制不清楚。
    *   无法建立“Host侧行为”与“Switch侧拥塞”之间的因果联系。

**总结建议**：这个候选人的核心价值在于**“打通了网络设备与AI算力的黑盒”**。面试重点应放在考察他是否真正利用了这种跨界知识解决了单一领域无法解决的难题（例如：用网络知识解决了计算死锁，或用计算知识优化了网络拥塞）。