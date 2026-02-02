# DeepEP 通信模式深度解析：分发 (Scenario A) 与 聚合 (Scenario B)

## 1. 核心架构定义与对比

在 DeepEP（DeepSeek 通信库）中，为了适配大规模 MoE 模型在 H100/A100 集群上的 **Rail-Only（同号卡物理直连）** 拓扑限制，通信路径被严格设计为两步走。

区别在于：**“跨卡数据搬运（Cross-Rank Shuffle）”是发生在发送端还是接收端。**

| 特性 | **Scenario A: 接收端分发** (Receiver-side Distribution) | **Scenario B: 发送端聚合** (Sender-side Aggregation) |
| :--- | :--- | :--- |
| **路径** | Source $\xrightarrow{RDMA}$ **Remote Fwd** $\xrightarrow{IPC}$ Dest | Source $\xrightarrow{IPC}$ **Local Fwd** $\xrightarrow{RDMA}$ Dest |
| **应用阶段** | **Decode (解码阶段)** | **Prefill (预填充阶段)** |
| **关键资源** | **延迟敏感 (Latency Sensitive)** | **带宽敏感 (Bandwidth Sensitive)** |
| **Buffer 状态** | **充足** (Sufficient, 全量预留) | **不足** (Insufficient, 需流式切分) |
| **硬件利用** | **IBGDA** (GPU Direct Async) 发射小包 | **NVLink** 聚合 + **RDMA** 大包流式传输 |

---

## 2. 深度剖析：Scenario A (Decode 阶段)

### 2.1 工作流程
1.  **发送 (Source)**: 源 GPU 计算完 Token，通过 **IBGDA** 直接向**目标机器的同号网关卡 (Remote Forwarder)** 的预留 Buffer 发起 RDMA Write。
    *   *关键点：源端不等待本地邻居，算一个发一个 (Fire-and-Forget)。*
2.  **中转 (Remote Forwarder)**: 目标机器的网关卡在 HBM 中收到数据。
3.  **分发 (IPC)**: 目标机器的网关卡通过轮询发现数据到达，利用 NVLink (IPC) 将数据搬运到同一节点内的**最终目的卡 (Destination)**。

### 2.2 为什么 Decode 选择 Scenario A？

*   **Buffer 充足带来的“免流控”优势**：
    *   在 Decode 阶段，Token 数据量小（KB级别）。DeepEP 可以在显存中为所有可能的 Source Rank **预先分配好专属槽位 (Slots)**。
    *   因为 Buffer 足够大，发送端不需要询问接收端“有没有空”，直接写即可。这消除了往返协商延迟 (RTT)。

*   **极致的低延迟 (Latency Hiding)**：
    *   **源端零同步**：如果采用聚合（Scenario B），源端 GPU 必须等待本地其他 7 张卡都准备好数据才能“打包”发送。Decode 阶段计算极快，这种等待（Barrier）会浪费宝贵的计算时间。
    *   **Scenario A 允许源端“甩手”**：源端把数据扔给 RDMA 后，立即开始计算下一层的 Attention 或路由，不用管后续 IPC 怎么搬运。

*   **IBGDA 解决小包瓶颈**：
    *   传统观点认为 Scenario A 会产生小包风暴。但利用 H100 的 IBGDA 技术，GPU 线程直接控制网卡发包，绕过了 CPU 开销，使得高 PPS (Packet Per Second) 成为可能。

### 2.3 计算通信掩盖 (Overlap) 收益
*   **掩盖逻辑**：`Layer N+1 的计算` 掩盖 `Layer N 的 RDMA + IPC`。
*   **收益点**：由于没有本地聚合的 Barrier，通信流水线的启动时间（Kick-off time）最早。只要网络传输时间 < 计算时间，通信延迟就被完全隐藏。

---

## 3. 深度剖析：Scenario B (Prefill 阶段)

### 3.1 工作流程
1.  **聚合 (Source $\to$ Local Forwarder)**: 源节点内的所有 GPU (Source) 将发往**同一个目标节点**的数据，通过 **NVLink (IPC)** 写入到本机负责该方向的网关卡 (Local Forwarder) 的 Buffer 中。
2.  **发送 (RDMA)**: 本机网关卡一旦凑齐了一个 Chunk（数据块），就通过 **RDMA** 发送给目标机器的同号网关卡。
3.  **直落 (Destination)**: 此时，RDMA 携带的是**纯净的、去往该目标卡的数据**（或者在接收端做极少量的简单分发）。

### 3.2 为什么 Prefill 选择 Scenario B？

*   **Buffer 不足引发的“流式传输”需求**：
    *   Prefill 阶段数据量巨大（GB级别），远远超过预留的通信 Buffer（通常仅几百 MB）。因此，通信必须是**流式（Streaming）**的，发完一块清空一块。
    *   **如果用 Scenario A**：接收端网关卡 (B:0) 接收了来自 A:0 的混合数据。B:0 必须暂停接收，先通过 IPC 把数据分发给 B:1, B:2... 以腾空 Buffer。这导致 **RDMA 链路频繁闲置等待 IPC**，带宽利用率雪崩。
    *   **采用 Scenario B**：发送端 A:0, A:1... 先把发往 B:0 的数据在本地聚合好。A:0 向 B:0 发送的是**连续的、无需分拣的数据流**。B:0 收到后直接消费或存入 HBM，**RDMA 链路可以一直处于满载状态**。

*   **利用 NVLink 换取 RDMA 效率**：
    *   节点内带宽 (NVLink ~900GB/s) 远大于 跨节点带宽 (RDMA ~50GB/s)。
    *   Scenario B 的本质是：**用廉价且快速的 NVLink 时间，在发送端就把数据整理好**，确保那条昂贵且慢速的 RDMA 管道里跑的每一比特都是有效载荷，且是大块连续传输 (High TLP Efficiency)。

### 3.3 计算通信掩盖 (Overlap) 收益
*   **掩盖逻辑**：多级流水线。
    *   Stage 1: GPU Kernel 计算 (生成 Token)。
    *   Stage 2: GPU Kernel 写入本地 IPC Buffer (掩盖下一块的计算)。
    *   Stage 3: Forwarder 发送 RDMA (掩盖下一块的 IPC)。
*   **收益点**：通过发送端聚合，将碎片化的内存访问（Scatter）转换成了连续的 DMA 传输。虽然引入了 IPC 拷贝延迟，但在 Prefill 的长耗时计算掩护下，这部分延迟可被忽略，换来的是 **RDMA 带宽的 100% 打满**。

---

## 4. 总结：决策树与关键差异

| 维度 | **Decode (Scenario A)** | **Prefill (Scenario B)** |
| :--- | :--- | :--- |
| **根本矛盾** | **快**：我想发，但我不想等邻居。 | **多**：太大了，不仅要分批，还不能堵车。 |
| **Buffer 策略** | **Static Allocation**<br>每人都有固定车位，直接停。 | **Streaming / Chunking**<br>车位不够，必须排队进出，动作要快。 |
| **分拣压力承担者** | **接收端 (Receiver)**<br>利用接收端等待数据的空闲时间做分发。 | **发送端 (Sender)**<br>利用超快的 NVLink 先整理，喂饱 RDMA。 |
| **网络包特征** | **小包、高频**<br>依赖 IBGDA 硬件加速。 | **大包、连续**<br>依赖 PCIe Burst 传输。 |
| **同步机制** | **无 (Async)**<br>Source 独立运作。 | **有 (Barrier)**<br>Source 需协同写入聚合 Buffer。 |

### 5. 一句话总结 DeepEP 的设计智慧

DeepEP 的核心洞察在于：
*   在 **Decode** 阶段，**延迟是瓶颈**，且 Buffer 够用，因此选择 **Scenario A** 以解耦发送端，利用 IBGDA 实现“乱序直发”。
*   在 **Prefill** 阶段，**带宽是瓶颈**，且 Buffer 不够，因此选择 **Scenario B** 利用节点内高带宽 (NVLink) 进行流量整形 (Traffic Shaping)，以换取跨节点链路 (RDMA) 的最高吞吐稳定性。