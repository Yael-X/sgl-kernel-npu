这是一个非常宏大的重构工程。将 **DeepEP (DeepSeek Expert Parallelism)** 的 **Streaming + Ring Buffer** 思想移植到 **昇腾 910B** 上，能够彻底解决原代码中 `SyncAll` 带来的同步开销，并打破由于全量 Buffer 导致的显存和 Scale 限制。

以下是整合了你的设计材料与老代码的**完整重构方案**。

---

### 1. 核心架构设计 (Architecture)

我们将原有的 **分阶段 (Phased)** 执行模式重构为 **流水线 (Pipelined)** 模式。我们将 AI Cores 划分为三种角色（Role Partitioning）：

1.  **Sender Cores (Intra-Node Dispatch)**:
    *   **职责**: 从本地 `ExpandX` (乱序) 中 Gather 数据，拼接到 UB，通过 **Direct Memory Access (DMA)** 写入同一 Server 内目标卡 (Peer) 的 `IPC Ring Buffer`。
    *   **关键技术**: UB Staging (解决非对齐读写), Flow Control (检查 Peer Head).

2.  **Forwarder Cores (Inter-Node Dispatch)**:
    *   **职责**: 轮询本地 `IPC Ring Buffer` (等待 Sender)，在 UB 做第一次 Reduce (加权求和)，然后通过 **HCCL BatchWrite** 写入远端 Server 的 `RDMA Ring Buffer`。
    *   **关键技术**: IPC Polling, UB Reduce, RDMA Batching.

3.  **Receiver Cores (Final Combine)**:
    *   **职责**: 轮询 `RDMA Ring Buffer` (等待 Forwarder)，在 UB 做第二次 Reduce (如果有)，写入最终 `XOut`。
    *   **关键技术**: RDMA Polling, Final Output.

---

### 2. 数据结构与接口定义 (Data Structures)

#### 2.1 新增 Tiling 参数 (`MoeDistributeCombineStreamingTilingData`)
我们需要替换老的 Tiling 结构，引入 Ring Buffer 相关的尺寸定义和 DeepEP 特有的前缀和信息。

```cpp
struct MoeDistributeCombineStreamingTilingData {
    // 基础信息
    uint32_t numLocalExperts;
    uint32_t numRanks;      // Global World Size
    uint32_t serverSize;    // e.g., 8 for 910B
    uint32_t numServers;    // numRanks / serverSize
    uint32_t hiddenSize;
    uint32_t epRankId;
    
    // Ring Buffer 配置 (由 Tiling 计算得出最佳 Chunk 大小)
    uint32_t ipcChunkSize;  // IPC 环形缓冲区单个 Chunk 的大小 (Bytes)
    uint32_t ipcQueueSize;  // IPC 环形缓冲区的 Chunk 数量 (e.g., 4 or 8)
    uint32_t rdmaChunkSize; // RDMA 环形缓冲区单个 Chunk 的大小
    uint32_t rdmaQueueSize; // RDMA 队列深度
    
    // DeepEP 特有前缀和矩阵 (用于确定数据位置)
    // 形状通常为 [NumExperts, NumRanks] 或压缩格式
    // 这里传入 GM 上的偏移量
    uint64_t recvRdmaChannelPrefixMatrixOffset; 
    uint64_t recvRdmaRankPrefixSumOffset;
    uint64_t recvGblChannelPrefixMatrixOffset;
    uint64_t recvGblRankPrefixSumOffset;
    
    // 角色分配 (Core Mapping)
    uint32_t senderCoreStart, senderCoreNum;
    uint32_t forwarderCoreStart, forwarderCoreNum;
    uint32_t receiverCoreStart, receiverCoreNum;
};
```

#### 2.2 内存布局 (GM Memory Layout)
不再申请巨大的 `IPC_DATA_SIZE`，而是申请紧凑的 Ring Buffers。

```cpp
// 伪代码视图
struct SharedMemoryLayout {
    // 1. 同步信号区 (Cache Line Aligned)
    struct Signals {
        volatile int32_t ipc_tail[8]; // 本机写入进度 (Sender 写)
        volatile int32_t ipc_head[8]; // 本机处理进度 (Forwarder 写)
        volatile int32_t rdma_tail[MAX_SERVERS]; // RDMA 写入进度
        volatile int32_t rdma_head[MAX_SERVERS]; // RDMA 处理进度
    } signals;
    
    // 2. IPC Ring Buffers (8个 Peer 发给我的数据)
    // data[source_local_rank][queue_idx][chunk_size]
    char ipc_buffer[8][IPC_QUEUE_SIZE][IPC_CHUNK_SIZE];
    
    // 3. RDMA Ring Buffers (N个 Server 发给我的数据)
    char rdma_buffer[MAX_SERVERS][RDMA_QUEUE_SIZE][RDMA_CHUNK_SIZE];
};
```

---

### 3. 具体执行代码重构 (Implementation)

以下是重构后的类定义和核心逻辑。我保留了 Ascend C 的风格并融入了 DeepEP 的逻辑。

#### 3.1 类定义与辅助函数

```cpp
#ifndef MOE_DISTRIBUTE_COMBINE_STREAMING_H
#define MOE_DISTRIBUTE_COMBINE_STREAMING_H

#include "kernel_operator.h"
#include "hccl/hccl.h"

using namespace AscendC;

// 定义 Ring Buffer 状态
struct RingBufferState {
    int32_t headCache; // 本地缓存的消费者进度
    int32_t tailLocal; // 本地维护的生产者进度 (绝对值，不取模)
    GM_ADDR headPtrGM; // GM 上的 Head 指针地址 (远端更新)
    GM_ADDR tailPtrGM; // GM 上的 Tail 指针地址 (本地更新)
    uint32_t queueSize;
    uint32_t chunkSize;
    uint32_t bufferSize; // queueSize * chunkSize
};

template <typename T>
class MoeDistributeCombineStreaming {
public:
    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR outputX, GM_ADDR scales, 
                                GM_ADDR prefixSums, // 包含所有 DeepEP 前缀表
                                GM_ADDR workspace, TPipe* pipe, 
                                const MoeDistributeCombineStreamingTilingData* tiling) {
        // ... 初始化基础成员 ...
        tpipe_ = pipe;
        blkIdx_ = GetBlockIdx();
        
        // 解析 Tiling 参数
        ipcChunkSize_ = tiling->ipcChunkSize;
        rdmaChunkSize_ = tiling->rdmaChunkSize;
        // ...
        
        // 初始化 HCCL (用于 RDMA)
        hccl_.Init(contextGM); 
        
        // 计算 Core 角色
        if (blkIdx_ >= tiling->senderCoreStart && blkIdx_ < tiling->senderCoreStart + tiling->senderCoreNum) {
            role_ = Role::SENDER;
        } else if (blkIdx_ >= tiling->forwarderCoreStart && blkIdx_ < tiling->forwarderCoreStart + tiling->forwarderCoreNum) {
            role_ = Role::FORWARDER;
        } else if (blkIdx_ >= tiling->receiverCoreStart && blkIdx_ < tiling->receiverCoreStart + tiling->receiverCoreNum) {
            role_ = Role::RECEIVER;
        } else {
            role_ = Role::IDLE;
        }
        
        // 初始化 UB Buffer (Ping-Pong)
        tpipe_->InitBuffer(dataQueue_, 2, ipcChunkSize_); // 双缓冲
    }

    __aicore__ inline void Process() {
        if (role_ == Role::SENDER) {
            RunSender();
        } else if (role_ == Role::FORWARDER) {
            RunForwarder();
        } else if (role_ == Role::RECEIVER) {
            RunReceiver();
        }
        // 全局同步，确保所有数据落盘
        SyncAll<true>();
    }

private:
    enum class Role { SENDER, FORWARDER, RECEIVER, IDLE };
    Role role_;
    TPipe* tpipe_;
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    TQue<QuePosition::VECIN, 2> dataQueue_; // 用于搬运计算
    
    // ... 成员变量定义 ...

    // --- Helper: 等待环形缓冲区有空间 (Flow Control) ---
    __aicore__ inline void WaitSpace(RingBufferState& rb, int32_t neededBytes) {
        // 快速检查
        int32_t used = rb.tailLocal - rb.headCache;
        if (rb.bufferSize - used >= neededBytes) return;
        
        // 慢速轮询
        while (true) {
            // 刷新 Cache 读取 GM 上的 Head
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                GlobalTensor<int32_t>(rb.headPtrGM));
            rb.headCache = *(int32_t*)rb.headPtrGM; // 伪代码，实际需用 GlobalTensor
            
            used = rb.tailLocal - rb.headCache;
            if (rb.bufferSize - used >= neededBytes) break;
            
            PipeBarrier<PIPE_ALL>(); // 避免总线死锁
        }
    }

    // --- Helper: 等待环形缓冲区有数据 (Consumer) ---
    __aicore__ inline void WaitData(RingBufferState& rb, int32_t neededBytes) {
        // 逻辑类似 WaitSpace，只是检查 Tail > Head
        // 需注意: Consumer 维护 Head (Local), 读取 Tail (Remote GM)
    }
    
    // --- Role Impl ---
    __aicore__ inline void RunSender();
    __aicore__ inline void RunForwarder();
    __aicore__ inline void RunReceiver();
};
```

#### 3.2 关键角色实现逻辑

这是重构的核心，整合了你的设计材料中的 `Offset` 计算和流式处理。

##### Role 1: Sender (Gather -> UB -> IPC Ring Buffer)

这个 Kernel 取代了原来的 `AlltoAllDispatch`。它不进行 `Muls`，只负责搬运。

```cpp
template <typename T>
__aicore__ inline void MoeDistributeCombineStreaming<T>::RunSender() {
    // 1. 获取任务范围 (根据 Core ID 切分 Expert 或 Rank)
    // 假设当前 Core 负责发送给 TargetRank 的数据
    uint32_t targetRank = ...; 
    uint32_t localPeerId = targetRank % serverSize_; // 目标在节点内的 ID
    
    // 初始化目标 Ring Buffer 状态
    RingBufferState rb;
    rb.tailLocal = 0; 
    rb.headCache = 0;
    rb.bufferSize = ipcQueueSize_ * ipcChunkSize_;
    // 获取 Peer 的 IPC 窗口地址 (通过 HCCL GetWindowsInAddr 或预计算的 Global Addr)
    uint64_t peerBaseAddr = GetPeerIpcBaseAddr(localPeerId); 
    GM_ADDR peerDataBuffer = peerBaseAddr + IPC_DATA_OFFSET; 
    rb.headPtrGM = peerBaseAddr + SIGNAL_OFFSET + HEAD_INDEX * 4; // 远端 Head
    rb.tailPtrGM = peerBaseAddr + SIGNAL_OFFSET + TAIL_INDEX * 4; // 远端 Tail (我们要写的)

    LocalTensor<T> ubStaging = dataQueue_.AllocTensor<T>();
    int32_t stagingOffset = 0;

    // 2. 遍历 Expert，根据 Prefix Matrix Gather 数据
    for (int expId = 0; expId < numLocalExperts_; ++expId) {
        // 使用 DeepEP 的 Prefix Matrix 获取位置
        // start = recv_gbl_channel_prefix_matrix[...]
        // len = ...
        int32_t dataLen = GetLenFromMetadata(expId, targetRank);
        int32_t dataOffset = GetOffsetFromMetadata(expId, targetRank);
        
        if (dataLen == 0) continue;

        // 3. 搬运到 UB Staging (解决对齐问题)
        // 此处省略具体的对齐 Read 代码 (Over-read + UB Crop)
        CopyGmToUb(ubStaging, stagingOffset, srcGlobal + dataOffset, dataLen);
        stagingOffset += dataLen;

        // 4. Staging 满或结束时，写入 Peer IPC Buffer
        if (stagingOffset >= ipcChunkSize_ || IsLast(expId)) {
            // Flow Control: 等待 Peer 有空间
            WaitSpace(rb, stagingOffset);
            
            // 写入 (处理 Wrap-around)
            int32_t writeIdx = rb.tailLocal % rb.bufferSize;
            if (writeIdx + stagingOffset <= rb.bufferSize) {
                // 连续写
                DataCopy(peerDataBuffer + writeIdx, ubStaging, stagingOffset);
            } else {
                // 拆分写
                int32_t seg1 = rb.bufferSize - writeIdx;
                DataCopy(peerDataBuffer + writeIdx, ubStaging, seg1);
                DataCopy(peerDataBuffer, ubStaging[seg1], stagingOffset - seg1);
            }
            
            // 确保写入完成
            SetFlag(PIPE_MTE3, PIPE_S, 0);
            WaitFlag(PIPE_MTE3, PIPE_S, 0);

            // 更新 Tail (原子更新通知 Peer)
            rb.tailLocal += stagingOffset;
            GlobalTensor<int32_t> tailGt(rb.tailPtrGM);
            tailGt.SetValue(0, rb.tailLocal);
            
            stagingOffset = 0; // 重置 Staging
        }
    }
}
```

##### Role 2: Forwarder (IPC Reduce -> RDMA)

这个 Kernel 取代了 `SumToWindow` 和 `AlltoAllServerDispatch`。

```cpp
template <typename T>
__aicore__ inline void MoeDistributeCombineStreaming<T>::RunForwarder() {
    // 假设当前 Core 负责处理去往 RemoteServer 的数据
    // 需要轮询本节点内 8 张卡发来的 IPC Buffer
    
    // 初始化 RDMA Buffer 状态 (Destination)
    RingBufferState rdmaRb;
    // ... 配置 rdmaRb 连接到远端 Server ...
    
    // 循环处理流
    while (!IsAllFinished()) {
        // 1. 轮询 IPC Buffer (Gather from Peers)
        // 检查 8 个 Peer 的 Tail 指针，看是否有新数据到达
        // 这一步可以使用 bitmask 优化，DeepEP 中使用了 complex signal logic
        
        // 2. 读取数据到 UB 并聚合
        LocalTensor<float> accumUb = ...;
        Duplicate(accumUb, 0.0f); // 清零
        
        for (int i = 0; i < 8; ++i) {
             // 如果 Peer i 有数据，读入并 Add 到 accumUb
             // Accumulate(accumUb, peer_data[i]);
             // Update Peer i's Head pointer (release space)
        }
        
        // 3. 发送 RDMA (BatchWrite)
        // 等待远端 RDMA Buffer 有空间
        WaitSpace(rdmaRb, chunkSize);
        
        // 构造 BatchWriteInfo (UB -> Remote GM)
        // 注意：HCCL BatchWrite 通常是从 GM 到 GM，或者 UB 到 GM
        // 如果是从 UB 发，需要确保 BatchWrite 支持 UB 源地址
        // 如果不支持，需要先写回本地 GM Scratchpad，再发 RDMA
        hccl_.BatchWrite(...); 
        
        // 4. 更新 RDMA Tail 信号
        // 这通常需要单独发一个小包或者随数据附带
        UpdateRemoteTail(...);
    }
}
```

##### Role 3: Receiver (RDMA Reduce -> Output)

这个 Kernel 取代了 `SumToServer`。

```cpp
template <typename T>
__aicore__ inline void MoeDistributeCombineStreaming<T>::RunReceiver() {
    // 轮询来自所有 Server 的 RDMA Buffer
    
    while (!IsFinished()) {
        // 1. 等待数据 (Check Tail)
        WaitData(localRdmaRb, chunkSize);
        
        // 2. 读入 UB
        // DataCopy(ub, localRdmaRb.data + offset);
        
        // 3. 最终累加 (如果有多个 Server 发来重叠数据，或者加 Bias)
        // Add(ub, ...);
        
        // 4. 写出到 Global Output (XOut)
        // 根据 recv_rdma_rank_prefix_sum 计算最终写入位置
        // DataCopy(XOut + final_offset, ub);
        
        // 5. 释放 RDMA Buffer (更新 Head)
        UpdateLocalHead(localRdmaRb);
    }
}
```

---

### 4. 必要的重构步骤清单

请按照以下步骤执行重构：

1.  **修改 Tiling 代码 (Host 侧)**:
    *   计算 `recv_rdma_channel_prefix_matrix` 等 4 个 DeepEP 所需的元数据表。
    *   根据 `Total_Size` 和 `HBM_Limit` 计算合适的 `Chunk_Size` (建议 64KB - 512KB)。
    *   计算 `Sender/Forwarder/Receiver` 的 Block 分配策略。

2.  **重写 Kernel 入口 (`Process` 函数)**:
    *   移除所有 `if (coreIdx < ...)` 的隐式逻辑。
    *   改为 `switch (role)` 的显式逻辑。

3.  **实现 `RingBuffer` 管理类**:
    *   封装 `WaitSpace`, `WaitData`, `AdvanceTail`, `AdvanceHead`。
    *   这是保证代码可读性和避免死锁的关键。

4.  **替换通信原语**:
    *   **IPC**: 使用 `GM` 指针直接读写 (`DataCopy`) + `GlobalTensor` 原子更新标志位。
    *   **RDMA**: 使用 `HCCL` 的 `BatchWrite` (注意攒批，不要每 32B 发一次，至少攒够数 KB)。

5.  **内存对齐与 UB 优化**:
    *   Sender 阶段必须引入 UB Staging Buffer，因为 Gather 来的数据长度不一，直接写 GM 会导致严重的未对齐性能下降。

### 5. 总结

这个方案将原来的同步式、大块内存的实现，彻底转变为基于 **DeepEP 理念的流式环形缓冲区实现**。

*   **限制解决**: 环形缓冲区不再要求能存下所有数据，解决了大 BS 下的内存墙。
*   **多机扩展**: 通过 `Forwarder` 角色专门负责 RDMA，且支持 N 个 Server，解决了双机硬编码限制。
*   **性能提升**: 通信（DMA/RDMA）与计算（Gather/Reduce）在流水线中完全重叠。

**下一步行动**: 请根据上述代码框架，填充具体的 `DataCopy` 参数和 `Wait` 逻辑细节。由于代码量较大，建议先在一个 Kernel 文件中实现 `Sender` 逻辑进行单元测试，验证 UB Staging 和 IPC 写入的正确性。