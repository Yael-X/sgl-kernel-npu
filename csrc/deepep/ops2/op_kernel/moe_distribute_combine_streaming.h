/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MOE_DISTRIBUTE_COMBINE_STREAMING_H
#define MOE_DISTRIBUTE_COMBINE_STREAMING_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_base.h"
#include "moe_distribute_combine_streaming_tiling.h"
#include "data_copy.h"

namespace MoeDistributeCombineStreamingImpl {

using namespace AscendC;

// Use template for data types to maintain flexibility
#define TemplateMC2TypeStreamingClass typename ExpandXType, typename ExpandIdxType
#define TemplateMC2TypeStreamingFunc ExpandXType, ExpandIdxType

template <TemplateMC2TypeStreamingClass>
class MoeDistributeCombineStreaming {
public:
    // Constants for Alignment and Architecture
    constexpr static uint32_t UB_ALIGN = 32U;
    constexpr static uint32_t CACHELINE_SIZE = 64U;
    constexpr static uint32_t SERVER_RANK_SIZE = 8U; // Intra-node rank size
    constexpr static uint32_t BLOCK_SIZE = 32U;
    
    // Ring Buffer Control Offsets (Relative to Base Address)
    // IPC Ring Buffer Layout: [Signals Area] [Data Area]
    // Signals: [8 Head Ptrs (8*64B)] [8 Tail Ptrs (8*64B)]
    constexpr static uint32_t IPC_SIGNAL_SIZE = 1024U; // 16 * 64B
    constexpr static uint32_t IPC_HEAD_OFFSET = 0U;
    constexpr static uint32_t IPC_TAIL_OFFSET = 512U; // 8 * 64
    constexpr static uint32_t IPC_DATA_OFFSET = 1024U;

    // RDMA Ring Buffer Layout
    // Signals: [N_Servers Head] [N_Servers Tail]
    // We reserve enough space for max servers (e.g. 64 servers -> 64*64*2 = 8KB)
    constexpr static uint32_t RDMA_SIGNAL_SIZE = 65536U; // 64KB safe margin
    constexpr static uint32_t RDMA_HEAD_OFFSET = 0U;
    
    // Batch Write Item Config
    constexpr static uint32_t BATCH_WRITE_ITEM_SIZE = 32U;

    __aicore__ inline MoeDistributeCombineStreaming() {}

    __aicore__ inline void Init(GM_ADDR expandX, GM_ADDR expandIdx, GM_ADDR sendCount, GM_ADDR scales, GM_ADDR XOut,
                                GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeDistributeCombineStreamingTilingData *tilingData,
                                __gm__ void *mc2InitTiling, __gm__ void *mc2CcTiling) {
        tpipe_ = pipe;
        
        // Input/Output Tensors
        expandXGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)expandX);
        expandIdxGlobal_.SetGlobalBuffer((__gm__ ExpandIdxType *)expandIdx);
        sendCountGlobal_.SetGlobalBuffer((__gm__ int32_t *)sendCount);
        expandScalesGlobal_.SetGlobalBuffer((__gm__ float *)scales);
        expandOutGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)XOut);
        
        // Tiling Parameters
        rankId_ = tilingData->moeDistributeCombineStreamingInfo.epRankId;
        worldSize_ = tilingData->moeDistributeCombineStreamingInfo.epWorldSize;
        serverNum_ = worldSize_ / SERVER_RANK_SIZE;
        serverId_ = rankId_ / SERVER_RANK_SIZE;
        localRankId_ = rankId_ % SERVER_RANK_SIZE;
        
        moeExpertNum_ = tilingData->moeDistributeCombineStreamingInfo.moeExpertNum;
        localMoeExpertNum_ = moeExpertNum_ / worldSize_;
        axisBS_ = tilingData->moeDistributeCombineStreamingInfo.bs;
        axisH_ = tilingData->moeDistributeCombineStreamingInfo.h;
        axisK_ = tilingData->moeDistributeCombineStreamingInfo.k;
        
        // Ring Buffer Config
        ipcChunkSize_ = tilingData->moeDistributeCombineStreamingInfo.ipcChunkSize;
        ipcQueueSize_ = tilingData->moeDistributeCombineStreamingInfo.ipcQueueSize;
        rdmaChunkSize_ = tilingData->moeDistributeCombineStreamingInfo.rdmaChunkSize;
        rdmaQueueSize_ = tilingData->moeDistributeCombineStreamingInfo.rdmaQueueSize;
        
        // Role Config
        senderCoreStart_ = tilingData->moeDistributeCombineStreamingInfo.senderCoreStart;
        senderCoreNum_ = tilingData->moeDistributeCombineStreamingInfo.senderCoreNum;
        forwarderCoreStart_ = tilingData->moeDistributeCombineStreamingInfo.forwarderCoreStart;
        forwarderCoreNum_ = tilingData->moeDistributeCombineStreamingInfo.forwarderCoreNum;
        receiverCoreStart_ = tilingData->moeDistributeCombineStreamingInfo.receiverCoreStart;
        receiverCoreNum_ = tilingData->moeDistributeCombineStreamingInfo.receiverCoreNum;
        
        // DeepEP Metadata
        recvRdmaChannelPrefixMatrix_ = tilingData->moeDistributeCombineStreamingInfo.recvRdmaChannelPrefixMatrix;
        recvRdmaRankPrefixSum_ = tilingData->moeDistributeCombineStreamingInfo.recvRdmaRankPrefixSum;
        recvGblChannelPrefixMatrix_ = tilingData->moeDistributeCombineStreamingInfo.recvGblChannelPrefixMatrix;
        recvGblRankPrefixSum_ = tilingData->moeDistributeCombineStreamingInfo.recvGblRankPrefixSum;
        
        // HCCL & Workspace
        hccl_.Init(AscendC::GetHcclContext<HCCL_GROUP_ID_0>(), mc2InitTiling);
        hccl_.SetCcTiling(mc2CcTiling);
        workspaceGlobal_.SetGlobalBuffer((__gm__ uint64_t *)workspaceGM);
        
        // GM Window setup for Ring Buffers
        // IPC Buffer is in windowIn (intra-node shared)
        windowInGM_ = hccl_.GetWindowsInAddr(rankId_);
        windowOutGM_ = hccl_.GetWindowsOutAddr(rankId_);
        
        // Setup base addresses for local pointers
        // IPC Ring: [8_Peers, Queue_Size] located at windowInGM + IPC_DATA_OFFSET
        // Signals located at windowInGM + IPC_HEAD/TAIL_OFFSET
        ipcRingBufferGM_.SetGlobalBuffer((__gm__ int8_t*)(windowInGM_ + IPC_DATA_OFFSET));
        
        // RDMA Ring: [N_Servers, Queue_Size] located at windowInGM + RDMA_DATA_OFFSET (Separate area)
        // Note: RDMA Ring usually placed after IPC Ring in the window
        uint64_t ipcTotalSize = IPC_DATA_OFFSET + (uint64_t)SERVER_RANK_SIZE * ipcQueueSize_;
        rdmaBaseOffset_ = (ipcTotalSize + 65536) / 65536 * 65536; // Align to 64KB
        rdmaRingBufferGM_.SetGlobalBuffer((__gm__ int8_t*)(windowInGM_ + rdmaBaseOffset_ + RDMA_SIGNAL_SIZE));
        
        coreIdx_ = GetBlockIdx();
        
        // Init UB Buffers
        tpipe_->InitBuffer(dataBuf_, 2 * ipcChunkSize_); // Double buffering size
        tpipe_->InitBuffer(sumBuf_, axisH_ * sizeof(float)); 
        tpipe_->InitBuffer(batchWriteBuf_, BATCH_WRITE_ITEM_SIZE * SERVER_RANK_SIZE);
    }

    __aicore__ inline void Process() {
        if (coreIdx_ >= senderCoreStart_ && coreIdx_ < senderCoreStart_ + senderCoreNum_) {
            RunSender();
        } else if (coreIdx_ >= forwarderCoreStart_ && coreIdx_ < forwarderCoreStart_ + forwarderCoreNum_) {
            RunForwarder();
        } else if (coreIdx_ >= receiverCoreStart_ && coreIdx_ < receiverCoreStart_ + receiverCoreNum_) {
            RunReceiver();
        }
        
        // Ensure all comms are done before exit (logical barrier handled by role completion)
        hccl_.Finalize();
    }

private:
    // -------------------------------------------------------------------------
    // Role Implementation: Sender (Intra-Node Gather -> IPC)
    // -------------------------------------------------------------------------
    __aicore__ inline void RunSender() {
        // Distribute work: 8 destination ranks, split among sender cores
        // Or simply iterate all if senderCoreNum == 1
        uint32_t tasksPerCore = (SERVER_RANK_SIZE + senderCoreNum_ - 1) / senderCoreNum_;
        uint32_t startRank = (coreIdx_ - senderCoreStart_) * tasksPerCore;
        uint32_t endRank = min(startRank + tasksPerCore, SERVER_RANK_SIZE);

        GlobalTensor<int32_t> prefixMatrix;
        prefixMatrix.SetGlobalBuffer((__gm__ int32_t*)recvGblChannelPrefixMatrix_);
        
        // Local cache for Tail pointers
        uint64_t localTails[SERVER_RANK_SIZE] = {0};

        for (uint32_t targetRank = startRank; targetRank < endRank; ++targetRank) {
            // Calculate IPC Ring Buffer address for this (Sender=Me, Target=targetRank) pair
            // The Forwarder on targetRank reads from 'MyRank' slot in its buffer.
            // Address: TargetWindow + IPC_DATA + (MyRank * QueueSize)
            // Wait, we write to OUR shared memory or THEIRS?
            // IPC usually implies shared GM in the same node. 
            // We write to windowIn of TargetRank.
            uint64_t targetWindowBase = hccl_.GetWindowsInAddr(rankId_ / SERVER_RANK_SIZE * SERVER_RANK_SIZE + targetRank);
            GlobalTensor<int8_t> targetRingBuffer;
            targetRingBuffer.SetGlobalBuffer((__gm__ int8_t*)(targetWindowBase + IPC_DATA_OFFSET + localRankId_ * ipcQueueSize_));
            
            // Pointers locations in Target Window
            GlobalTensor<uint64_t> targetHeadPtr;
            targetHeadPtr.SetGlobalBuffer((__gm__ uint64_t*)(targetWindowBase + IPC_HEAD_OFFSET + localRankId_ * sizeof(uint64_t)));
            GlobalTensor<uint64_t> targetTailPtr;
            targetTailPtr.SetGlobalBuffer((__gm__ uint64_t*)(targetWindowBase + IPC_TAIL_OFFSET + localRankId_ * sizeof(uint64_t)));

            // Gather Data
            for (uint32_t expertId = 0; expertId < localMoeExpertNum_; ++expertId) {
                // Get start offset and length for (Expert, TargetRank)
                // Matrix Layout assumed: [LocalExpert, WorldSize] or [LocalExpert, 8]?
                // DeepEP typically uses GblChannelPrefix to map to 'expandX'.
                // Assuming [expertId * worldSize + targetRank] gives start index in expandX
                // NOTE: This logic depends on exact matrix layout from Tiling.
                // Simplified here: Gather tokens for targetRank.
                
                // Assuming simple linear iteration for demo logic:
                // We need to implement the Gather -> UB -> Ring Write loop.
                // ... (Implementation of gather logic using DataCopy)
                
                // Example Chunk Write:
                uint32_t bytesToSend = axisH_ * sizeof(ExpandXType); // Dummy size
                
                WaitRingBufferSpace(targetHeadPtr, localTails[targetRank], bytesToSend, ipcQueueSize_);
                
                // Copy to Ring Buffer (Handle Wrap-around)
                WriteToRingBuffer(targetRingBuffer, dataBuf_, bytesToSend, localTails[targetRank], ipcQueueSize_);
                
                localTails[targetRank] += bytesToSend;
                
                // Update Tail in GM for Forwarder to see
                targetTailPtr.SetValue(0, localTails[targetRank]);
                DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(targetTailPtr);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Role Implementation: Forwarder (IPC -> Reduce -> RDMA)
    // -------------------------------------------------------------------------
    __aicore__ inline void RunForwarder() {
        // Forwarder aggregates data from 8 local ranks and sends to N remote servers.
        // It consumes from IPC Ring Buffers (MyWindow + IPC_DATA + SrcRank*Size)
        
        uint64_t localHeads[SERVER_RANK_SIZE] = {0}; // IPC Heads
        uint64_t rdmaTails[64] = {0}; // RDMA Tails (max 64 servers)
        
        // Loop until all data processed (Needs total count check in real impl)
        bool done = false; 
        while(!done) {
            // 1. Read from IPC Rings
            for (uint32_t srcRank = 0; srcRank < SERVER_RANK_SIZE; ++srcRank) {
                // Ptrs in My Window
                GlobalTensor<uint64_t> ipcTailPtr;
                ipcTailPtr.SetGlobalBuffer((__gm__ uint64_t*)(windowInGM_ + IPC_TAIL_OFFSET + srcRank * sizeof(uint64_t)));
                
                // Check if data available
                // WaitRingBufferData(ipcTailPtr, localHeads[srcRank], ChunkSize, ipcQueueSize_);
                // ... (Read & Reduce logic)
                
                // Update Head
                GlobalTensor<uint64_t> ipcHeadPtr;
                ipcHeadPtr.SetGlobalBuffer((__gm__ uint64_t*)(windowInGM_ + IPC_HEAD_OFFSET + srcRank * sizeof(uint64_t)));
                ipcHeadPtr.SetValue(0, localHeads[srcRank]);
                DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(ipcHeadPtr);
            }
            
            // 2. Write to RDMA Ring
            // Target Server S logic...
            uint32_t targetServer = 0; // Derived from data
            uint64_t remoteWindowBase = 0; // Needs calc via HCCL or lookup
            
            // Remote Ptrs
            GlobalTensor<uint64_t> remoteHeadPtr; // Need to RDMA Read this? Or Receiver writes it to us?
            // DeepEP assumption: Receiver writes Head to Forwarder's GM.
            // Let's assume Forwarder's WindowIn + RDMA_SIGNAL_OFFSET + serverId stores the Head from remote.
            
            // Write Data via BatchWrite or AIVRDMAPostSend
            // ...
            
            done = true; // Placeholder
        }
    }

    // -------------------------------------------------------------------------
    // Role Implementation: Receiver (RDMA -> Reduce -> Output)
    // -------------------------------------------------------------------------
    __aicore__ inline void RunReceiver() {
        // Receiver consumes from RDMA Ring Buffer (MyWindow + RDMA_DATA + SrcServer*Size)
        
        uint64_t localHeads[64] = {0}; // RDMA Heads for N servers
        
        bool done = false;
        while (!done) {
            for (uint32_t srcServer = 0; srcServer < serverNum_; ++srcServer) {
                // Check Tail (Written by Forwarder via RDMA)
                // My Window + RDMA_SIGNAL + TailOffset
                
                // Read Data
                
                // Reduce & Output
                
                // Send Head back to Forwarder (RDMA Write)
                // ...
            }
            done = true;
        }
    }

    // -------------------------------------------------------------------------
    // Ring Buffer Primitives
    // -------------------------------------------------------------------------
    
    // Producer Wait: Poll Head until (Tail - Head + WriteSize <= BufferSize)
    __aicore__ inline void WaitRingBufferSpace(GlobalTensor<uint64_t>& headPtrGM, uint64_t localTail, uint32_t writeSize, uint32_t bufferSize) {
        uint64_t head = 0;
        // Simple polling with backoff
        while (true) {
            // Invalidate cache to see updates from other cores/devices
            DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(headPtrGM);
            head = headPtrGM.GetValue(0);
            
            if (bufferSize - (localTail - head) >= writeSize) {
                break;
            }
            // Add slight delay/backoff
            PipeBarrier<PIPE_ALL>();
        }
    }

    // Consumer Wait: Poll Tail until (Tail - Head >= ReadSize)
    __aicore__ inline void WaitRingBufferData(GlobalTensor<uint64_t>& tailPtrGM, uint64_t localHead, uint32_t readSize, uint32_t bufferSize) {
        uint64_t tail = 0;
        while (true) {
            DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(tailPtrGM);
            tail = tailPtrGM.GetValue(0);
            
            if (tail - localHead >= readSize) {
                break;
            }
            PipeBarrier<PIPE_ALL>();
        }
    }

    // Write with Wrap-Around
    template <typename T>
    __aicore__ inline void WriteToRingBuffer(GlobalTensor<int8_t>& ringGM, TBuf<T>& srcBuf, uint32_t size, uint64_t tail, uint32_t bufferSize) {
        uint32_t offset = tail % bufferSize;
        uint32_t len1 = (offset + size > bufferSize) ? (bufferSize - offset) : size;
        uint32_t len2 = size - len1;
        
        LocalTensor<int8_t> data = srcBuf.Get<int8_t>();
        
        // First Part
        DataCopy(ringGM[offset], data, len1); // Needs alignment handling in real code
        
        // Second Part (Wrap)
        if (len2 > 0) {
            DataCopy(ringGM[0], data[len1], len2);
        }
        
        // Barrier to ensure data visibility before Tail update
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    
    // Helper: Align Up
    __aicore__ inline uint32_t AlignUp(uint32_t x, uint32_t align) {
        return (x + align - 1) & ~(align - 1);
    }

    // Members
    TPipe *tpipe_{nullptr};
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    
    GlobalTensor<ExpandXType> expandXGlobal_;
    GlobalTensor<ExpandIdxType> expandIdxGlobal_;
    GlobalTensor<int32_t> sendCountGlobal_;
    GlobalTensor<float> expandScalesGlobal_;
    GlobalTensor<ExpandXType> expandOutGlobal_;
    
    GlobalTensor<uint64_t> workspaceGlobal_;
    GlobalTensor<int8_t> ipcRingBufferGM_;
    GlobalTensor<int8_t> rdmaRingBufferGM_;
    
    TBuf<QuePosition::VECOUT> dataBuf_;
    TBuf<QuePosition::VECCALC> sumBuf_;
    TBuf<QuePosition::VECIN> batchWriteBuf_;
    
    uint64_t windowInGM_;
    uint64_t windowOutGM_;
    uint64_t rdmaBaseOffset_;
    
    // Metadata Ptrs
    uint64_t recvRdmaChannelPrefixMatrix_;
    uint64_t recvRdmaRankPrefixSum_;
    uint64_t recvGblChannelPrefixMatrix_;
    uint64_t recvGblRankPrefixSum_;
    
    // Config
    uint32_t rankId_, worldSize_, serverNum_, serverId_, localRankId_;
    uint32_t moeExpertNum_, localMoeExpertNum_;
    uint32_t axisBS_, axisH_, axisK_;
    uint32_t coreIdx_;
    
    uint32_t ipcChunkSize_, ipcQueueSize_;
    uint32_t rdmaChunkSize_, rdmaQueueSize_;
    
    uint32_t senderCoreStart_, senderCoreNum_;
    uint32_t forwarderCoreStart_, forwarderCoreNum_;
    uint32_t receiverCoreStart_, receiverCoreNum_;
};

} // namespace MoeDistributeCombineStreamingImpl

#endif // MOE_DISTRIBUTE_COMBINE_STREAMING_H