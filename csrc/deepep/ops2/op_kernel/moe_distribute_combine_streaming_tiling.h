/*!
 * \file moe_distribute_combine_streaming_tiling.h
 * \brief Tiling data structure for MoE Distribute Combine Streaming Pipeline
 */

#ifndef MOE_DISTRIBUTE_COMBINE_STREAMING_TILING_H
#define MOE_DISTRIBUTE_COMBINE_STREAMING_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_combine_a2_tiling.h"

namespace MoeDistributeCombineStreamingImpl {

struct MoeDistributeCombineStreamingInfo {
    // Inherited from original A2 info
    uint32_t epWorldSize;          // epWorldSize
    uint32_t tpWorldSize;          // tpWorldSize
    uint32_t epRankId;             // epRankId
    uint32_t tpRankId;             // tpRankId
    uint32_t expertSharedType;     // expert type
    uint32_t sharedExpertRankNum;  // shared expert number
    uint32_t moeExpertNum;         // moe expert number
    uint32_t globalBs;             // globalBs = BS * worldSize
    uint32_t bs;                   // bs
    uint32_t k;                    // k
    uint32_t h;                    // h
    uint32_t aivNum;               // aivNum
    uint64_t totalUbSize;          // total UB size
    uint32_t hcclBufferSize;       // HCCL windows, unit:B
    uint32_t rsd;                  // reserved field

    // Ring Buffer configuration
    uint32_t ipcChunkSize;         // Size of each chunk in IPC Ring Buffer (bytes)
    uint32_t ipcQueueSize;         // Number of chunks in IPC Ring Buffer queue
    uint32_t rdmaChunkSize;        // Size of each chunk in RDMA Ring Buffer (bytes)
    uint32_t rdmaQueueSize;        // Number of chunks in RDMA Ring Buffer queue

    // Role allocation for AI Cores
    uint32_t senderCoreStart;      // Starting core index for Sender role
    uint32_t senderCoreNum;        // Number of cores for Sender role
    uint32_t forwarderCoreStart;   // Starting core index for Forwarder role
    uint32_t forwarderCoreNum;     // Number of cores for Forwarder role
    uint32_t receiverCoreStart;    // Starting core index for Receiver role
    uint32_t receiverCoreNum;      // Number of cores for Receiver role
};

struct MoeDistributeCombineStreamingTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MoeDistributeCombineStreamingInfo moeDistributeCombineStreamingInfo;
};

}  // namespace MoeDistributeCombineStreamingImpl

#endif  // MOE_DISTRIBUTE_COMBINE_STREAMING_TILING_H