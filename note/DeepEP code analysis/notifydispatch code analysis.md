
struct SourceMeta {
    int src_rdma_rank, is_token_in_nvl_rank_bits;

    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8, "Invalid number of maximum NVL peers");

    __forceinline__ SourceMeta() = default;

    // TODO: faster encoding
    __device__ __forceinline__ SourceMeta(int rdma_rank, const bool* is_token_in_nvl_ranks) {
        src_rdma_rank = rdma_rank;
        is_token_in_nvl_rank_bits = is_token_in_nvl_ranks[0];
        #pragma unroll
        for (int i = 1; i < NUM_MAX_NVL_PEERS; ++i)
            is_token_in_nvl_rank_bits |= is_token_in_nvl_ranks[i] << i;
    }

    __device__ __forceinline__ bool is_token_in_nvl_rank(int nvl_rank) const { return (is_token_in_nvl_rank_bits >> nvl_rank) & 1; }
};

EP_STATIC_ASSERT(sizeof(SourceMeta) % sizeof(int) == 0, "Invalid size of `SourceMeta`");

int get_source_meta_bytes() {
    return sizeof(SourceMeta);
}

__host__ __device__ __forceinline__ int get_num_bytes_per_token(int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights) {
    return static_cast<int>(align_up(hidden_int4 * sizeof(int4) + sizeof(SourceMeta) + num_scales * sizeof(float) +
                                         num_topk_idx * sizeof(int) + num_topk_weights * sizeof(float),
                                     sizeof(int4)));
}

__host__ __device__ __forceinline__ std::pair<int, int> get_rdma_clean_meta(int hidden_int4,
                                                                            int num_scales,
                                                                            int num_topk_idx,
                                                                            int num_topk_weights,
                                                                            int num_rdma_ranks,
                                                                            int num_rdma_recv_buffer_tokens,
                                                                            int num_channels) {
    // Return `int32_t` offset and count to clean
    return {(get_num_bytes_per_token(hidden_int4, num_scales, num_topk_idx, num_topk_weights) * num_rdma_recv_buffer_tokens *
             num_rdma_ranks * 2 * num_channels) /
                sizeof(int),
            (NUM_MAX_NVL_PEERS * 2 + 4) * num_rdma_ranks * 2 * num_channels};
}

__host__ __device__ __forceinline__ std::pair<int, int> get_nvl_clean_meta(int hidden_int4,
                                                                           int num_scales,
                                                                           int num_topk_idx,
                                                                           int num_topk_weights,
                                                                           int num_rdma_ranks,
                                                                           int num_nvl_ranks,
                                                                           int num_nvl_recv_buffer_tokens,
                                                                           int num_channels,
                                                                           bool is_dispatch) {
    // Return `int32_t` offset and to clean
    EP_STATIC_ASSERT(sizeof(SourceMeta) % sizeof(int) == 0, "Invalid size of `SourceMeta`");

    return {
        (num_nvl_recv_buffer_tokens * get_num_bytes_per_token(hidden_int4, num_scales, num_topk_idx, num_topk_weights) * num_nvl_ranks *
         num_channels) /
            sizeof(int),
        num_nvl_ranks * (2 * num_rdma_ranks + 2) * num_channels,
    };
}

template <bool kLowLatencyMode>
__forceinline__ __device__ int translate_dst_rdma_rank(const int dst_rdma_rank, const int nvl_rank) {
    return kLowLatencyMode ? (dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank) : dst_rdma_rank;
}

template <bool kLowLatencyMode>
__forceinline__ __device__ void nvshmem_sync_with_same_gpu_idx(const nvshmem_team_t& rdma_team) {
    kLowLatencyMode ? void(nvshmem_sync(rdma_team)) : nvshmem_sync_all();
}

template <bool kLowLatencyMode, int kNumRDMARanks>
__global__ void notify_dispatch(const int* num_tokens_per_rank,
                                int* moe_recv_counter_mapped,
                                int num_ranks,
                                const int* num_tokens_per_rdma_rank,
                                int* moe_recv_rdma_counter_mapped,
                                const int* num_tokens_per_expert,
                                int* moe_recv_expert_counter_mapped,
                                int num_experts,
                                const bool* is_token_in_rank,
                                int num_tokens,
                                int num_worst_tokens,
                                int num_channels,
                                int expert_alignment,
                                const int rdma_clean_offset,
                                const int rdma_num_int_clean,
                                const int nvl_clean_offset,
                                const int nvl_num_int_clean,
                                int* rdma_channel_prefix_matrix,
                                int* recv_rdma_rank_prefix_sum,
                                int* gbl_channel_prefix_matrix,
                                int* recv_gbl_rank_prefix_sum,
                                void* rdma_buffer_ptr,
                                void** buffer_ptrs,
                                int** barrier_signal_ptrs,
                                int rank,
                                const nvshmem_team_t rdma_team) {
    auto sm_id = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x), warp_id = thread_id / 32, lane_id = get_lane_id();
    auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;

    auto rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
    auto num_rdma_experts = num_experts / kNumRDMARanks, num_nvl_experts = num_rdma_experts / NUM_MAX_NVL_PEERS;

    if (sm_id == 0) {
        // Communication with others
        // Global barrier: the first warp does intra-node sync, the second warp does internode sync
        EP_DEVICE_ASSERT(num_warps > 1);
        EP_DEVICE_ASSERT(kNumRDMARanks <= num_threads);

        // waiting for all previous inflight wrs to complete,
        // in case of rewriting cleared rdma_buffer
        auto qps_per_rdma_rank = ibgda_get_state()->num_rc_per_pe * ibgda_get_state()->num_devices_initialized;
        for (int i = thread_id; i < qps_per_rdma_rank * (kNumRDMARanks - 1); i += num_threads) {
            auto dst_rdma_rank = (i / qps_per_rdma_rank + rdma_rank + 1) % kNumRDMARanks;
            auto qp_id = i % qps_per_rdma_rank;
            nvshmemi_ibgda_quiet(translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), qp_id);
        }
        __syncthreads();

        if (thread_id == 32)
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
        barrier_block<NUM_MAX_NVL_PEERS, true>(barrier_signal_ptrs, nvl_rank);

        // Send numbers of tokens per rank/expert to RDMA ranks
        auto rdma_buffer_ptr_int = static_cast<int*>(rdma_buffer_ptr);
        auto rdma_recv_num_tokens_mixed = SymBuffer<int>(rdma_buffer_ptr, NUM_MAX_NVL_PEERS + num_rdma_experts + 1, kNumRDMARanks);

        // Clean up for later data dispatch
        EP_DEVICE_ASSERT(rdma_recv_num_tokens_mixed.total_bytes <= rdma_clean_offset * sizeof(int));
        #pragma unroll
        for (int i = thread_id; i < rdma_num_int_clean; i += num_threads)
            rdma_buffer_ptr_int[rdma_clean_offset + i] = 0;

        // Copy to send buffer
        #pragma unroll
        for (int i = thread_id; i < num_ranks; i += num_threads)
            rdma_recv_num_tokens_mixed.send_buffer(i / NUM_MAX_NVL_PEERS)[i % NUM_MAX_NVL_PEERS] = num_tokens_per_rank[i];
        #pragma unroll
        for (int i = thread_id; i < num_experts; i += num_threads)
            rdma_recv_num_tokens_mixed.send_buffer(i / num_rdma_experts)[NUM_MAX_NVL_PEERS + i % num_rdma_experts] =
                num_tokens_per_expert[i];
        if (thread_id < kNumRDMARanks)
            rdma_recv_num_tokens_mixed.send_buffer(thread_id)[NUM_MAX_NVL_PEERS + num_rdma_experts] = num_tokens_per_rdma_rank[thread_id];
        __syncthreads();

        // Issue send
        // TODO: more light fence or barrier or signaling
        // TODO: overlap EP barrier and NVL cleaning
        for (int i = warp_id; i < kNumRDMARanks; i += num_warps) {
            if (i != rdma_rank) {
                nvshmemi_ibgda_put_nbi_warp<true>(reinterpret_cast<uint64_t>(rdma_recv_num_tokens_mixed.recv_buffer(rdma_rank)),
                                                  reinterpret_cast<uint64_t>(rdma_recv_num_tokens_mixed.send_buffer(i)),
                                                  (NUM_MAX_NVL_PEERS + num_rdma_experts + 1) * sizeof(int),
                                                  translate_dst_rdma_rank<kLowLatencyMode>(i, nvl_rank),
                                                  0,
                                                  lane_id,
                                                  0);
            } else {
                UNROLLED_WARP_COPY(1,
                                   lane_id,
                                   NUM_MAX_NVL_PEERS + num_rdma_experts + 1,
                                   rdma_recv_num_tokens_mixed.recv_buffer(rdma_rank),
                                   rdma_recv_num_tokens_mixed.send_buffer(i),
                                   ld_volatile_global,
                                   st_na_global);
            }
        }
        __syncthreads();

        // Wait previous operations to be finished
        if (thread_id < kNumRDMARanks and thread_id != rdma_rank)
            nvshmemi_ibgda_quiet(translate_dst_rdma_rank<kLowLatencyMode>(thread_id, nvl_rank), 0);
        __syncthreads();

        // Barrier
        if (thread_id == 0)
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
        __syncthreads();

        // NVL buffers
        auto nvl_send_buffer = thread_id < NUM_MAX_NVL_PEERS ? buffer_ptrs[thread_id] : nullptr;
        auto nvl_recv_buffer = buffer_ptrs[nvl_rank];
        auto nvl_reduced_num_tokens_per_expert = Buffer<int>(nvl_recv_buffer, num_rdma_experts).advance_also(nvl_send_buffer);
        auto nvl_send_num_tokens_per_rank = AsymBuffer<int>(nvl_send_buffer, kNumRDMARanks, NUM_MAX_NVL_PEERS);
        auto nvl_send_num_tokens_per_expert = AsymBuffer<int>(nvl_send_buffer, num_nvl_experts, NUM_MAX_NVL_PEERS);
        auto nvl_recv_num_tokens_per_rank = AsymBuffer<int>(nvl_recv_buffer, kNumRDMARanks, NUM_MAX_NVL_PEERS);
        auto nvl_recv_num_tokens_per_expert = AsymBuffer<int>(nvl_recv_buffer, num_nvl_experts, NUM_MAX_NVL_PEERS);

        // Clean up for later data dispatch
        auto nvl_buffer_ptr_int = static_cast<int*>(buffer_ptrs[nvl_rank]);
        EP_DEVICE_ASSERT(nvl_reduced_num_tokens_per_expert.total_bytes + nvl_send_num_tokens_per_rank.total_bytes +
                             nvl_send_num_tokens_per_expert.total_bytes <=
                         nvl_clean_offset * sizeof(int));
        #pragma unroll
        for (int i = thread_id; i < nvl_num_int_clean; i += num_threads)
            nvl_buffer_ptr_int[nvl_clean_offset + i] = 0;

        // Reduce number of tokens per expert into the NVL send buffer
        // TODO: may use NVSHMEM reduction
        EP_DEVICE_ASSERT(num_rdma_experts <= num_threads);
        if (thread_id < num_rdma_experts) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumRDMARanks; ++i)
                sum += rdma_recv_num_tokens_mixed.recv_buffer(i)[NUM_MAX_NVL_PEERS + thread_id];
            nvl_reduced_num_tokens_per_expert[thread_id] = sum;
        }
        __syncthreads();

        // Reduce RDMA received tokens
        if (thread_id == 0) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumRDMARanks; ++i) {
                sum += rdma_recv_num_tokens_mixed.recv_buffer(i)[NUM_MAX_NVL_PEERS + num_rdma_experts];
                recv_rdma_rank_prefix_sum[i] = sum;
            }
            if (num_worst_tokens == 0) {
                while (ld_volatile_global(moe_recv_rdma_counter_mapped) != -1)
                    ;
                *moe_recv_rdma_counter_mapped = sum;
            }
        }

        // Send numbers of tokens per rank/expert to NVL ranks
        EP_DEVICE_ASSERT(NUM_MAX_NVL_PEERS <= num_threads);
        if (thread_id < NUM_MAX_NVL_PEERS) {
            #pragma unroll
            for (int i = 0; i < kNumRDMARanks; ++i)
                nvl_send_num_tokens_per_rank.buffer(nvl_rank)[i] = rdma_recv_num_tokens_mixed.recv_buffer(i)[thread_id];
            #pragma unroll
            for (int i = 0; i < num_nvl_experts; ++i)
                nvl_send_num_tokens_per_expert.buffer(nvl_rank)[i] = nvl_reduced_num_tokens_per_expert[thread_id * num_nvl_experts + i];
        }
        barrier_block<NUM_MAX_NVL_PEERS>(barrier_signal_ptrs, nvl_rank);

        // Reduce the number of tokens per rank/expert
        EP_DEVICE_ASSERT(num_nvl_experts <= num_threads);
        if (thread_id == 0) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < num_ranks; ++i) {
                int src_rdma_rank = i / NUM_MAX_NVL_PEERS, src_nvl_rank = i % NUM_MAX_NVL_PEERS;
                sum += nvl_recv_num_tokens_per_rank.buffer(src_nvl_rank)[src_rdma_rank];
                recv_gbl_rank_prefix_sum[i] = sum;
            }
            if (num_worst_tokens == 0) {
                while (ld_volatile_global(moe_recv_counter_mapped) != -1)
                    ;
                *moe_recv_counter_mapped = sum;
            }
        }
        if (thread_id < num_nvl_experts) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
                sum += nvl_recv_num_tokens_per_expert.buffer(i)[thread_id];
            sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment;
            if (num_worst_tokens == 0) {
                while (ld_volatile_global(moe_recv_expert_counter_mapped + thread_id) != -1)
                    ;
                moe_recv_expert_counter_mapped[thread_id] = sum;
            }
        }

        // Finally barrier
        if (thread_id == 32)
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
        barrier_block<NUM_MAX_NVL_PEERS>(barrier_signal_ptrs, nvl_rank);
    } else {
        // Calculate meta data
        int dst_rdma_rank = sm_id - 1;
        for (int channel_id = warp_id; channel_id < num_channels; channel_id += num_warps) {
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

            // Iterate over tokens
            int total_count = 0, per_nvl_rank_count[NUM_MAX_NVL_PEERS] = {0};
            for (int64_t i = token_start_idx + lane_id; i < token_end_idx; i += 32) {
                EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(uint64_t), "Invalid number of NVL peers");
                auto is_token_in_rank_uint64 =
                    *reinterpret_cast<const uint64_t*>(is_token_in_rank + i * num_ranks + dst_rdma_rank * NUM_MAX_NVL_PEERS);
                auto is_token_in_rank_values = reinterpret_cast<const bool*>(&is_token_in_rank_uint64);
                #pragma unroll
                for (int j = 0; j < NUM_MAX_NVL_PEERS; ++j)
                    per_nvl_rank_count[j] += is_token_in_rank_values[j];
                total_count += (is_token_in_rank_uint64 != 0);
            }

            // Warp reduce
            total_count = warp_reduce_sum(total_count);
            #pragma unroll
            for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
                per_nvl_rank_count[i] = warp_reduce_sum(per_nvl_rank_count[i]);

            // Write into channel matrix
            if (elect_one_sync()) {
                #pragma unroll
                for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
                    gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + i) * num_channels + channel_id] = per_nvl_rank_count[i];
                rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id] = total_count;
            }
        }

        // Calculate prefix sum
        __syncthreads();
        if (thread_id == 0) {
            auto prefix_row = rdma_channel_prefix_matrix + dst_rdma_rank * num_channels;
            #pragma unroll
            for (int i = 1; i < num_channels; ++i)
                prefix_row[i] += prefix_row[i - 1];
        }

        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");
        if (thread_id < NUM_MAX_NVL_PEERS) {
            auto prefix_row = gbl_channel_prefix_matrix + (dst_rdma_rank * NUM_MAX_NVL_PEERS + thread_id) * num_channels;
            #pragma unroll
            for (int i = 1; i < num_channels; ++i)
                prefix_row[i] += prefix_row[i - 1];
        }
    }
}

void notify_dispatch(const int* num_tokens_per_rank,
                     int* moe_recv_counter_mapped,
                     int num_ranks,
                     const int* num_tokens_per_rdma_rank,
                     int* moe_recv_rdma_counter_mapped,
                     const int* num_tokens_per_expert,
                     int* moe_recv_expert_counter_mapped,
                     int num_experts,
                     const bool* is_token_in_rank,
                     int num_tokens,
                     int num_worst_tokens,
                     int num_channels,
                     int hidden_int4,
                     int num_scales,
                     int num_topk,
                     int expert_alignment,
                     int* rdma_channel_prefix_matrix,
                     int* recv_rdma_rank_prefix_sum,
                     int* gbl_channel_prefix_matrix,
                     int* recv_gbl_rank_prefix_sum,
                     void* rdma_buffer_ptr,
                     int num_max_rdma_chunked_recv_tokens,
                     void** buffer_ptrs,
                     int num_max_nvl_chunked_recv_tokens,
                     int** barrier_signal_ptrs,
                     int rank,
                     cudaStream_t stream,
                     int64_t num_rdma_bytes,
                     int64_t num_nvl_bytes,
                     bool low_latency_mode) {
#define NOTIFY_DISPATCH_LAUNCH_CASE(num_rdma_ranks)                                                                                    \
    {                                                                                                                                  \
        auto notify_dispatch_func = low_latency_mode ? notify_dispatch<true, num_rdma_ranks> : notify_dispatch<false, num_rdma_ranks>; \
        LAUNCH_KERNEL(&cfg,                                                                                                            \
                      notify_dispatch_func,                                                                                            \
                      num_tokens_per_rank,                                                                                             \
                      moe_recv_counter_mapped,                                                                                         \
                      num_ranks,                                                                                                       \
                      num_tokens_per_rdma_rank,                                                                                        \
                      moe_recv_rdma_counter_mapped,                                                                                    \
                      num_tokens_per_expert,                                                                                           \
                      moe_recv_expert_counter_mapped,                                                                                  \
                      num_experts,                                                                                                     \
                      is_token_in_rank,                                                                                                \
                      num_tokens,                                                                                                      \
                      num_worst_tokens,                                                                                                \
                      num_channels,                                                                                                    \
                      expert_alignment,                                                                                                \
                      rdma_clean_meta.first,                                                                                           \
                      rdma_clean_meta.second,                                                                                          \
                      nvl_clean_meta.first,                                                                                            \
                      nvl_clean_meta.second,                                                                                           \
                      rdma_channel_prefix_matrix,                                                                                      \
                      recv_rdma_rank_prefix_sum,                                                                                       \
                      gbl_channel_prefix_matrix,                                                                                       \
                      recv_gbl_rank_prefix_sum,                                                                                        \
                      rdma_buffer_ptr,                                                                                                 \
                      buffer_ptrs,                                                                                                     \
                      barrier_signal_ptrs,                                                                                             \
                      rank,                                                                                                            \
                      cpu_rdma_team);                                                                                                  \
    }                                                                                                                                  \
    break

    constexpr int kNumThreads = 512;
    const auto num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;

    // Get clean meta
    auto rdma_clean_meta =
        get_rdma_clean_meta(hidden_int4, num_scales, num_topk, num_topk, num_rdma_ranks, num_max_rdma_chunked_recv_tokens, num_channels);
    auto nvl_clean_meta = get_nvl_clean_meta(hidden_int4,
                                             num_scales,
                                             num_topk,
                                             num_topk,
                                             num_rdma_ranks,
                                             NUM_MAX_NVL_PEERS,
                                             num_max_nvl_chunked_recv_tokens,
                                             num_channels,
                                             true);
    EP_HOST_ASSERT((rdma_clean_meta.first + rdma_clean_meta.second) * sizeof(int) <= num_rdma_bytes);
    EP_HOST_ASSERT((nvl_clean_meta.first + nvl_clean_meta.second) * sizeof(int) <= num_nvl_bytes);
    EP_HOST_ASSERT(num_rdma_bytes < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(num_nvl_bytes < std::numeric_limits<int>::max());

    // Launch kernel
    SETUP_LAUNCH_CONFIG(1 + num_rdma_ranks, kNumThreads, stream);
    SWITCH_RDMA_RANKS(NOTIFY_DISPATCH_LAUNCH_CASE);
#undef NOTIFY_DISPATCH_LAUNCH_CASE
}
