#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/types.h>
#include <tuple>
#include <vector>

#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#include "config.hpp"
#include "event.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME deep_ep_cpp
#endif

namespace deep_ep {

struct Buffer {
    int64_t rank, rdma_rank;
    int64_t num_ranks;

    int64_t num_nvl_bytes;
    int64_t num_rdma_bytes;

    bool low_latency_mode = false;
    bool is_padding = false;
    at::Tensor ori_x;
    at::Tensor new_topk_idx;
    at::Tensor new_scales;
    at::Tensor padding_x = 

private:
    std::string moe_all_to_all_group_name;

    int device_id;

    HcclComm ep_comm;

    bool available = false;

public:
    Buffer(int64_t rank, int64_t num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes, bool low_latency_mode,
           std::string moe_all_to_all_group_name);

    ~Buffer() noexcept(false);

    bool is_available() const;

    std::tuple<at::Tensor, std::optional<at::Tensor>, at::Tensor, at::Tensor, at::Tensor, std::optional<EventHandle>,
               std::optional<std::function<void()>>>
    low_latency_dispatch(const at::Tensor &x, const at::Tensor &topk_idx,
                         const std::optional<at::Tensor> &cumulative_local_expert_recv_stats,
                         int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts, bool use_fp8, bool round_scale,
                         bool use_ue8m0, bool async, bool return_recv_hook, int64_t sharedExpertRankNum = 0);

    int get_rdma_rank() const;

    std::tuple<at::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>> low_latency_combine(
        const at::Tensor &x, const at::Tensor &topk_idx, const at::Tensor &topk_weights, const at::Tensor &src_info,
        const at::Tensor &layout_range, int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts,
        const at::Tensor &ep_send_count, bool zero_copy, bool async, bool return_recv_hook,
        const std::optional<at::Tensor> &out, int64_t shared_expert_rank_num);
};

}  // namespace deep_ep
