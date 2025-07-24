#include <memory>
#include <pybind11/functional.h>
#include <torch/python.h>

#include "hccl/hccl.h"
#include "exception.hpp"
#include "deep_ep.hpp"
#include "pytorch_npu_helper.hpp"


namespace deep_ep {

Buffer::Buffer(int64_t rank, int64_t num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes, bool low_latency_mode,
               std::string moe_all_to_all_group_name)
    : rank(rank),
      num_ranks(num_ranks),
      num_nvl_bytes(num_nvl_bytes),
      num_rdma_bytes(num_rdma_bytes),
      low_latency_mode(low_latency_mode),
      moe_all_to_all_group_name(moe_all_to_all_group_name)
{
    rdma_rank = rank;
    EP_HOST_ASSERT(0 <= rank and rank < num_ranks);

    if (moe_all_to_all_group_name.empty()) {
        char *ranktable_file = std::getenv("RANK_TABLE_FILE");
        EP_HOST_ASSERT(ranktable_file != nullptr)
        ACL_CHECK(aclrtGetDevice(&device_id));

        // ep domain
        HCCL_CHECK(HcclCommInitClusterInfo(ranktable_file, device_id, &ep_comm));
    } else {
        EP_HOST_ASSERT(moe_all_to_all_group_name.size() < 128);
    }
}

Buffer::~Buffer() noexcept(false) {
}

bool Buffer::is_available() const {
    return available;
}

std::tuple<at::Tensor, std::optional<at::Tensor>, at::Tensor, at::Tensor, at::Tensor, std::optional<EventHandle>,
    std::optional<std::function<void()>>>
    Buffer::low_latency_dispatch(const at::Tensor &x, const at::Tensor &topk_idx,
        const std::optional<at::Tensor> &cumulative_local_expert_recv_stats, int64_t num_max_dispatch_tokens_per_rank,
        int64_t num_experts, bool use_fp8, bool round_scale, bool use_ue8m0, bool async, bool return_recv_hook)
{
    this->is_padding = false;
    EP_HOST_ASSERT(low_latency_mode);
    at::Tensor new_x = x;
    this->new_topk_idx = topk_idx;
    if (topk_idx.size(0) == 0) {
        this->is_padding = true;
        this->ori_x = x.clone();
        new_x = torch::ones({1, 7168}, x.options());
        this->new_topk_idx = torch::arange(0, 8, topk_idx.options()).reshape({1, 8});
    }

    auto num_tokens = static_cast<int>(new_x.size(0)), hidden = static_cast<int>(new_x.size(1));
    auto num_scales = hidden / 128, num_topk = static_cast<int>(new_topk_idx.size(1));
    auto num_local_experts = num_experts / num_ranks;

    // Allocate packed tensors
    auto device = new_x.device();
    auto packed_recv_x = at::empty({num_local_experts * num_ranks * num_max_dispatch_tokens_per_rank, hidden},
        new_x.options().dtype(use_fp8 ? at::kChar : at::kBFloat16));
    auto packed_recv_x_scales = at::empty(
        {num_local_experts * num_ranks * num_max_dispatch_tokens_per_rank}, at::dtype(at::kFloat).device(device));
    auto expandIdx = at::empty({num_tokens * num_topk}, at::dtype(at::kInt).device(device));
    auto packed_recv_count = at::empty({num_local_experts * num_ranks}, at::dtype(at::kInt).device(device));
    auto tp_recv_count = at::empty({1}, at::dtype(at::kInt).device(device));
    auto expertTokenNumsOut = at::empty({num_local_experts}, at::dtype(at::kLong).device(device));
    auto expandScales = at::empty({1}, at::dtype(at::kFloat).device(device));
    at::Tensor scales;
    at::Tensor activateMask;
    auto expert_scales = at::empty({1}, at::dtype(at::kFloat).device(device));
    int64_t quantMode = use_fp8 ? 2 : 0;
    int64_t tpSize = 1;
    int64_t tpRank = 0;
    int64_t shardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t expertTokenNumsType = 1;
    int64_t globalBS = num_max_dispatch_tokens_per_rank * num_ranks;
    int64_t sharedExpertRankNum = 0;

    // get ep & tp name
    char hcomEpName[128];
    if (!moe_all_to_all_group_name.empty()) {
        std:memcpy(hcomEpName, moe_all_to_all_group_name.data(), moe_all_to_all_group_name.size() + 1);
    } else {
        HCCL_CHECK(HcclGetCommName(ep_comm, hcomEpName));
    }

    EXEC_NPU_CMD(aclnnMoeDistributeDispatch,
        new_x,
        new_topk_idx,
        scales,         // smooth scales,
        activateMask,   // activateMask
        expert_scales,  // expert_scales
        hcomEpName,     // ep
        num_ranks,      // rankSize
        rank,           // rankId
        num_experts,
        hcomEpName,           // tp
        tpSize,               // tpSize
        tpRank,               // tpRank
        shardType,            // shardType
        sharedExpertNum,      // sharedExpertNum
        sharedExpertRankNum,  // sharedExpertRankNum
        quantMode,
        globalBS,             // globalBS
        expertTokenNumsType,  // expertTokenNumsType
        packed_recv_x,
        packed_recv_x_scales,  // dynamicScalesOut
        expandIdx,
        expertTokenNumsOut,
        packed_recv_count,
        tp_recv_count,
        expandScales);

    // Wait streams
    std::optional<EventHandle> event;

    // Return values
    return {packed_recv_x, packed_recv_x_scales, packed_recv_count, expandIdx, expertTokenNumsOut, event, std::function<void()>([]{})};
}

int Buffer::get_rdma_rank() const {
    return rdma_rank;
}

std::tuple<at::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>> Buffer::low_latency_combine(
    const at::Tensor &x, const at::Tensor &topk_idx, const at::Tensor &topk_weights, const at::Tensor &src_info,
    const at::Tensor &layout_range, int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts,
    const at::Tensor &ep_send_count, bool zero_copy, bool async, bool return_recv_hook,
    const std::optional<at::Tensor> &out)
{
    at::Tensor new_idx = topk_idx;
    at::Tensor new_scales = topk_weights;
    if (this->is_padding) {
        new_idx = this->new_topk_idx;
        this->new_scales = torch::zeros({1, 8}, topk_weights.options());
        new_scales = this->new_scales;
    }
    // Tensor checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous() and x.scalar_type() == at::kBFloat16);
    // EP_HOST_ASSERT(x.size(0) == num_experts / num_ranks);

    // get ep & tp name
    char hcomEpName[128];
    if (!moe_all_to_all_group_name.empty()) {
        std:memcpy(hcomEpName, moe_all_to_all_group_name.data(), moe_all_to_all_group_name.size() + 1);
    } else {
        HCCL_CHECK(HcclGetCommName(ep_comm, hcomEpName));
    }

    auto device = x.device();
    at::Tensor expand_x = x;
    at::Tensor expert_ids = new_idx;
    at::Tensor expand_idx = src_info; // handle[0] = src_info
    at::Tensor ep_send_counts = ep_send_count;
    at::Tensor expert_scales = new_scales;
    at::Tensor tp_send_counts = at::empty({1}, at::dtype(at::kInt).device(device));
    at::Tensor x_active_mask, activation_scale, weight_scale, group_list, expand_scales;

    int64_t tpWorldSize = 1;
    int64_t tpRankId = 0;
    int64_t expertSharedType = 0;
    int64_t shared_expert_num = 1;
    int64_t globalBS = num_max_dispatch_tokens_per_rank * num_ranks;
    int64_t sharedExpertRankNum = 0;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;

    auto num_combined_tokens = static_cast<int>(new_scales.size(0));
    auto hidden = static_cast<int>(x.size(1));
    at::Tensor combined_x = at::empty({num_combined_tokens, hidden}, x.options());
    std::optional<EventHandle> event;

    EXEC_NPU_CMD(aclnnMoeDistributeCombine,
        expand_x,
        expert_ids,
        expand_idx,
        ep_send_counts,
        expert_scales,
        tp_send_counts,
        x_active_mask,
        activation_scale,
        weight_scale,
        group_list,
        expand_scales,
        hcomEpName,
        num_ranks,
        rank,
        num_experts,
        hcomEpName,
        tpWorldSize,
        tpRankId,
        expertSharedType,
        shared_expert_num,
        sharedExpertRankNum,
        globalBS,
        outDtype,
        commQuantMode,
        groupListType,
        combined_x);
    if (this->is_padding) {
        combined_x = this->ori_x;
    }
    return {combined_x, event, std::function<void()>([]{})};
}

} // namespace deep_ep
