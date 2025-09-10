/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_combine_v2.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_V2_H
#define MOE_DISTRIBUTE_COMBINE_V2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_base.h"
#include "moe_distribute_combine_v2_tiling.h"
#include "check_winsize.h"
namespace MoeDistributeCombineV2Impl {
constexpr uint8_t BUFFER_NUM = 2;                        // 多buf
constexpr uint32_t STATE_OFFSET = 32U;                  // 状态空间偏移地址
constexpr uint32_t STATE_SIZE = 1024UL * 1024UL; // 1M
constexpr uint32_t UB_ALIGN = 32U;                       // UB按32字节对齐
constexpr uint32_t COMBINE_STATE_OFFSET = 64U * 1024U;  // 本卡状态空间偏移地址，前面的地址给dispatch用
constexpr uint8_t EP_DOMAIN = 0;
constexpr uint8_t TP_DOMAIN = 1;
constexpr uint32_t FLOAT_PER_UB_ALIGN = 8U;
constexpr uint64_t WIN_STATE_OFFSET = 500UL * 1024UL;
constexpr uint64_t STATE_WIN_OFFSET = 975UL * 1024UL;   // 预留48*512内存
constexpr uint32_t EXPAND_IDX_INFO = 3U;                // expand_idx是按3元组保存信息，分别为rank_id token_id topk_id
constexpr uint32_t ALIGNED_LEN = 256U;                  // blockReduceMax中，最多支持连续256字节数据参与计算
constexpr float SCALE_PARAM = 127.0;                    // 计算量化参数所需的缩放倍数
constexpr uint32_t BLOCK_NUM = ALIGNED_LEN / UB_ALIGN;  // blockReduceMax中，最多支持连续256字节数据参与计算
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint32_t REDUCE_NUM = 8U;

template<AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

#define TemplateMC2TypeClass typename ExpandXType, typename XType, typename ExpandIdxType, bool IsNeedReduceScatter, bool IsShareExpert, bool IsInt8Quant
#define TemplateMC2TypeFunc ExpandXType, XType, ExpandIdxType, IsNeedReduceScatter, IsShareExpert, IsInt8Quant

using namespace AscendC;
template <TemplateMC2TypeClass>
class MoeDistributeCombineV2 {
public:
    __aicore__ inline MoeDistributeCombineV2() {};
    __aicore__ inline void Init(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR tpSendCount,
                                GM_ADDR expertScales, GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR XOut,
                                GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeCombineV2TilingData *tilingData);
    __aicore__ inline void Process();
private:
    __aicore__ inline void InitDataStatus();
    __aicore__ inline void InitInputAndOutput(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx,
                                              GM_ADDR epSendCount, GM_ADDR expertScales, GM_ADDR xActiveMask,
                                              GM_ADDR sharedExpertX, GM_ADDR XOut);
    __aicore__ inline void InitAttrs(const MoeDistributeCombineV2TilingData *tilingData);
    __aicore__ inline void InitInt8Quant();
    __aicore__ inline void AlltoAllBuffInitAndMaskCal();
    __aicore__ inline void ReduceScatterTrans();
    __aicore__ inline void TokenMaskCalCnt();
    __aicore__ inline void ExpertMaskCalCnt();
    __aicore__ inline void SetWaitTpStatusAndDisPatch();
    __aicore__ inline void CustomAdd(LocalTensor<XType> &dst, LocalTensor<XType> &src0, LocalTensor<XType> &src1);
    __aicore__ inline void ExpertAlltoAllDispatchInnerCopyAdd(uint32_t toRankId, uint32_t tokenId, uint32_t topkId, uint32_t tkIndex);
    __aicore__ inline void ExpertAlltoAllDispatchCopyAdd();
    __aicore__ inline void Int8QuantProcess(); 
    __aicore__ inline void Int8DequantProcess(LocalTensor<XType> &src);
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void BuffInit();
    __aicore__ inline void SplitCoreCal();
    __aicore__ inline void WaitDispatch(uint32_t tokenIndex);
    __aicore__ inline void CalcInvalidPerTokenVec();
    __aicore__ GM_ADDR GetWinAddrByRankId(const int32_t rankId, const uint8_t domain)
    {
        if (domain == EP_DOMAIN) {
            return (GM_ADDR)((epRankId_ == rankId) ? epWinContext_->localWindowsIn :
                        ((HcclRankRelationResV2 *)(epWinContext_->remoteRes[rankId].nextDevicePtr))->windowsIn) +
                        winDataSizeOffset_;
        } else {
            return (GM_ADDR)((tpRankId_ == rankId) ? tpWinContext_->localWindowsIn :
                       ((HcclRankRelationResV2 *)(tpWinContext_->remoteRes[rankId].nextDevicePtr))->windowsIn) + winDataSizeOffset_;
        }
    }

    __aicore__ GM_ADDR GetWinStateAddrByRankId(const int32_t rankId, const uint8_t domain)
    {
        if (domain == EP_DOMAIN) {
            return (GM_ADDR)((epRankId_ == rankId) ? epWinContext_->localWindowsExp :
                ((HcclRankRelationResV2*)(epWinContext_->remoteRes[rankId].nextDevicePtr))->windowsExp) + winStatusOffset_;
        } else {
            return (GM_ADDR)((tpRankId_ == rankId) ? tpWinContext_->localWindowsExp :
                ((HcclRankRelationResV2*)(tpWinContext_->remoteRes[rankId].nextDevicePtr))->windowsExp) + winStatusOffset_;
        }
    }

    __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y)
    {
        return (x < y) ? x : y;
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<ExpandXType> expandXGM_;
    GlobalTensor<bool> xActiveMaskGM_;
    GlobalTensor<int32_t> expertIdsGM_;
    GlobalTensor<ExpandIdxType> expandIdxGM_;
    GlobalTensor<ExpandIdxType> epSendCountGM_;
    GlobalTensor<ExpandIdxType> tpSendCountGM_;
    GlobalTensor<float> expertScalesGM_;
    GlobalTensor<XType> sharedExpertXGM_;
    GlobalTensor<XType> expandOutGlobal_;
    GlobalTensor<XType> rankWindow_;                 // 用于存对端window的变量
    GlobalTensor<XType> tpRankWindow_;
    GlobalTensor<XType> rowTmpGlobal_;
    GM_ADDR epWindowGM_;
    GM_ADDR tpWindowGM_;
    GM_ADDR stateGM_;

    LocalTensor<XType> winTpSendCountTensor_;
    LocalTensor<ExpandXType> gmTpSendCountTensor_;
    LocalTensor<XType> outTensor_;
    LocalTensor<float> winTpSendCountFloatTensor_;
    LocalTensor<float> gmTpSendCountFloatTensor_;

    // tiling侧已确保数据上限， 相乘不会越界，因此统一采用uin32_t进行处理
    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t epWorldSize_{0};
    uint32_t tpWorldSize_{0};
    uint32_t epRankId_{0};
    uint32_t tpRankId_{0};
    uint32_t coreIdx_{0};  // aiv id
    uint32_t sharedExpertNum_{0};
    uint32_t moeExpertPerRankNum_{0};     // 每张卡部署的moe专家数
    uint32_t moeSendNum_{0};              // moeExpertPerRankNum_ * epWorldSize_
    __gm__ HcclOpResParam *epWinContext_{nullptr};
    __gm__ HcclOpResParam *tpWinContext_{nullptr};
    uint32_t tpStateOffsetOnWin_{0};
    uint32_t bsKNum_{0};
    uint32_t startTokenId_{0};
    uint32_t endTokenId_{0};
    uint32_t sendCntNum_{0};
    uint32_t ubSize_{0};
    uint32_t dataState_{0};
    uint32_t stateOffset_{0};
    uint64_t activeMaskBsCnt_{0};
    uint64_t winDataSizeOffset_{0};
    uint64_t winStatusOffset_{0};
    uint64_t totalWinSize_{0};
    uint32_t selfSendCnt_{0};
    uint32_t tpRemoteSendCnt_{0};
    uint32_t activeMaskAlignSize_{0};
    uint32_t hExpandXTypeSize_{0};
    uint32_t hAlign32Size_{0};
    uint32_t hFloatAlign32Size_{0};
    uint32_t hFloatAlign256Size_{0};
    uint32_t hExpandXAlign32Size_{0};
    uint32_t hAlignWinSize_{0};
    uint32_t hAlignWinCnt_{0};
    uint32_t tokenScaleCnt_{0};
    uint32_t scaleNumAlignSize_{0};
    uint32_t flagRcvCount_{0};
    uint32_t axisBsAlignSize_{0};

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> moeQueue_;
    TQue<QuePosition::VECIN, 1> moeSumQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> gmTpSendCountQueue_;
    TQue<QuePosition::VECIN, 1> gmTpSendCountInQueue_;
    TQue<QuePosition::VECIN, 1> winTpSendCountInQueue_;
    TQue<QuePosition::VECOUT, 1> xOutQueue_;
    TBuf<> readStateBuf_;
    TBuf<> expertScalesBuf_;
    TBuf<> rowTmpFloatBuf_;
    TBuf<> sumFloatBuf_;
    TBuf<> mulBuf_;
    TBuf<> indexCountsBuf_;
    TBuf<> winTpSendCountFloatBuf_;
    TBuf<> gmTpSendCountFloatBuf_;
    TBuf<> tokenBuf_;
    TBuf<> xActMaskTBuf_;
    TBuf<> xActMaskCastTBuf_;
    TBuf<> tokenTargetTBuf_;
    TBuf<> vaildBsIndexTBuf_;
    TBuf<> xActMaskSumTBuf_;
    TBuf<> stateBuf_;
    TBuf<> expertIdsBuf_;
    TBuf<> negOneTensorBuf_;
    TBuf<> expertIdsCmpTensorBuf_;
    TBuf<> expertIdsCmpFloatBuf_;
    // TBuf<> expertIdsCmpHalfBuf_;
    TBuf<> workLocalBuf_;
    TBuf<> topkInvalidCntFloatBuf_;
    TBuf<> topkInvalidCntTensorBuf_;
    bool isInputTokenMaskFlag_ = false;
    bool isInputExpertMaskFlag_ = false;
    bool hasSharedExpertX_ = false;

    // int8量化
    TBuf<> xAbsBuf_;
    TBuf<> xMaxBuf_;
    TBuf<> xScaleMulBuf_;

    LocalTensor<int8_t> castLocalTensor_;
    LocalTensor<half> fp16CastTensor_;
    LocalTensor<float> absFloatTensor_;
    LocalTensor<float> reduceMaxFloatTensor_;
    LocalTensor<XType> scaleDivTensor_;
    LocalTensor<float> scaleDivFloatTensor_;
    LocalTensor<float> scaleDupLocalTensor_;
    LocalTensor<XType> sendLocalTensor_;
    LocalTensor<half> tokenTargetTensor_;
    LocalTensor<int32_t> vaildBsIndexTensor_;
    LocalTensor<int32_t> expertIdsTensor_;
    LocalTensor<int32_t> topkInvalidCntTensor_;

    uint32_t mask_{0};
    uint32_t repeatNum_{0};
    uint32_t scaleNum_{0};
    float scaleValFloat_;
};

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<TemplateMC2TypeFunc>::TokenMaskCalCnt()
{
    // 一维mask, 计算得到有效bs数量
    LocalTensor<bool> xActiveMaskTensor = xActMaskTBuf_.Get<bool>();
    LocalTensor<half> tempTensor = xActMaskCastTBuf_.Get<half>();
    LocalTensor<half> sumOutTensor = xActMaskSumTBuf_.Get<half>();
    DataCopyExtParams xActiveMaskParams{1U, static_cast<uint32_t>(axisBS_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPadExtParams<bool> xActiveMaskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(xActiveMaskTensor, xActiveMaskGM_, xActiveMaskParams, xActiveMaskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> xActiveMaskInt8Tensor = xActiveMaskTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, xActiveMaskInt8Tensor, RoundMode::CAST_NONE, axisBS_);
    PipeBarrier<PIPE_V>();
    SumParams params{1, axisBsAlignSize_, axisBS_};
    Sum(sumOutTensor, tempTensor, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    activeMaskBsCnt_ = static_cast<int32_t>(sumOutTensor.GetValue(0));
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<TemplateMC2TypeFunc>::ExpertMaskCalCnt()
{
    // 二维mask, 挑选有效token
    uint64_t rsvdCnt = 0;
    uint32_t mask = axisBS_;
    LocalTensor<bool> maskStrideTensor = tokenBuf_.Get<bool>();
    LocalTensor<half> tempTensor = rowTmpFloatBuf_.Get<half>();
    LocalTensor<half> maskTempTensor = sumFloatBuf_.Get<half>();
    LocalTensor<uint8_t> maskTensor = tokenBuf_.Get<uint8_t>();
    LocalTensor<int32_t> bsIndexTensor = mulBuf_.Get<int32_t>();
    LocalTensor<uint32_t> maskTensorInt32 = tokenBuf_.Get<uint32_t>();
    DataCopyExtParams xActiveMaskParams{
        static_cast<uint16_t>(axisBS_), static_cast<uint32_t>(axisK_ * sizeof(bool)),  0U, 0U, 0U};
    DataCopyPadExtParams<bool> xActiveMaskCopyPadParams{true, 0U, static_cast<uint8_t>(UB_ALIGN - axisK_), 0U};
    SumParams axisBsSumParams{
        1, static_cast<uint32_t>(Ceil(axisBS_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half)), axisBS_};
    uint32_t calCnt = Ceil(axisBS_ * sizeof(half), ALIGNED_LEN) * ALIGNED_LEN / sizeof(half);

    Duplicate<half>(maskTempTensor, (half)0, calCnt);
    DataCopyPad(maskStrideTensor, xActiveMaskGM_, xActiveMaskParams, xActiveMaskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskStrideInt8Tensor = maskStrideTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, maskStrideInt8Tensor, RoundMode::CAST_NONE, activeMaskAlignSize_);
    PipeBarrier<PIPE_V>();
    uint32_t innerAlign = Ceil(axisK_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half) * BUFFER_NUM;
    SumParams axisKSumParams{axisBS_, innerAlign, axisK_};
    Sum(tokenTargetTensor_, tempTensor, axisKSumParams);
    PipeBarrier<PIPE_V>();
    Mins(maskTempTensor, tokenTargetTensor_, static_cast<half>(1), axisBS_);
    PipeBarrier<PIPE_V>();
    CompareScalar(maskTensor, maskTempTensor, static_cast<half>(1), AscendC::CMPMODE::EQ, calCnt);
    CreateVecIndex(bsIndexTensor, 0, axisBS_);
    PipeBarrier<PIPE_V>();
    GatherMask(vaildBsIndexTensor_, bsIndexTensor, maskTensorInt32, true, mask, {1, 1, 0, 0}, activeMaskBsCnt_);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<TemplateMC2TypeFunc>::InitDataStatus()
{
    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    epWinContext_ = (__gm__ HcclOpResParam*)contextGM0;
    GM_ADDR statusDataSpaceGm = (GM_ADDR)epWinContext_->localWindowsExp;

    GlobalTensor<int32_t> selfDataStatusTensor;
    selfDataStatusTensor.SetGlobalBuffer((__gm__ int32_t*)(statusDataSpaceGm + STATE_WIN_OFFSET + coreIdx_ * WIN_ADDR_ALIGN));
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfDataStatusTensor);
    dataState_ = selfDataStatusTensor(0);
    selfDataStatusTensor(0) = ((dataState_ == 0) ? 1 : 0);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfDataStatusTensor);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<TemplateMC2TypeFunc>::InitInputAndOutput(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR expertScales,
    GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR XOut)
{
    expandXGM_.SetGlobalBuffer((__gm__ ExpandXType*)expandX);
    expertIdsGM_.SetGlobalBuffer((__gm__ int32_t*)expertIds);
    expandIdxGM_.SetGlobalBuffer((__gm__ ExpandIdxType*)expandIdx);
    epSendCountGM_.SetGlobalBuffer((__gm__ int32_t*)epSendCount);
    expertScalesGM_.SetGlobalBuffer((__gm__ float*)expertScales);
    xActiveMaskGM_.SetGlobalBuffer((__gm__ bool*)xActiveMask);
    sharedExpertXGM_.SetGlobalBuffer((__gm__ XType*)sharedExpertX);

    expandOutGlobal_.SetGlobalBuffer((__gm__ XType*)XOut);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<TemplateMC2TypeFunc>::InitAttrs(const MoeDistributeCombineV2TilingData *tilingData)
{
    axisBS_ = tilingData->moeDistributeCombineV2Info.bs;
    axisH_ = tilingData->moeDistributeCombineV2Info.h;
    axisK_ = tilingData->moeDistributeCombineV2Info.k;
    aivNum_ = tilingData->moeDistributeCombineV2Info.aivNum;
    ubSize_ = tilingData->moeDistributeCombineV2Info.totalUbSize;
    sharedExpertNum_ = tilingData->moeDistributeCombineV2Info.sharedExpertNum;
    moeExpertPerRankNum_ = tilingData->moeDistributeCombineV2Info.moeExpertPerRankNum;
    epWorldSize_ = tilingData->moeDistributeCombineV2Info.epWorldSize;
    epRankId_ = tilingData->moeDistributeCombineV2Info.epRankId;
    moeSendNum_ = epWorldSize_ * moeExpertPerRankNum_;
    tpWorldSize_ = tilingData->moeDistributeCombineV2Info.tpWorldSize;
    tpRankId_ = tilingData->moeDistributeCombineV2Info.tpRankId;
    totalWinSize_ = tilingData->moeDistributeCombineV2Info.totalWinSize;
    isInputTokenMaskFlag_ = tilingData->moeDistributeCombineV2Info.isTokenMask;
    isInputExpertMaskFlag_ = tilingData->moeDistributeCombineV2Info.isExpertMask;
    hasSharedExpertX_ = tilingData->moeDistributeCombineV2Info.hasSharedExpertX;

    stateOffset_ = STATE_OFFSET;
    uint32_t hFloatSize = axisH_ * static_cast<uint32_t>(sizeof(float));
    hAlign32Size_ = Ceil(axisH_, UB_ALIGN) * UB_ALIGN;
    hFloatAlign32Size_ = Ceil(hFloatSize, UB_ALIGN) * UB_ALIGN;
    hFloatAlign256Size_ = Ceil(hFloatSize, ALIGNED_LEN) * ALIGNED_LEN;
    hExpandXTypeSize_ = axisH_ * sizeof(ExpandXType);
    hExpandXAlign32Size_ = Ceil(hExpandXTypeSize_, UB_ALIGN) * UB_ALIGN;
    hAlignWinSize_ = Ceil(hExpandXTypeSize_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    hAlignWinCnt_ = hAlignWinSize_ / sizeof(ExpandXType);
    bsKNum_ = axisBS_ * axisK_;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<TemplateMC2TypeFunc>::InitInt8Quant()
{
    scaleValFloat_ = static_cast<float>(1.0f / SCALE_PARAM);
    uint32_t scaleGranu = static_cast<uint32_t>(UB_ALIGN / sizeof(float)); // 计算每个block得到的reducemax结果数量
    scaleNum_ = (hExpandXAlign32Size_ / sizeof(ExpandXType)) / scaleGranu; // 得到有效scale的个数
    repeatNum_ = static_cast<uint32_t>(hFloatAlign256Size_ / ALIGNED_LEN); // BlockReduceMax 与 Brcb的重复迭代次数，每次256b参与计算
    mask_ = static_cast<uint32_t>(ALIGNED_LEN / sizeof(float));
    tokenScaleCnt_ = hAlign32Size_ / sizeof(ExpandXType) + scaleNum_; // int8_align + scale有效个数
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<TemplateMC2TypeFunc>::Init(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx,
    GM_ADDR epSendCount, GM_ADDR tpSendCount, GM_ADDR expertScales,
    GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR XOut,
    GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeCombineV2TilingData *tilingData)
{
    tpipe_ = pipe;

    coreIdx_ = GetBlockIdx();

    InitDataStatus();

    InitInputAndOutput(expandX, expertIds, expandIdx, epSendCount, expertScales, xActiveMask, sharedExpertX, XOut);

    InitAttrs(tilingData);

    // 检查hcclwinsize是否越界
    auto realWinSize = epWinContext_->winSize;
    CheckWindowSize(totalWinSize_, realWinSize, tpipe_, XOut);

    if constexpr (IsInt8Quant) {
        InitInt8Quant();
    }

    PipeBarrier<PIPE_ALL>();

    // 当前win区划分为前后两半区，连续两次dispatch，切换半区
    winDataSizeOffset_ = static_cast<uint64_t>(dataState_) * (tilingData->moeDistributeCombineV2Info.totalWinSize / 2UL);
    winStatusOffset_ = COMBINE_STATE_OFFSET + dataState_ * WIN_STATE_OFFSET; // 前面的预留给dispatch使用
    epWindowGM_ = GetWinAddrByRankId(epRankId_, EP_DOMAIN);
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    for (int tempepRankId = 0; tempepRankId < epWorldSize_; tempepRankId++) {
        OOMCheckAddrRange<XType>((__gm__ XType*)(GetWinAddrByRankId(tempepRankId, EP_DOMAIN)), totalWinSize_);
        OOMCheckAddrRange<float>((__gm__ float*)(GetWinStateAddrByRankId(tempepRankId, EP_DOMAIN)), STATE_SIZE);
    }
#endif
    if constexpr (IsShareExpert) {
        DataCacheCleanAndInvalid<ExpandIdxType, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(epSendCountGM_[epWorldSize_ - 1]);
        selfSendCnt_ = epSendCountGM_(epWorldSize_ - 1);
    } else {
        DataCacheCleanAndInvalid<ExpandIdxType, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(epSendCountGM_[moeSendNum_ - 1]);
        selfSendCnt_ = epSendCountGM_(moeSendNum_ - 1);
    }
    SplitCoreCal();
    if constexpr (IsNeedReduceScatter) {
        auto contextGM1 = AscendC::GetHcclContext<1>();
        tpWinContext_ = (__gm__ HcclOpResParam*)contextGM1;
        tpSendCountGM_.SetGlobalBuffer((__gm__ int32_t*)tpSendCount);
        tpWorldSize_ = tilingData->moeDistributeCombineV2Info.tpWorldSize;
        tpRankId_ = tilingData->moeDistributeCombineV2Info.tpRankId;
        tpWindowGM_ = GetWinAddrByRankId(tpRankId_, TP_DOMAIN);
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    for (int temptpRankId = 0; temptpRankId < epWorldSize_; temptpRankId++) {
        OOMCheckAddrRange<XType>((__gm__ XType*)(GetWinAddrByRankId(temptpRankId, TP_DOMAIN)), totalWinSize_);
        OOMCheckAddrRange<int32_t>((__gm__ int32_t*)(GetWinStateAddrByRankId(temptpRankId, TP_DOMAIN)), STATE_SIZE);
    }
#endif
        tpStateOffsetOnWin_ = tpRankId_ * WIN_ADDR_ALIGN;
        tpRankWindow_.SetGlobalBuffer((__gm__ XType*)tpWindowGM_);
        tpRemoteSendCnt_ = tpSendCountGM_(1 - tpRankId_);
    }
    tpipe_->InitBuffer(moeQueue_, BUFFER_NUM, hExpandXAlign32Size_);
    flagRcvCount_ = axisK_ + sharedExpertNum_;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<TemplateMC2TypeFunc>::BuffInit()
{
    tpipe_->Reset();
    tpipe_->InitBuffer(readStateBuf_, UB_ALIGN);  // 32
    if constexpr (IsNeedReduceScatter) {
        tpipe_->InitBuffer(gmTpSendCountInQueue_, BUFFER_NUM, hExpandXAlign32Size_);   // 28K 存储输入拷过来的token
        tpipe_->InitBuffer(xOutQueue_, BUFFER_NUM, hExpandXAlign32Size_);              // 14K 存储输出token
        tpipe_->InitBuffer(winTpSendCountInQueue_, BUFFER_NUM, hExpandXAlign32Size_);  // 14K * 2 存储对端win区token
        if constexpr (AscendC::IsSameType<XType, bfloat16_t>::value) {
            tpipe_->InitBuffer(winTpSendCountFloatBuf_, hFloatAlign32Size_);           // 28K 参与量化及customAdd中token的v核运算
            tpipe_->InitBuffer(gmTpSendCountFloatBuf_, hFloatAlign32Size_);            // 28K 参与量化及customAdd中token的v核运算
            winTpSendCountFloatTensor_ = winTpSendCountFloatBuf_.Get<float>();
            gmTpSendCountFloatTensor_ = gmTpSendCountFloatBuf_.Get<float>();
        }
    } else {
        tpipe_->InitBuffer(gmTpSendCountQueue_, BUFFER_NUM, hExpandXAlign32Size_);   // 28K 存储搬入token
        if constexpr (IsInt8Quant) {
            uint32_t tokenScaleAlign32Size = Ceil(tokenScaleCnt_ * sizeof(ExpandXType), UB_ALIGN) * UB_ALIGN;
            tpipe_->InitBuffer(xOutQueue_, BUFFER_NUM, tokenScaleAlign32Size);              // 28K 输出token搬运
            tpipe_->InitBuffer(xAbsBuf_, hFloatAlign256Size_);                              // 28K blockReduceMax计算及后续Cast计算，256对齐
            uint32_t hFloatAlign256Cnt = hFloatAlign256Size_ / sizeof(float);
            tpipe_->InitBuffer(xMaxBuf_, (hFloatAlign256Cnt / REDUCE_NUM) * sizeof(float));  // 3.5K 存储ReduceMax结果
            tpipe_->InitBuffer(xScaleMulBuf_, hFloatAlign256Size_);                          // 28K 参与Brcb计算，256对齐
            tpipe_->InitBuffer(winTpSendCountFloatBuf_, hFloatAlign32Size_);                 // 28K 参与Div等token v核运算

            winTpSendCountFloatTensor_ = winTpSendCountFloatBuf_.Get<float>();
            absFloatTensor_ = xAbsBuf_.Get<float>();
            reduceMaxFloatTensor_ = xMaxBuf_.Get<float>();
            scaleDupLocalTensor_ = xScaleMulBuf_.Get<float>();
            fp16CastTensor_ = xAbsBuf_.Get<half>();
            Duplicate(absFloatTensor_, float(0), hFloatAlign256Cnt); // 统一写0
        }
    }
    tpipe_->InitBuffer(indexCountsBuf_, sendCntNum_ * EXPAND_IDX_INFO * sizeof(int32_t));
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<TemplateMC2TypeFunc>::AlltoAllBuffInitAndMaskCal()
{
    tpipe_->Reset();
    activeMaskBsCnt_ = axisBS_;
    uint32_t maxSizeTokenBuf = hExpandXAlign32Size_;
    uint32_t maxSizeRowTmpFloatBuf = hFloatAlign32Size_;
    if (isInputExpertMaskFlag_) {
        activeMaskAlignSize_ = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN);
        uint32_t activeMaskAlignHalfSize = activeMaskAlignSize_ * sizeof(half);
        maxSizeTokenBuf = (activeMaskAlignSize_ > hExpandXAlign32Size_ ? activeMaskAlignSize_ : hExpandXAlign32Size_);
        maxSizeRowTmpFloatBuf = (activeMaskAlignHalfSize > hFloatAlign32Size_ ? activeMaskAlignHalfSize : hFloatAlign32Size_);
    }
    tpipe_->InitBuffer(expertScalesBuf_, axisBS_ * axisK_ * sizeof(float));  // BS * K * 4 = 32K
    tpipe_->InitBuffer(tokenBuf_, maxSizeTokenBuf);                          // 16K 用于搬入输入token
    tpipe_->InitBuffer(rowTmpFloatBuf_, maxSizeRowTmpFloatBuf);              // 32K 用于存储cast之后的fp32 token数据
    tpipe_->InitBuffer(mulBuf_, hFloatAlign256Size_);                        // 32K buffer复用， 最大用于存储Brcb之后的token，需要256对齐
    tpipe_->InitBuffer(sumFloatBuf_, hFloatAlign32Size_);                    // 32K add
    tpipe_->InitBuffer(moeSumQueue_, BUFFER_NUM, hExpandXAlign32Size_);      // 32K 搬入
    tpipe_->InitBuffer(stateBuf_, (flagRcvCount_) * STATE_OFFSET);
    if constexpr (IsInt8Quant) {
        scaleNumAlignSize_ = Ceil(scaleNum_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(xAbsBuf_, scaleNumAlignSize_);
        fp16CastTensor_ = mulBuf_.Get<half>();
        absFloatTensor_ = rowTmpFloatBuf_.Get<float>();
        scaleDupLocalTensor_ = mulBuf_.Get<float>();
        scaleDivFloatTensor_ = xAbsBuf_.Get<float>();
    }
    if (isInputTokenMaskFlag_) {
        axisBsAlignSize_ = Ceil(axisBS_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(xActMaskTBuf_, axisBsAlignSize_);
        tpipe_->InitBuffer(xActMaskCastTBuf_, axisBsAlignSize_ * sizeof(half));
        tpipe_->InitBuffer(xActMaskSumTBuf_, axisBsAlignSize_ * sizeof(half));
        TokenMaskCalCnt(); // 计算一维mask
    }
    if (isInputExpertMaskFlag_) {
        tpipe_->InitBuffer(tokenTargetTBuf_, Ceil(axisBS_ * sizeof(half), UB_ALIGN) * UB_ALIGN);
        tpipe_->InitBuffer(vaildBsIndexTBuf_, Ceil(axisBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
        tokenTargetTensor_ = tokenTargetTBuf_.Get<half>();
        vaildBsIndexTensor_ = vaildBsIndexTBuf_.Get<int32_t>();
        ExpertMaskCalCnt(); // 计算二维mask
    }
    uint32_t expertIdsSize = bsKNum_ * sizeof(int32_t);
    uint32_t expertIdsFloatSize = bsKNum_ * sizeof(float);
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsSize);             // BS * K * 4 = 32K
    tpipe_->InitBuffer(negOneTensorBuf_, expertIdsSize);
    tpipe_->InitBuffer(expertIdsCmpTensorBuf_, bsKNum_ * sizeof(uint8_t));
    tpipe_->InitBuffer(topkInvalidCntTensorBuf_, axisBS_ * sizeof(int32_t));
    tpipe_->InitBuffer(topkInvalidCntFloatBuf_, axisBS_ * sizeof(float));
    tpipe_->InitBuffer(expertIdsCmpFloatBuf_, expertIdsFloatSize);
    tpipe_->InitBuffer(workLocalBuf_, expertIdsSize);

    expertIdsTensor_ = expertIdsBuf_.Get<int32_t>();
    topkInvalidCntTensor_ = topkInvalidCntTensorBuf_.Get<int32_t>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<TemplateMC2TypeFunc>::SplitCoreCal()
{
    // 对需要发送的token数平均分核，得到每个核上处理的卡的数量
    sendCntNum_ = selfSendCnt_ / aivNum_;
    uint32_t remainderRankNum = selfSendCnt_ % aivNum_;
 
    startTokenId_ = sendCntNum_ * coreIdx_;
 
    if (coreIdx_ < remainderRankNum) {
        sendCntNum_++;
        startTokenId_ += coreIdx_;
    } else {
        startTokenId_ += remainderRankNum;
    }
    endTokenId_ = startTokenId_ + sendCntNum_;
}

// 当前逻辑为tp=2场景，泛化待重新适配，本卡token在最前面
// 当tp为2时，直接把对端tp的数据分核处理发送
template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<TemplateMC2TypeFunc>::ReduceScatterTrans()
{
    uint32_t tokenTpOffset = selfSendCnt_;
    uint32_t offset = selfSendCnt_ * axisH_;
    GlobalTensor<ExpandXType> dataCopyInGM = expandXGM_[offset];
    GM_ADDR rankGM = GetWinAddrByRankId(1 - static_cast<int32_t>(tpRankId_), TP_DOMAIN);
    rankWindow_.SetGlobalBuffer((__gm__ XType*)rankGM);
    uint32_t tpSendCntNum = tpRemoteSendCnt_ / aivNum_;
    uint32_t remainderRankNum = tpRemoteSendCnt_ % aivNum_;
    uint32_t copyStartIdx = tpSendCntNum * coreIdx_;
    if (coreIdx_ < remainderRankNum) {
        tpSendCntNum++;
        copyStartIdx += coreIdx_;
    } else {
        copyStartIdx += remainderRankNum;
    }
    if (tpSendCntNum == 0U) {
        return;
    }
    uint32_t copyEndIdx = copyStartIdx + tpSendCntNum;

    LocalTensor<ExpandXType> tmpUb;

    // 确定rankid
    DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    for (uint32_t tokenNumIdx = copyStartIdx; tokenNumIdx < copyEndIdx; tokenNumIdx++) {
        tmpUb = moeQueue_.AllocTensor<ExpandXType>();
        DataCopyPad(tmpUb, dataCopyInGM[tokenNumIdx * axisH_], expandXCopyParams, copyPadExtParams);
        moeQueue_.EnQue(tmpUb);
        tmpUb = moeQueue_.DeQue<ExpandXType>();
        DataCopyPad(rankWindow_[tokenNumIdx * hAlignWinCnt_], tmpUb, expandXCopyParams);
        moeQueue_.FreeTensor<ExpandXType>(tmpUb);
    }
}

// 流水流程
// 46 -> gm -> ub syncall win->gm add -> alltoall
// 2 -> win wait syncall gm -> ub win ->gm add -> alltoall
template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<TemplateMC2TypeFunc>::SetWaitTpStatusAndDisPatch()
{
    PipeBarrier<PIPE_ALL>();
    if ((coreIdx_ >= tpRemoteSendCnt_) && (coreIdx_ >= selfSendCnt_)) {
        return;
    }
    if constexpr (IsNeedReduceScatter) {
        uint32_t tpToRankId = 1U - tpRankId_; // 当前适配按tpWorldSize_==2来写
        PipeBarrier<PIPE_ALL>();
        LocalTensor<int32_t> statusFlagUb = readStateBuf_.Get<int32_t>();
        statusFlagUb(0) = 1;
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        GlobalTensor<int32_t> tpStatusWinTensor;
        stateGM_ = GetWinStateAddrByRankId(tpToRankId, TP_DOMAIN) + coreIdx_ * WIN_ADDR_ALIGN;
        tpStatusWinTensor.SetGlobalBuffer((__gm__ int32_t*)stateGM_);
        DataCopy<int32_t>(tpStatusWinTensor, statusFlagUb, 8UL); // 8是数据大小，按32对齐拷贝
        SyncFunc<AscendC::HardEvent::MTE3_S>();

        GM_ADDR tpStatusWin = GetWinStateAddrByRankId(tpRankId_, TP_DOMAIN) + coreIdx_ * WIN_ADDR_ALIGN;
        GlobalTensor<int32_t> selfStatusWinTensor;
        selfStatusWinTensor.SetGlobalBuffer((__gm__ int32_t*)tpStatusWin);
        int32_t sumOfFlag = 0;
        while (sumOfFlag != 1) {
            DataCopy<int32_t>(statusFlagUb, selfStatusWinTensor, 8);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            sumOfFlag = statusFlagUb.GetValue(0);
            SyncFunc<AscendC::HardEvent::S_MTE2>();
        }
        selfStatusWinTensor(0) = 0;
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfStatusWinTensor);
    }

    // Copy win gm->ub add ->alltoall send
    ExpertAlltoAllDispatchCopyAdd();
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<TemplateMC2TypeFunc>::ExpertAlltoAllDispatchCopyAdd()
{
    if (sendCntNum_ == 0U) { // 空闲核，直接返回
        return;
    }

    LocalTensor<ExpandIdxType> expandIdxLocal = indexCountsBuf_.Get<ExpandIdxType>();
    const DataCopyExtParams bskParams{1U, static_cast<uint32_t>(sendCntNum_ * EXPAND_IDX_INFO * sizeof(uint32_t)), 0U,
                                      0U, 0U};
    const DataCopyPadExtParams<ExpandIdxType> copyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expandIdxLocal, expandIdxGM_[startTokenId_ * EXPAND_IDX_INFO], bskParams, copyPadParams);
    LocalTensor<float> statusTensor = readStateBuf_.AllocTensor<float>();
    Duplicate<float>(statusTensor, (float)1, FLOAT_PER_UB_ALIGN);

    SyncFunc<AscendC::HardEvent::MTE2_S>();
    for (uint32_t loop = 0; loop < sendCntNum_; loop++) {
        uint32_t tkIndex = startTokenId_ + ((loop + epRankId_) % sendCntNum_); // 错位发送
        uint32_t baseOffset = (tkIndex - startTokenId_) * EXPAND_IDX_INFO;
        uint32_t toRankId = static_cast<uint32_t>(expandIdxLocal(baseOffset));     // 位置0是rank_id
        uint32_t tokenId = static_cast<uint32_t>(expandIdxLocal(baseOffset + 1));  // 位置1是token_id
        uint32_t topkId = static_cast<uint32_t>(expandIdxLocal(baseOffset + 2));   // 位置2是topk_id

        ExpertAlltoAllDispatchInnerCopyAdd(toRankId, tokenId, topkId, tkIndex);
        PipeBarrier<PIPE_ALL>();
        GM_ADDR stateGM = GetWinStateAddrByRankId(toRankId, EP_DOMAIN) + tokenId * flagRcvCount_ * stateOffset_ +
            topkId * stateOffset_;  // 计算地址偏移
        GlobalTensor<float> stateGMTensor;
        stateGMTensor.SetGlobalBuffer((__gm__ float*)stateGM);
        DataCopy<float>(stateGMTensor, statusTensor, FLOAT_PER_UB_ALIGN);

        // ===== Debug Log =====
        if (coreIdx_ == 0) {
            PRINTF("[DispatchSetStatus] epRankId=%u coreIdx=%u -> toRankId=%u tokenId=%u topkId=%u tkIndex=%u stateGM=%p val=%.1f\n",
                   epRankId_, coreIdx_, toRankId, tokenId, topkId, tkIndex,
                   stateGM, statusTensor(0));
        }
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<TemplateMC2TypeFunc>::Int8QuantProcess()
{
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    castLocalTensor_ = sendLocalTensor_.template ReinterpretCast<int8_t>(); // 长度为int8H_Align + scaleNum
    scaleDivTensor_ = castLocalTensor_[hAlign32Size_].template ReinterpretCast<XType>(); // 偏移前面的int8

    Cast(winTpSendCountFloatTensor_, gmTpSendCountTensor_, RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();
    Abs(absFloatTensor_, winTpSendCountFloatTensor_, axisH_); // absFloatTensor_ align到256并写0，支持ReduceMax与Brcb
    PipeBarrier<PIPE_V>();
    BlockReduceMax(reduceMaxFloatTensor_, absFloatTensor_, repeatNum_, mask_, 1, 1, BLOCK_NUM); // 32->1 256->8
    PipeBarrier<PIPE_V>();
    Muls(reduceMaxFloatTensor_, reduceMaxFloatTensor_, scaleValFloat_, scaleNum_); // 有效个数
    PipeBarrier<PIPE_V>();
    Cast(scaleDivTensor_, reduceMaxFloatTensor_, RoundMode::CAST_RINT, scaleNum_); // 有效个数
    PipeBarrier<PIPE_V>();
    Brcb(scaleDupLocalTensor_, reduceMaxFloatTensor_, repeatNum_, {1, BLOCK_NUM}); // 一次256
    PipeBarrier<PIPE_V>();
    Div(winTpSendCountFloatTensor_, winTpSendCountFloatTensor_, scaleDupLocalTensor_, axisH_); // 有效个数
    PipeBarrier<PIPE_V>();
    Cast(fp16CastTensor_, winTpSendCountFloatTensor_, RoundMode::CAST_RINT, axisH_);
    PipeBarrier<PIPE_V>();
    Cast(castLocalTensor_, fp16CastTensor_, RoundMode::CAST_RINT, axisH_);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<TemplateMC2TypeFunc>::ExpertAlltoAllDispatchInnerCopyAdd(
    uint32_t toRankId, uint32_t tokenId, uint32_t topkId, uint32_t tkIndex)
{
    uint32_t dataCnt = axisH_;
    uint32_t epOffset = tokenId * (axisK_ + sharedExpertNum_) + topkId;
    uint32_t tokenGMOffset = tkIndex * axisH_;
    uint32_t tokenWinOffset = tkIndex * hAlignWinCnt_;
    GM_ADDR rankGM = GetWinAddrByRankId(toRankId, EP_DOMAIN) + epOffset * hAlignWinSize_;
    rankWindow_.SetGlobalBuffer((__gm__ XType*)rankGM);
    DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    DataCopyExtParams xScaleCopyParams{1U, static_cast<uint32_t>(tokenScaleCnt_ * sizeof(ExpandXType)), 0U, 0U, 0U};
    if constexpr (IsNeedReduceScatter) {
        gmTpSendCountTensor_ = gmTpSendCountInQueue_.AllocTensor<ExpandXType>();
        DataCopyPad(gmTpSendCountTensor_, expandXGM_[tokenGMOffset], expandXCopyParams, copyPadExtParams);
        gmTpSendCountInQueue_.EnQue(gmTpSendCountTensor_);
        winTpSendCountTensor_ = winTpSendCountInQueue_.AllocTensor<ExpandXType>();
        DataCopyPad(winTpSendCountTensor_, tpRankWindow_[tokenWinOffset], expandXCopyParams, copyPadExtParams);
        winTpSendCountInQueue_.EnQue(winTpSendCountTensor_);
        gmTpSendCountTensor_ = gmTpSendCountInQueue_.DeQue<ExpandXType>();
        winTpSendCountTensor_ = winTpSendCountInQueue_.DeQue<ExpandXType>();
        outTensor_ = xOutQueue_.AllocTensor<ExpandXType>();
        CustomAdd(outTensor_, winTpSendCountTensor_, gmTpSendCountTensor_);
        gmTpSendCountInQueue_.FreeTensor<ExpandXType>(gmTpSendCountTensor_);
        winTpSendCountInQueue_.FreeTensor<ExpandXType>(winTpSendCountTensor_);
        xOutQueue_.EnQue(outTensor_);
        outTensor_ = xOutQueue_.DeQue<ExpandXType>();
        DataCopyPad(rankWindow_, outTensor_, expandXCopyParams);
        xOutQueue_.FreeTensor<ExpandXType>(outTensor_);
    } else {
        if constexpr (IsInt8Quant) {
            gmTpSendCountTensor_ = gmTpSendCountQueue_.AllocTensor<ExpandXType>();
            DataCopyPad(gmTpSendCountTensor_, expandXGM_[tokenGMOffset], expandXCopyParams, copyPadExtParams);
            gmTpSendCountQueue_.EnQue(gmTpSendCountTensor_);
            gmTpSendCountTensor_ = gmTpSendCountQueue_.DeQue<ExpandXType>();
            sendLocalTensor_ = xOutQueue_.AllocTensor<ExpandXType>();
            Int8QuantProcess();
            xOutQueue_.EnQue(sendLocalTensor_);
            sendLocalTensor_ = xOutQueue_.DeQue<ExpandXType>();
            DataCopyPad(rankWindow_, sendLocalTensor_, xScaleCopyParams);
            gmTpSendCountQueue_.FreeTensor<ExpandXType>(gmTpSendCountTensor_);
            xOutQueue_.FreeTensor<ExpandXType>(sendLocalTensor_);
        } else {
            gmTpSendCountTensor_ = gmTpSendCountQueue_.AllocTensor<ExpandXType>();
            DataCopyPad(gmTpSendCountTensor_, expandXGM_[tokenGMOffset], expandXCopyParams, copyPadExtParams);
            gmTpSendCountQueue_.EnQue(gmTpSendCountTensor_);
            gmTpSendCountTensor_ = gmTpSendCountQueue_.DeQue<ExpandXType>();
            DataCopyPad(rankWindow_, gmTpSendCountTensor_, expandXCopyParams);
            gmTpSendCountQueue_.FreeTensor<ExpandXType>(gmTpSendCountTensor_);
        }
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<TemplateMC2TypeFunc>::CustomAdd(LocalTensor<XType> &dst,
  LocalTensor<XType> &src0, LocalTensor<XType> &src1)
{
    if constexpr (AscendC::IsSameType<XType, bfloat16_t>::value) {
        Cast(winTpSendCountFloatTensor_, src0, RoundMode::CAST_NONE, axisH_);
        Cast(gmTpSendCountFloatTensor_, src1, RoundMode::CAST_NONE, axisH_);
        PipeBarrier<PIPE_V>();
        Add(winTpSendCountFloatTensor_, winTpSendCountFloatTensor_, gmTpSendCountFloatTensor_, axisH_);
        PipeBarrier<PIPE_V>();
        Cast(dst, winTpSendCountFloatTensor_, RoundMode::CAST_RINT, axisH_);
    } else {
        Add(dst, src0, src1, axisH_);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<TemplateMC2TypeFunc>::CalcInvalidPerTokenVec()
{
    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(bsKNum_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expertIdsTensor_, expertIdsGM_, expertIdsCntParams, expertIdsCntCopyPadParams);
    PipeBarrier<PIPE_V>();
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    // if (coreIdx_ == 0) {
    //     PRINTF("\n===[combine] rank:%d, coreIdx_:%d, expertIdsTensor_:%d %d %d %d %d %d %d %d \n", epRankId_, coreIdx_,
    //         expertIdsTensor_.GetValue(0),
    //         expertIdsTensor_.GetValue(1),
    //         expertIdsTensor_.GetValue(2),
    //         expertIdsTensor_.GetValue(3),
    //         expertIdsTensor_.GetValue(4),
    //         expertIdsTensor_.GetValue(5),
    //         expertIdsTensor_.GetValue(6),
    //         expertIdsTensor_.GetValue(7));
    // }

    // Step1: 构造常量 -1 tensor，与 expertIdsTensor_ 同 shape
    LocalTensor<int32_t> negOneTensor = negOneTensorBuf_.Get<int32_t>();  // shape = [axisBS_ * axisK_]
    Duplicate<int32_t>(negOneTensor, -1, bsKNum_);
    PipeBarrier<PIPE_V>();
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    // if (coreIdx_ == 0) {
    //     PRINTF("\n===[combine] rank:%d, coreIdx_:%d, negOneTensor:%d %d %d %d %d %d %d %d \n", epRankId_, coreIdx_,
    //         negOneTensor.GetValue(0),
    //         negOneTensor.GetValue(1),
    //         negOneTensor.GetValue(2),
    //         negOneTensor.GetValue(3),
    //         negOneTensor.GetValue(4),
    //         negOneTensor.GetValue(5),
    //         negOneTensor.GetValue(6),
    //         negOneTensor.GetValue(7));
    // }

    // Step2: 比较 expertIdsTensor_ == -1，得到0/1 tensor
    LocalTensor<uint8_t> cmpTensor = expertIdsCmpTensorBuf_.Get<uint8_t>();    // shape = [axisBS_ * axisK_]
    Compare(cmpTensor, expertIdsTensor_, negOneTensor, CMPMODE::EQ, bsKNum_);
    PipeBarrier<PIPE_V>();
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    SyncFunc<AscendC::HardEvent::V_S>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    PipeBarrier<PIPE_V>();
    // if (coreIdx_ == 0) {
    //     PRINTF("\n===[combine] rank:%d, coreIdx_:%d, cmpTensor:%u(0x%x) %u(0x%x) %u(0x%x) %u(0x%x) %u(0x%x) %u(0x%x) %u(0x%x) %u(0x%x) \n", epRankId_, coreIdx_,
    //         static_cast<uint8_t>(cmpTensor.GetValue(0)), static_cast<uint8_t>(cmpTensor.GetValue(0)),
    //         static_cast<uint8_t>(cmpTensor.GetValue(1)), static_cast<uint8_t>(cmpTensor.GetValue(1)),
    //         static_cast<uint8_t>(cmpTensor.GetValue(2)), static_cast<uint8_t>(cmpTensor.GetValue(2)),
    //         static_cast<uint8_t>(cmpTensor.GetValue(3)), static_cast<uint8_t>(cmpTensor.GetValue(3)),
    //         static_cast<uint8_t>(cmpTensor.GetValue(4)), static_cast<uint8_t>(cmpTensor.GetValue(4)),
    //         static_cast<uint8_t>(cmpTensor.GetValue(5)), static_cast<uint8_t>(cmpTensor.GetValue(5)),
    //         static_cast<uint8_t>(cmpTensor.GetValue(6)), static_cast<uint8_t>(cmpTensor.GetValue(6)),
    //         static_cast<uint8_t>(cmpTensor.GetValue(7)), static_cast<uint8_t>(cmpTensor.GetValue(7)));
    // }

    // LocalTensor<half> cmpHalf = expertIdsCmpHalfBuf_.Get<half>(); // shape = [axisBS_ * axisK_]
    LocalTensor<float> cmpFloat = expertIdsCmpFloatBuf_.Get<float>(); // shape = [axisBS_ * axisK_]
    // Cast: int32 -> float。RoundMode::CAST_NONE 表示直接转换数值
    // Cast(cmpHalf, cmpTensor, RoundMode::CAST_NONE, bsKNum_);
    // Cast(cmpFloat, cmpHalf, RoundMode::CAST_NONE, bsKNum_);
    for (int i = 0; i < bsKNum_; ++i) {
        // 找到对应的byte和bit位置
        int byteIdx = i / 8;
        int bitIdx = i % 8;
        uint8_t val = cmpTensor.GetValue(byteIdx);
        uint8_t bit = (val >> bitIdx) & 0x1;
        int32_t tmp = static_cast<int32_t>(bit);
        cmpFloat.SetValue(i, static_cast<float>(tmp));
    }
    PipeBarrier<PIPE_V>();
    SyncFunc<AscendC::HardEvent::V_S>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    // if (coreIdx_ == 0) {
    //     PRINTF("\n===[combine] rank:%d, coreIdx_:%d, cmpFloat:%f %f %f %f %f %f %f %f \n", epRankId_, coreIdx_,
    //         cmpFloat.GetValue(0),
    //         cmpFloat.GetValue(1),
    //         cmpFloat.GetValue(2),
    //         cmpFloat.GetValue(3),
    //         cmpFloat.GetValue(4),
    //         cmpFloat.GetValue(5),
    //         cmpFloat.GetValue(6),
    //         cmpFloat.GetValue(7));
    // }
    // Step3: 按 axisK_ 维度做 reduce sum，得到每行 -1 个数
    uint32_t shape[] = { axisBS_, axisK_ };
    LocalTensor<float> topkInvalidCntFloat = topkInvalidCntFloatBuf_.Get<float>();
    ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(topkInvalidCntFloat, cmpFloat, workLocalBuf_.Get<uint8_t>(), shape, true);
    SyncFunc<AscendC::HardEvent::V_S>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    // if (coreIdx_ == 0) {
    //     PRINTF("\n===[combine] rank:%d, coreIdx_:%d, topkInvalidCntFloat:%f %f %f %f %f %f %f %f \n", epRankId_, coreIdx_,
    //         topkInvalidCntFloat.GetValue(0),
    //         topkInvalidCntFloat.GetValue(1),
    //         topkInvalidCntFloat.GetValue(2),
    //         topkInvalidCntFloat.GetValue(3),
    //         topkInvalidCntFloat.GetValue(4),
    //         topkInvalidCntFloat.GetValue(5),
    //         topkInvalidCntFloat.GetValue(6),
    //         topkInvalidCntFloat.GetValue(7));
    // }
    Cast(topkInvalidCntTensor_, topkInvalidCntFloat, RoundMode::CAST_FLOOR, axisBS_);
    PipeBarrier<PIPE_V>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    // if (coreIdx_ == 0) {
    //     PRINTF("\n===[combine] rank:%d, coreIdx_:%d, topkInvalidCntTensor_:%d %d %d %d %d %d %d %d \n", epRankId_, coreIdx_,
    //         topkInvalidCntTensor_.GetValue(0),
    //         topkInvalidCntTensor_.GetValue(1),
    //         topkInvalidCntTensor_.GetValue(2),
    //         topkInvalidCntTensor_.GetValue(3),
    //         topkInvalidCntTensor_.GetValue(4),
    //         topkInvalidCntTensor_.GetValue(5),
    //         topkInvalidCntTensor_.GetValue(6),
    //         topkInvalidCntTensor_.GetValue(7));
    // }
}
template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<TemplateMC2TypeFunc>::WaitDispatch(uint32_t tokenIndex)
{
    uint32_t copyCount = flagRcvCount_ * FLOAT_PER_UB_ALIGN;
    uint32_t targetCount = copyCount;
    targetCount = (flagRcvCount_ - topkInvalidCntTensor_.GetValue(tokenIndex)) * FLOAT_PER_UB_ALIGN;
    if (isInputExpertMaskFlag_) {
        int32_t tokenTarget = static_cast<int32_t>(tokenTargetTensor_.GetValue(tokenIndex)) + sharedExpertNum_;
        targetCount = tokenTarget * FLOAT_PER_UB_ALIGN;
    }

    // 计算地址偏移
    GM_ADDR stateGM = GetWinStateAddrByRankId(epRankId_, EP_DOMAIN) + tokenIndex * flagRcvCount_ * stateOffset_;
    GlobalTensor<float> stateGMTensor;
    stateGMTensor.SetGlobalBuffer((__gm__ float*)stateGM);

    float localState = 0;
    float target = (float)1.0 * targetCount;
    float minTarget = target - (float)0.5;
    float maxTarget = target + (float)0.5;
    SumParams sumParams{1, copyCount, copyCount};
    LocalTensor<float> stateTensor = stateBuf_.Get<float>();

    // Debug control：采样输出间隔与超时阈值（可调）
    constexpr int LOG_INTERVAL = 102400;        // 每多少次打印一次（压测下改大）
    constexpr int MAX_ITER = 1 << 20;         // 到达后打印详细快照并 break（避免无限挂起）
    int iter = 0;

    while ((localState < minTarget) || (localState > maxTarget)) {
        // 小心：MTE2 sync / DataCopy
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        DataCopy<float>(stateTensor, stateGMTensor, copyCount);
        SyncFunc<AscendC::HardEvent::MTE2_V>();

        // reduce sum -> stateTensor(0)
        Sum(stateTensor, stateTensor, sumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        localState = stateTensor(0);

        // 限量日志：每 LOG_INTERVAL 次打印一次基本信息（只由 coreIdx_==0 打印）
        if ((iter % LOG_INTERVAL) == 0 && coreIdx_ == 0) {
            PRINTF("[WaitDispatch] rank:%d core:%d token:%u iter:%d localState:%f target:%f copyCount:%u flagRcvCount:%d topkInvalid:%d \n",
                   epRankId_, coreIdx_, tokenIndex, iter, localState, target, copyCount, flagRcvCount_, topkInvalidCntTensor_.GetValue(tokenIndex));
        }

        if (iter >= MAX_ITER) {
            if (coreIdx_ == 0) {
                PRINTF("[WaitDispatch:TIMEOUT] rank:%d core:%d token:%u iter:%d localState:%f target:%f copyCount:%u flagRcvCount:%d topkInvalid:%d -> dumping head of stateTensor \n",
                       epRankId_, coreIdx_, tokenIndex, iter, localState, target, copyCount, flagRcvCount_, topkInvalidCntTensor_.GetValue(tokenIndex));

                // 打印 stateTensor 的前若干个元素（避免过多日志）
                int dumpN = (copyCount < 32) ? copyCount : 32;
                for (int i = 0; i < dumpN; ++i) {
                    PRINTF(" state[%d]=%f", i, stateTensor.GetValue(i));
                }
                PRINTF("\\n");
            }
        }

        ++iter;
    }

    // 若正常退出循环或超时退出，清零并写回 GM
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<float>(stateTensor, (float)0.0, copyCount);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy<float>(stateGMTensor, stateTensor, copyCount);
    // SyncFunc<AscendC::HardEvent::MTE2_S>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<TemplateMC2TypeFunc>::Int8DequantProcess(LocalTensor<XType>& src)
{
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    castLocalTensor_ = src.template ReinterpretCast<int8_t>();
    scaleDivTensor_ = src[hAlign32Size_ / 2];

    SyncFunc<AscendC::HardEvent::S_V>();
    Cast(scaleDivFloatTensor_, scaleDivTensor_, RoundMode::CAST_NONE, scaleNum_);
    Cast(fp16CastTensor_, castLocalTensor_, RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();
    Cast(absFloatTensor_, fp16CastTensor_, RoundMode::CAST_NONE, axisH_);
    Brcb(scaleDupLocalTensor_, scaleDivFloatTensor_, repeatNum_, {1, BLOCK_NUM});
    PipeBarrier<PIPE_V>();
    Mul(absFloatTensor_, absFloatTensor_, scaleDupLocalTensor_, axisH_);
    PipeBarrier<PIPE_V>();
    Cast(src, absFloatTensor_, RoundMode::CAST_RINT, axisH_);
    PipeBarrier<PIPE_V>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<TemplateMC2TypeFunc>::LocalWindowCopy()
{
    if (activeMaskBsCnt_ == 0U) {
        return;
    }
    uint32_t beginIndex = 0U;
    uint32_t endIndex = 0U;
    uint32_t processLen = 0U;
    uint32_t tokenOffset = 0U;
    uint32_t tokenPerAivNum = activeMaskBsCnt_ / aivNum_;
    uint32_t remainderToken = activeMaskBsCnt_ % aivNum_;

    beginIndex = tokenPerAivNum * coreIdx_;
    if (coreIdx_ < remainderToken) {
        tokenPerAivNum++;
        beginIndex += coreIdx_;
    } else {
        beginIndex += remainderToken;
    }
    endIndex = beginIndex + tokenPerAivNum;
    if (tokenPerAivNum == 0U) {
        return;
    }
    processLen = axisH_;
    LocalTensor<float> expertScalesLocal = expertScalesBuf_.Get<float>();
    LocalTensor<float> rowTmpFloatLocal = rowTmpFloatBuf_.Get<float>();
    LocalTensor<float> mulBufLocal = mulBuf_.Get<float>();
    LocalTensor<float> sumFloatBufLocal = sumFloatBuf_.Get<float>();

    const DataCopyExtParams bskParams{1U, static_cast<uint32_t>(bsKNum_ * sizeof(uint32_t)), 0U, 0U, 0U};
    const DataCopyPadExtParams<float> copyPadFloatParams{false, 0U, 0U, 0U};
    const DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    const DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    DataCopyPad(expertScalesLocal, expertScalesGM_, bskParams, copyPadFloatParams);
    const DataCopyExtParams xScaleCopyParams{
        1U, static_cast<uint32_t>(tokenScaleCnt_ * sizeof(ExpandXType)), 0U, 0U, 0U};
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    for (uint32_t curIdx = beginIndex; curIdx < endIndex; curIdx++) {
        uint32_t tokenIndex = curIdx;
        if (isInputExpertMaskFlag_) {
            tokenIndex = vaildBsIndexTensor_.GetValue(curIdx);
        }
        if (coreIdx_ == 0) {
            PRINTF("[Before WaitDispatch] rank:%d core:%d curIdx:%u tokenIndex:%u \n", epRankId_, coreIdx_, curIdx, tokenIndex );
        }
        WaitDispatch(tokenIndex);
        uint32_t index = tokenIndex * axisK_;
        float scaleVal = 0.0;
        GM_ADDR wAddr;
        SyncFunc<AscendC::HardEvent::MTE3_V>(); // 与结果搬出datacopy同tensor
        Duplicate(sumFloatBufLocal, static_cast<float>(0), axisH_);
        LocalTensor<XType> tmpUb;
        uint32_t tokenIndexOffset = tokenIndex * (axisK_ + sharedExpertNum_);
        for (uint32_t topkId = 0U; topkId < axisK_; topkId++) {
            scaleVal = expertScalesLocal.GetValue(index);
            wAddr = (__gm__ uint8_t*)(epWindowGM_) + (tokenIndexOffset + topkId) * hAlignWinSize_;
            rowTmpGlobal_.SetGlobalBuffer((__gm__ XType*)wAddr);
            tmpUb = moeSumQueue_.AllocTensor<XType>();
            if constexpr (IsInt8Quant) {
                DataCopyPad(tmpUb, rowTmpGlobal_, xScaleCopyParams, copyPadExtParams);
            } else {
                DataCopyPad(tmpUb, rowTmpGlobal_, expandXCopyParams, copyPadExtParams);
            }
            moeSumQueue_.EnQue(tmpUb);
            tmpUb = moeSumQueue_.DeQue<XType>();
            if constexpr (IsInt8Quant) {
                Int8DequantProcess(tmpUb);
            }
            Cast(rowTmpFloatLocal, tmpUb, AscendC::RoundMode::CAST_NONE, processLen);
            PipeBarrier<PIPE_V>();
            AscendC::Muls(mulBufLocal, rowTmpFloatLocal, scaleVal, processLen);
            PipeBarrier<PIPE_V>();
            AscendC::Add(sumFloatBufLocal, sumFloatBufLocal, mulBufLocal, processLen);
            index++;
            moeSumQueue_.FreeTensor<XType>(tmpUb);
        }
        for (uint32_t topkId = axisK_; topkId < (axisK_ + sharedExpertNum_); topkId++) {
            wAddr = (__gm__ uint8_t*)(epWindowGM_) + (tokenIndexOffset + topkId) * hAlignWinSize_;
            rowTmpGlobal_.SetGlobalBuffer((__gm__ XType*)wAddr);
            tmpUb = moeSumQueue_.AllocTensor<XType>();
            if constexpr (IsInt8Quant) {
                DataCopyPad(tmpUb, rowTmpGlobal_, xScaleCopyParams, copyPadExtParams);
            } else {
                DataCopyPad(tmpUb, rowTmpGlobal_, expandXCopyParams, copyPadExtParams);
            }
            moeSumQueue_.EnQue(tmpUb);
            tmpUb = moeSumQueue_.DeQue<XType>();
            if constexpr (IsInt8Quant) {
                Int8DequantProcess(tmpUb);
            }
            Cast(rowTmpFloatLocal, tmpUb, AscendC::RoundMode::CAST_NONE, processLen);
            PipeBarrier<PIPE_V>();
            AscendC::Add(sumFloatBufLocal, sumFloatBufLocal, rowTmpFloatLocal, processLen);
            PipeBarrier<PIPE_V>();
            moeSumQueue_.FreeTensor<XType>(tmpUb);
        }
        if (hasSharedExpertX_) {
            LocalTensor<XType> rowTmpLocal = tokenBuf_.Get<XType>();
            SyncFunc<AscendC::HardEvent::V_MTE2>();  // 与结果搬出Cast同地址
            DataCopyPad(rowTmpLocal, sharedExpertXGM_[tokenIndex * axisH_], expandXCopyParams, copyPadExtParams);
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            Cast(rowTmpFloatLocal, rowTmpLocal, AscendC::RoundMode::CAST_NONE, processLen);
            PipeBarrier<PIPE_V>();
            AscendC::Add(sumFloatBufLocal, sumFloatBufLocal, rowTmpFloatLocal, processLen);
        }

        // 结果搬出
        PipeBarrier<PIPE_V>();
        LocalTensor<XType> sumBufLocal = tokenBuf_.Get<XType>();
        Cast(sumBufLocal, sumFloatBufLocal, AscendC::RoundMode::CAST_RINT, processLen);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(expandOutGlobal_[tokenIndex * axisH_ + tokenOffset], sumBufLocal, expandXCopyParams);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<TemplateMC2TypeFunc>::Process()
{
    if ASCEND_IS_AIV { // 全aiv处理
        if constexpr (IsNeedReduceScatter) {
            ReduceScatterTrans();
        }
        if (coreIdx_ == 0) {
            PRINTF("[Combine enter] epRankId=%u coreIdx=%u ==================================\n",
                   epRankId_, coreIdx_);
        }
        BuffInit();
        SetWaitTpStatusAndDisPatch();
        AlltoAllBuffInitAndMaskCal();
        CalcInvalidPerTokenVec();
        LocalWindowCopy();
    }
}

} // MoeDistributeCombineV2Impl
#endif // MOE_DISTRIBUTE_COMBINE_IMPL_H