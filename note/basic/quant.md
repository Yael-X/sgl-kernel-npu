# 深度学习数值精度体系：BF16 / FP16 / FP8 / FP4

> **核心标签**：#Quantization #HPC #LLM #Hardware #System
> **一句话总结**：大模型的发展史，就是一部不断压榨数值精度、用精度换带宽（Bandwidth）和算力（FLOPS）的历史。

---

## 1. Why：为什么我们需要低精度？

### 1.1 Memory Wall（存储墙）—— 真正的瓶颈

在现代 AI 芯片（GPU / NPU）上，算力的增长速度远超显存带宽的增长速度：

*   **算力（FLOPS）**：每代增长约 **2x - 4x**
*   **带宽（HBM Bandwidth）**：每代仅增长 **1.2x - 1.5x**

**后果**：

> **模型越来越大，计算单元（Tensor Core）经常处于“空转”状态，等待数据从 HBM 搬运过来。**

**低精度的收益**：

1.  **减少搬运量**：FP8 (1B) 对比 FP16 (2B)，带宽利用率直接翻倍。
2.  **KV Cache 优化**：推理时长序列的瓶颈在于 KV Cache 显存占用，低精度量化是长文本（Long Context）的必经之路。

### 1.2 Compute Bound vs. Memory Bound

*   **Compute Bound（计算受限）**：矩阵乘法（GEMM）极大，算力跑满。
    *   *对策*：利用 Tensor Core 的低精度指令（如 FP8 Tensor Core 吞吐量是 BF16 的 2 倍）。
*   **Memory Bound（访存受限）**：Element-wise 操作（如 Activation、Normalization）或 MoE 的 Expert Dispatch。
    *   *对策*：越小的数据格式越好，因为瓶颈在 I/O。

### 1.3 Power Wall（能耗墙）

不仅是为了快，更是为了省电。数据移动的能耗远高于计算能耗。

| 操作 (Operation)    | 相对能耗 (Relative Energy) | 备注                         |
| :------------------ | :------------------------- | :--------------------------- |
| **FP32 ADD**        | 1x                         | 基准                         |
| **FP32 MUL**        | 4x                         |                              |
| **FP16 MUL**        | ~1x                        | 显著降低                     |
| **INT8 ADD**        | 0.03x                      | 整数加法极省电               |
| **SRAM 读取**       | **50x**                    | **读数据的能耗远大于计算！** |
| **DRAM (HBM) 读取** | **2000x+**                 | 必须减少 HBM 访问            |

---

## 2. What：浮点数的解剖学

IEEE 754 标准公式：
$$
V = (-1)^S \times 2^{E - \text{bias}} \times (1 + M)
$$


*   **Exponent (指数位)** --> 决定 **动态范围 (Range)** (能表示多大的数，防止溢出/下溢)。
*   **Mantissa (尾数位)** -->  决定 **精度 (Precision)** (数值的分辨率)。
*   **Sign(符号位)**

### 2.1 格式详解与可视对比

```text
[FP32]  Sign(1) | Exponent(8)  | Mantissa(23)            | 经典基准
[FP16]  Sign(1) | Exponent(5)  | Mantissa(10)            | 范围太窄，易溢出(NaN)
[BF16]  Sign(1) | Exponent(8)  | Mantissa(7)             | 截断的FP32，范围完美
[FP8-E4M3] (1)  | Exponent(4)  | Mantissa(3)             | 只有范围，精度极低
[FP8-E5M2] (1)  | Exponent(5)  | Mantissa(2)             | 类似FP16的微缩版
```

### 2.2 关键差异点 (Key Takeaways)

1.  **FP16 的致命伤**：指数位只有 5 位，最大值仅 65504。训练大模型时，梯度经常下溢（变为0）或 Loss 上溢（NaN）。**必须使用 Loss Scaling**。
2.  **BF16 的胜利**：指数位与 FP32 一致（8位）。**它可以表示 FP32 能表示的所有数量级**。训练极度稳定，不需要 Loss Scaling。
    *   *本质*：神经网络对“数值有多大”很敏感，但对“数值的小数点后第几位”具有鲁棒性（耐噪）。
3.  **FP8 的分裂**：
    *   **E4M3**：精度稍好，用于 **Weights（权重）** 和 **Activations（激活）**。
    *   **E5M2**：范围大（梯度跨度大），用于 **Gradients（梯度）**。

---

## 3. Comparison：全景对比表

| Type     | Bits |   Exponent (Range)   | Mantissa (Precision) | 典型硬件支持       | 核心应用场景                            |
| :------- | :--: | :------------------: | :------------------: | :----------------- | :-------------------------------------- |
| **FP32** |  32  |          8           |          23          | All GPU            | 科学计算、参考基准、累加器(Accumulator) |
| **FP16** |  16  |          5           |          10          | V100, T4, RTX20/30 | 推理、早期的混合精度训练 (需 Scaling)   |
| **BF16** |  16  |    **8** (同FP32)    |          7           | A100, H100, TPU    | **大模型训练默认格式** (稳定第一)       |
| **FP8**  |  8   | 4 (E4M3)<br>5 (E5M2) |        3<br>2        | H100, RTX4090      | 高性能推理、训练的前向传播              |
| **FP4**  |  4   |       2 (E2M1)       |          1           | Blackwell (B200)   | 极致推理速度 (需配合 Block Scaling)     |
| **INT8** |  8   |          -           |          -           | T4, A100...        | 传统 CNN 量化，LLM 中用于 W8A8          |

---

## 4. Trend & Cutting Edge：前沿演进

### 4.1 演进路线图

> 趋势：从通用精度向特定领域精度（Domain Specific Format）演进。

1.  **2016-2018 (NVIDIA Pascal/Volta)**: **FP16** 引入。
    *   *痛点*：训练不稳定，需要复杂的混合精度策略。
2.  **2019-2020 (Google TPU / NVIDIA Ampere)**: **BF16** 爆发。
    *   *突破*：解决了 Range 问题，大模型训练开始普及。
3.  **2022-2023 (NVIDIA Hopper)**: **FP8** (Transformer Engine)。
    *   *技术*：硬件原生支持 FP8 计算，FP16/32 累加。
4.  **2024+ (NVIDIA Blackwell)**: **FP4** 登场。
    *   *革新*：为了应对万亿参数模型，必须打破 8-bit 界限。

### 4.2 为什么 FP4 还能用？(Block Scaling)

FP4 的表示能力极差（只有 16 个离散值）。如果直接把权重转 FP4，模型会直接崩塌。
解决方案是 **Micro-scaling (Block Quantization)**：

*   不把整个 tensor 统一量化。
*   将 tensor 切成小块（如每 16 个元素一组）。
*   **每组单独维护一个高精度（FP8/BF16）的 Scale Factor**。
*   $Value \approx \text{Scale}_{\text{block}} \times \text{FP4}_{\text{data}}$
*   **Blackwell 硬件层面支持这种解包计算**，从而在保持精度的同时享受 4-bit 的传输带宽。

---

## 5. Practical Notes：避坑指南

1.  **PyTorch 加载模型时**：
    *   如果你有 A100/3090/4090：`torch_dtype=torch.bfloat16` 是首选。
    *   如果你用 V100/2080Ti/T4：只能用 `torch_dtype=torch.float16`（硬件不支持 BF16 计算）。
2.  **关于 OOM (显存溢出)**：
    *   量化（8bit/4bit）是解决 OOM 的最快手段，但会轻微牺牲精度。
    *   使用 `bitsandbytes` 库加载 4-bit 模型：`load_in_4bit=True`。
3.  **训练 vs 推理**：
    *   **训练**：推荐 BF16（主权重 FP32，计算 BF16）。FP8 训练目前仍主要由 NVIDIA Transformer Engine 等底层库处理，手写较难收敛。
    *   **推理**：大胆使用 FP16 或 FP8/INT8 量化。推理对精度损失容忍度很高。