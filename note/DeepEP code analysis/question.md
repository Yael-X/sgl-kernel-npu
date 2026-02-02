## 1.请你解释一下DEEPEP为什么在dispatch阶段用了fp8量化，但是combine阶段没有用fp8量化

这是一个非常敏锐且触及 **MoE（混合专家模型）架构设计痛点** 的问题。

在 DeepSeek-V3/R1 的架构设计以及 DeepEP（Deep Expert Parallelism）通信库的实现中，**Dispatch 阶段使用 FP8 而 Combine 阶段坚持使用 BF16**，并不是因为技术上做不到（通信库当然可以传输 FP8），而是出于 **数值稳定性（Numerical Stability）** 和 **模型收敛精度** 的考量。

以下是深层原因的技术分析：

### 1. 算术操作本质不同：Copy vs. Reduction

这是最根本的原因。

*   **Dispatch 阶段 (Scatter/Copy)**：
    *   **操作**：这本质上是一个**搬运**过程。将 Token $X$ 搬运到 Expert $E$。
    *   **影响**：如果在搬运过程中将 BF16 量化为 FP8，引入的是**量化噪声 (Quantization Noise)**。神经网络（特别是 Transformer 的前向传播）对输入特征的轻微噪声具有很强的鲁棒性。这就像给输入图片加了一层极淡的滤镜，模型依然能识别。
    *   **收益**：带宽减半，通信速度翻倍。这对于 All-to-All 这种带宽密集型操作至关重要。

*   **Combine 阶段 (Gather/Reduction)**：
    *   **操作**：这本质上是一个**加权求和**过程。
        公式：$Y = \sum_{i=1}^K w_i \cdot \text{Expert}_i(X)$
    *   **问题**：
        1.  **累加精度丢失**：MoE 的 Combine 是多个专家输出的叠加。如果每个专家的输出都被量化为 FP8（只有 152 种数值表示，动态范围有限），再进行相加，会发生严重的**下溢（Underflow）**或**精度截断**。特别是当 $w_i$（门控权重）很小的时候，FP8 可能直接把某些专家的微弱贡献变成 0。
        2.  **Swamping 现象**：如果一个专家的输出值很大，另一个很小，在低精度下相加，小值会被大值完全“吞没”。
    *   **代码证据**：
        在你提供的代码 `combine_token` 函数中，我们可以清晰地看到：
        ```cpp
        // Reduce bias
        float values[kDtypePerInt4] = {0}; // <--- 强制使用 float32 进行累加
        // ...
        // Reduce all-to-all results
        values[k] += static_cast<float>(recv_value_dtypes[k]);
        ```
        即便传输的是 BF16，**累加操作都必须在 FP32 下进行**以保证精度，更不用说如果传输是 FP8 了。如果在传输前就压成 FP8，那么累加前的源数据就已经损失了无法挽回的精度。

### 2. 反向传播的梯度敏感性

Combine 阶段处于前向传播的末端，但却是**反向传播（Back Propagation）的开端**。

*   在反向传播时，梯度 $\frac{\partial L}{\partial Y}$ 会首先通过 Combine 算子分发给各个专家。
*   如果 Combine 阶段在前向传播时使用了 FP8，那么在计算梯度的链式法则中，梯度的精度也会受到影响。
*   DeepSeek-V3 的技术报告中提到，为了保证训练的稳定性（Loss 不发散），关键路径（如 Residual connection 和 Combine）必须保持高精度。专家的输出往往包含对 Token 表示的“修正量”（Residual Update），这些修正量通常数值较小，FP8 极易导致这些精细的修正量丢失，从而导致模型无法收敛。

### 3. 数据分布特征 (Outliers)

*   **Dispatch 数据（Activations）**：
    经过 LayerNorm 或 RMSNorm 后，输入给专家的 Token 激活值分布通常比较规整（接近高斯分布），适合 FP8 量化。而且现在的 FP8 量化技术（如 Per-Token Scaling）能很好地处理激活值。

*   **Combine 数据（Output of MLPs）**：
    专家 MLP 的输出（即 Up-proj * Down-proj 之后的结果）往往存在**异常值（Outliers）**。某些维度的值可能非常大（Feature spikes）。
    *   FP8 的动态范围（E4M3 或 E5M2）非常有限。
    *   BF16 拥有和 FP32 相同的指数位（8-bit exponent），能很好地覆盖极宽的动态范围。
    *   如果在 Combine 传输前强制截断到 FP8，这些 Outliers 会被截断（Clamping），严重改变模型的输出语义。

### 4. 硬件与重叠（Overlap）的权衡

虽然 FP8 能节省带宽，但在 Combine 阶段，计算和通信的重叠方式与 Dispatch 不同。

*   **Dispatch**：可以在专家计算前很久就开始发送数据，通信掩盖比较容易。
*   **Combine**：必须等专家计算完才能发数据。此时，GPU 往往处于“等待数据回来进行下一步 Attention/Block 计算”的临界路径上。虽然理论上带宽很重要，但此时**准确地把结果加回来**比**快速地传一个错误结果回来**更重要。
*   DeepEP 的优化重点在于通过 **NVLink (Intra-node) 聚合** 来减少节点间通信量，这已经在算法层面极大地降低了带宽需求（减少了 $1/R_{intra}$ 的流量），因此对 FP8 进一步压缩带宽的迫切性没有 Dispatch 阶段那么高。

### 总结

DeepEP 在 Dispatch 阶段使用 FP8 是为了**用精度换速度**（输入数据的容错性高），而在 Combine 阶段坚持使用 BF16 是为了**保住底线**（累加操作对精度极度敏感，且不仅影响前向输出，更致命地影响反向梯度的正确性）。

这是一次经典的**系统效率与算法数学特性**之间的 Trade-off。

## 2.什么是Per-Token Scaling
**Per-Token Scaling（逐Token量化缩放）** 是一种高精度的量化技术，主要用于处理大语言模型（LLM）中的**激活值（Activations）**。

简单来说，它的核心思想是：**“不搞一刀切，为每一个 Token 单独计算一个量化比例尺。”**

为了让你透彻理解，我们从**为什么需要它**、**它是怎么做的**以及**它在 DeepSeek/DeepEP 中的作用**三个层面来解析。

---

### 1. 为什么需要它？（背景：异常值问题）

在 LLM（如 LLaMA、DeepSeek）中，激活值（输入矩阵 $X$）的数值分布有一个非常显著的特点：**异常值（Outliers）极其严重**。

*   **场景**：假设你的输入是一句话：“我爱吃苹果”。
    *   Token A ("我") 的特征向量数值可能都很小，在 [-0.5, 0.5] 之间。
    *   Token B ("爱") 的特征向量中，某几个维度的值可能突然飙升到 100.0（这被称为 Feature Spikes）。
*   **传统做法（Per-Tensor Scaling）**：
    *   如果你对整句话的所有 Token 使用**同一个**缩放因子（Scale），为了包容 Token B 的那个 100.0，你的刻度必须拉得很大。
    *   **结果**：Token A 的那些 [-0.5, 0.5] 的细腻数值，在巨大的刻度下，被强制量化成了 0，或者全部挤在同一个量化格子里。**Token A 的信息丢了。**

**Per-Token Scaling 的解决方案**：
*   给 Token A 一个细腻的比例尺（专门处理小数值）。
*   给 Token B 一个粗犷的比例尺（专门包容大数值）。
*   这样大家都保住了精度。

### 2. 它是怎么做的？（技术原理）

假设输入激活矩阵 $X$ 的形状是 `[Batch_Size, Seq_Len, Hidden_Dim]`，我们将 `Batch_Size * Seq_Len` 统称为 $T$ (Token总数)。矩阵在逻辑上通过 Reshape 变为 `[T, Hidden_Dim]`。

#### 步骤：
1.  **分组**：把矩阵看作 $T$ 行。
2.  **找最大值**：对于每一行（即每一个 Token），单独找出这一行中绝对值的最大值。
    $$ \text{max}_i = \max(|x_{i, :}|) $$
3.  **计算 Scale**：根据目标格式（比如 FP8 E4M3，最大表示值为 448）计算这一行的缩放因子。
    $$ \text{scale}_i = \frac{\text{max}_i}{448} $$
4.  **量化**：用这一行专属的 scale 去除这一行的所有元素，然后取整。
    $$ x_{quant, i, j} = \text{round}(\frac{x_{i, j}}{\text{scale}_i}) $$

#### 结果：
*   **量化后的数据**：`[T, Hidden_Dim]` 的 FP8/Int8 矩阵。
*   **缩放因子表**：`[T, 1]` 的高精度（FP32/BF16）向量。

### 3. 直观对比：Per-Tensor vs Per-Token

想象你要给一群人拍照（量化），这群人里有一个巨人（异常值）和一群普通人。

| 特性 | Per-Tensor Scaling (整张量) | Per-Token Scaling (逐Token) |
| :--- | :--- | :--- |
| **做法** | 用一张巨大的广角镜头拍所有人。 | 给每个人单独拍特写，根据身高调整焦距。 |
| **对巨人的处理** | 拍得清楚。 | 拍得清楚。 |
| **对普通人的处理** | **普通人变成了蚂蚁，看不清脸（精度丢失）。** | 普通人也能拍得清清楚楚。 |
| **存储开销** | 极小（整个矩阵这就 1 个 float）。 | 较小（每行多存 1 个 float）。 |
| **计算复杂度** | 低。 | **较高**（需要在矩阵乘法时动态解量化）。 |

### 4. 在 DeepEP 和 DeepSeek-V3 中的应用

回到你之前的问题，为什么 DeepEP 在 Dispatch 阶段敢用 FP8？正是因为有了 **Per-Token Scaling**。

1.  **Dispatch 阶段的数据**：
    *   此时的数据是经过 LayerNorm 之后的激活值。虽然 LayerNorm 试图归一化，但 LLM 中仍然存在严重的 Token 间差异。
    *   如果使用 Per-Token Scaling，DeepEP 可以把每个 Token 压缩成 FP8，同时附带一个 FP32 的 Scale。
    *   **传输内容** = `FP8_Data` (大头) + `FP32_Scales` (极小，仅占 1/Hidden_Dim)。
    *   **接收端**：专家收到数据后，用 Scale 还原（反量化）数据，或者直接支持 FP8 计算的 Tensor Core 利用这些 Scale 进行矩阵乘法。

2.  **为什么不说 "Per-Channel"？**
    *   **Per-Channel**（逐通道/逐列）通常用于**权重（Weights）**量化。因为权重是静态的，每一列代表一个特征通道，列内分布相对一致。
    *   **Per-Token**（逐行）专门用于**激活（Activations）**量化。因为激活是动态的，每一行代表一个独立的 Token 语义。

### 5. 硬件支持 (H100/H800)

NVIDIA Hopper 架构（H100/H800）的 Tensor Core 原生支持这种计算模式。

在进行 FP8 矩阵乘法 $C = A \times B$ 时：
*   $A$ (激活) 使用 **Per-Token** scales (向量)。
*   $B$ (权重) 使用 **Per-Channel** scales (向量)。
*   Tensor Core 能够高效地处理 $A_{fp8} \times B_{fp8}$，然后利用这两个 Scale 向量在累加器中恢复结果的数值范围。

**总结：Per-Token Scaling 是让 FP8 在 LLM 训练/推理中保持高精度的“魔法”，它解决了 Token 之间数值差异过大的问题，使得 Dispatch 阶段的通信带宽压缩成为可能且安全的操作。**



## 3.为什么deepep里的输入x要通过 int4 加载？
在 CUDA 高性能编程（特别是针对带宽敏感的算子）中，使用 `int4` 加载输入 `x` 是一个标准的**向量化内存访问（Vectorized Memory Access）** 优化手段。

这里的 `x` 数据类型实际上是 `bf16` (2 bytes)，但代码强制将其转换为 `int4` (16 bytes = 128 bits) 进行处理。

以下是具体的四大技术原因：

### 1. 极致的内存带宽利用率 (Memory Bandwidth Saturation)

*   **原理**：GPU 的显存控制器（Memory Controller）和 L2 Cache 并不是以 2 字节（BF16）为单位工作的，而是以 32 字节（Sector）或 128 字节（Cache Line）为单位传输数据的。
*   **对比**：
    *   **如果用 `bf16` 加载**：一个线程发起一次读取指令，只拿回 2 字节。为了填满一个 32 字节的 Sector，需要 Warp 里多个线程完美配合（Coalescing）。
    *   **如果用 `int4` 加载**：**一条指令直接加载 16 字节（128 bits）**。这意味着一个线程一次就搬运了 **8 个 BF16 元素**。
*   **效果**：这就像是用“集装箱”搬货，而不是用“小勺子”舀水。它能以最少的指令数打满 GPU 的内存带宽。

### 2. 显著减少指令发射数 (Reduced Instruction Count)

GPU 的指令发射单元（Warp Scheduler / Dispatch Unit）也是有瓶颈的。

*   **数量级差异**：
    *   处理 8 个 `bf16` 数据，如果用标量加载 (`ld.global.b16`)，需要发射 **8 条指令**。
    *   使用 `int4` 加载 (`ld.global.v4.b32` 或 `ld.128`)，只需要发射 **1 条指令**。
*   **收益**：
    *   减少了流水线停顿（Pipeline Stalls）。
    *   减少了指令缓存（I-Cache）的压力。
    *   让 SM (Streaming Multiprocessor) 把更多的周期留给数学计算，而不是在忙着发射“读取内存”的命令。

### 3. TMA (Tensor Memory Accelerator) 的硬性要求

代码中大量使用了 **TMA**（Hopper 架构的异步拷贝引擎）。

*   **对齐要求**：TMA 是为了大块数据搬运设计的。虽然它支持不同的 Tensor Map，但在底层物理层面上，它极度依赖 **128-bit (16 Byte) 对齐**来实现最高效率。
*   **数据类型无关**：TMA 不关心你是 float 还是 bf16，它只关心比特流。`int4` 是 CUDA C++ 中表示“128-bit 连续内存块”的标准数据类型。
*   **代码证据**：
    ```cpp
    // 强制把 smem_ptr 转换成 int4* 指针，告诉编译器按 16 字节对齐处理
    auto tma_load_buffer = [=](const int& i, const int& j) -> int4* { ... };
    ```
    如果没有这种强制转换，TMA 指令可能无法生成，或者编译器会生成效率低下的普通 Load/Store 指令。

### 4. 简化数据处理逻辑 (Type Agnostic Data Movement)

在 `combine` 算子的很多阶段（特别是搬运和存入 Buffer 阶段），SM 并不需要知道这 128 bit 里面代表的是数值。

*   **视为二进制块**：将数据视为 `int4` 可以方便地进行“大块拷贝”。
*   **只在计算时拆包**：只有在真正需要做加法（Reduction）的时候，代码才利用 `reinterpret_cast` 把 `int4` 拆解回 8 个 `dtype_t` (BF16)：
    ```cpp
    // 只有在计算时才转回 dtype_t
    auto recv_value_dtypes = reinterpret_cast<const dtype_t*>(tma_load_buffer(stage_idx, j) + lane_id);
    #pragma unroll
    for (int k = 0; k < kDtypePerInt4; ++k) // kDtypePerInt4 = 16 / 2 = 8
        values[k] += static_cast<float>(recv_value_dtypes[k]);
    ```

### 总结

这里的 `int4` 与整数运算无关，它是一个**容器**。
**输入 `x` 通过 `int4` 加载，是为了告诉编译器生成 `LDG.128` (128-bit Load) 指令，从而实现 8 倍于标量加载的吞吐效率，并满足 Hopper 架构 TMA 引擎的对齐需求。**

## 4.为什么说如果用 int4 加载：一条指令直接加载 16 字节（128 bits）。这意味着一个线程一次就搬运了 8 个 BF16 元素。
在深度学习模型量化（Quantization）的语境下，我们常说的 `int4` 通常指 **4-bit Integer**（4位整数），即一个字节存两个数。

**但是！**
在 CUDA C++ 编程（以及本代码）的语境下，`int4` **完全是另一个意思**。

它是 CUDA 内置的一个 **向量数据类型 (Vector Data Type)**。

下面是严谨的数学换算和硬件逻辑：

### 1. 拆解 `int4` 的真实定义

在 CUDA 的头文件 `vector_types.h` 中，`int4` 是这样定义的（简化版）：

```cpp
struct int4 {
    int x;
    int y;
    int z;
    int w;
};
```

这里面的 `int` 是标准的 **32-bit (4 Byte) Integer**。

所以，一个 `int4` 变量的总大小是：
$$ \text{Size}(\text{int4}) = 4 \times \text{Size}(\text{int}) = 4 \times 4 \text{ Bytes} = \mathbf{16 \text{ Bytes}} $$
$$ 16 \text{ Bytes} \times 8 \text{ bits/Byte} = \mathbf{128 \text{ bits}} $$

### 2. 拆解 `bf16` 的大小

`bf16` (Brain Floating Point) 顾名思义，是 **16-bit** 的浮点数。
$$ \text{Size}(\text{bf16}) = 16 \text{ bits} = \mathbf{2 \text{ Bytes}} $$

### 3. 为什么是“搬运了 8 个元素”？

现在我们把它们放在一起看。

*   **容器（加载指令）**：一次搬运 16 Bytes (`int4`)
*   **货物（数据元素）**：每个重 2 Bytes (`bf16`)

$$ \frac{\text{一次搬运的总容量}}{\text{单个元素的大小}} = \frac{16 \text{ Bytes}}{2 \text{ Bytes}} = \mathbf{8 \text{ 个元素}} $$

### 4. 代码是如何“欺骗”编译器的？

在 C++ 代码中，内存里存的明明是 `bf16`，怎么能用 `int4` 去读呢？这利用了 **指针强转 (Reinterpret Cast)**。

代码逻辑如下：

1.  **内存视角**：显存里连续排列着 8 个 `bf16`：
    `[BF16_0][BF16_1][BF16_2][BF16_3][BF16_4][BF16_5][BF16_6][BF16_7]`
    总共占据 $2 \times 8 = 16$ 字节。

2.  **指针转换**：
    ```cpp
    // 假设 input_ptr 指向 bf16 数组
    // 强制把这个地址看作是 int4 类型的地址
    int4* vec_ptr = reinterpret_cast<int4*>(input_ptr);
    ```

3.  **加载指令**：
    ```cpp
    int4 data = vec_ptr[0]; // 发起 LDG.128 指令
    ```
    GPU 硬件不管这些位代表什么浮点数，它只知道：“去那个地址，给我一口气抓 128 bit (16 Bytes) 回来，放到寄存器里。”

4.  **数据就在寄存器里了**：
    此时，`data.x` 包含了 `BF16_0` 和 `BF16_1` 的位信息（32 bits），`data.y` 包含了 `BF16_2` 和 `BF16_3`，依此类推。

### 总结图示

```text
显存 (Global Memory):
| Byte 0-1 | Byte 2-3 | Byte 4-5 | Byte 6-7 | Byte 8-9 | Byte 10-11 | Byte 12-13 | Byte 14-15 |
|  BF16_0  |  BF16_1  |  BF16_2  |  BF16_3  |  BF16_4  |   BF16_5   |   BF16_6   |   BF16_7   |
|____________________________________________________________________________________________|
                                            ^
                                            |
                  一条 int4 加载指令 (LDG.128) 一次性全部抓取
                                            |
寄存器 (Registers):                         v
|           int4.x          |           int4.y           | ... (z, w)
| (BF16_1 << 16) | BF16_0   | (BF16_3 << 16) | BF16_2    | ...
```

所以，通过这种方式，原本需要 **8 次** 2 字节的读取操作，被合并成了 **1 次** 16 字节的读取操作，极大地提高了效率。

## 5.昇腾有没有类似NV的TMA的对齐需求？
**是的，昇腾（Ascend）的 MTE 绝对有类似的要求，甚至比起 CUDA 的 `int4` (128-bit) 要求更加严格和“粗粒度”。**

在昇腾架构（达芬奇架构）中，这个概念被称为 **32-Byte 对齐（Block Alignment）**。

如果说 CUDA 的 `int4` 优化是为了凑齐 **16 字节**（128 bits），那么昇腾的 MTE 则是为了凑齐 **32 字节**（256 bits）。

以下是详细的对比和技术解析：

### 1. 核心差异：128-bit vs. 256-bit

*   **CUDA (NVIDIA)**:
    *   **黄金粒度**：**16 Bytes (128 bits)**。
    *   **实现方式**：通过 `int4` 加载指令 (`LDG.128`)。
    *   **对应数据量**：8 个 `bf16`。

*   **Ascend (Huawei)**:
    *   **黄金粒度**：**32 Bytes (256 bits)**。这是昇腾内部总线（AXI）和统一缓冲区（Unified Buffer, UB）管理的最小原子单位，通常称为一个 **Block**。
    *   **实现方式**：通过 **MTE 指令**（如 `DataCopy`）。MTE 在搬运小于 32 字节的数据，或者地址没有对齐到 32 字节时，效率会显著下降。
    *   **对应数据量**：**16 个 `bf16`** (或 16 个 `fp16`)。

### 2. 为什么昇腾要求 32 字节对齐？

这就好比物流运输：

*   **NVIDIA** 的线程像是一个个搬运工，每个搬运工一次最多能抱 **1 个箱子（16字节）**。如果不把 8 个 `bf16` 装进一个箱子（`int4`），搬运工就要跑 8 趟。
*   **Ascend MTE** 像是一辆卡车（DMA 引擎）。这辆卡车的最小装载单位是 **1 个托盘（32字节）**。
    *   如果你只让它运 2 个字节（1 个 `bf16`），卡车也必须拉走整整一个托盘（里面 30 个字节是空的，或者无效数据）。这叫**带宽浪费**。
    *   更糟糕的是，如果你的数据横跨了两个托盘（即**地址未对齐**），卡车就得拉两个托盘才能凑齐你要的那一点数据。

### 3. 具体的技术要求与代价

在昇腾算子开发（Ascend C / TIK）中，如果不满足 32 字节对齐，会引发严重的性能惩罚：

#### A. 只有满足 "32-Byte Aligned" 才能开启全速 Burst 模式
MTE 最快的工作方式是连续突发传输（Burst）。
*   **要求**：起始地址必须能被 32 整除（`addr % 32 == 0`）。
*   **后果**：如果地址不对齐，MTE 硬件内部可能需要将其拆分为多次传输，无法打满带宽。

#### B. 致命的 "Read-Modify-Write" 问题 (针对写入)
这是昇腾架构中一个非常经典的性能坑。
假设你要把计算结果（在 UB 中）写回 Global Memory（GM），但你只写了 16 个字节（半个 Block），或者地址没对齐。
由于 ECC 校验和 DRAM 控制器的特性，MTE 不能只写半个 Block。它必须执行以下流程：
1.  **Read**: 先把目标显存里旧的那 32 字节**读**回来。
2.  **Modify**: 在缓存中把你新写的 16 字节拼进去。
3.  **Write**: 再把完整的 32 字节**写**回显存。

**原本 1 次写操作，变成了“1读 + 1改 + 1写”，带宽效率直接暴跌 50% 以上。**
所以，在昇腾上，输出数据一定要凑够 32 字节的倍数再写回。

### 4. 编程层面如何应对？(Ascend C vs CUDA)

*   **CUDA 写法 (Trick)**:
    开发者需要**显式**地进行类型转换。
    ```cpp
    // 程序员手动强转，欺骗编译器生成 LDG.128
    int4 v = *reinterpret_cast<int4*>(ptr);
    ```

*   **Ascend C 写法 (Constraint)**:
    开发者不需要做类型强转，MTE 是独立引擎，不看类型只看地址。你需要做的是**Padding（填充）**和**Tiling（切分）**。
    ```cpp
    // 假设 total_size = 100 bytes (不是 32 的倍数)
    // 错误做法：直接搬运 100 bytes
    // DataCopy(dst, src, 100); -> 效率低，尾部可能有 Read-Modify-Write

    // 正确做法（在 Host 端 Tiling 时规划）：
    // 将数据 Padding 到 128 bytes (32 * 4)
    // 每次 DataCopy 的长度都是 32 的整数倍
    DataCopy(dst_ub, src_gm, 128); 
    ```

### 总结

DeepEP 代码中输入 `x` 用 `int4` 加载是为了满足 CUDA 的 16 字节向量化要求。

如果你要把这段代码移植到昇腾 910B/C 上，你需要确保：
1.  **数据首地址**是对齐到 **32 字节**的（最好是 64 字节以适应 Cache Line）。
2.  **隐藏层维度 (Hidden Size)** 最好能被 **16** 整除（16 * sizeof(bf16) = 32 Bytes）。
    *   DeepEP 代码里断言 `hidden_int4 % 32 == 0`，这意味着 `hidden` 维度已经是 128 字节对齐了，这**完美满足**昇腾的要求（128 是 32 的倍数）。
3.  在调用 `DataCopy` 时，长度参数应当是 32 字节的倍数。

所以，结论是：**DeepEP 现有的数据对齐策略（为了 TMA 做的 128 字节对齐）已经天然满足并超越了昇腾 MTE 的 32 字节对齐要求，移植时非常友好。**

## 6.zerobuffer的原理？
## 7.DeepEP和DeepGemm如何搭配的？
## 8.MOE层的组成和特点
## 9.bankconflict的原理与具体场景
