这是一个非常关键的发现。云脉（YunScale）在标准的 NVIDIA/Mellanox `peer_memory_client` 结构体末尾扩展了两个新的函数指针。

如果直接使用原有的 Version 2 代码（它认为结构体较短）去注册，**会导致严重的内核内存越界访问**。当云脉驱动尝试访问那两个新增加的成员时，会读取到未知的内存数据，可能引发 Kernel Panic。

以下是针对云脉网卡适配 NPU Direct RDMA 的设计分析文档。

---

# NPU Direct RDMA 适配云脉（YunScale）网卡设计分析文档

## 1. 概述
本项目旨在将基于 Ascend NPU 的 Direct RDMA（Peer Memory）驱动移植至云脉（MCR/XScale）网卡环境。根据代码审计，云脉驱动层虽然保留了 `ib_register_peer_memory_client` 接口名称，但修改了核心结构体 `struct peer_memory_client` 的定义，破坏了二进制兼容性。本文档详细描述适配方案。

## 2. 差异分析 (Gap Analysis)

### 2.1 结构体定义差异
标准 MOFED (Mellanox) 定义与云脉定义的对比如下：

| 字段位置 | 标准 MOFED 定义 | 云脉 (YunScale) 定义 | 差异影响 |
| :--- | :--- | :--- | :--- |
| 0-1 | `name`, `version` | `name`, `version` | 无 |
| 2-8 | 基础回调 (`acquire` ... `release`) | 基础回调 (`acquire` ... `release`) | 无 |
| **9 (新增)** | **不存在** | **`get_context_private_data`** | **致命：内存越界** |
| **10 (新增)** | **不存在** | **`put_context_private_data`** | **致命：内存越界** |

**风险评估**：
云脉驱动在注册或使用 Peer Client 时，可能会检查或调用最后这两个函数指针。如果使用旧结构体，云脉驱动会读取到越界的堆栈数据。如果该数据非空，内核将跳转到非法地址执行，导致系统崩溃。

### 2.2 接口符号
*   **注册接口**：`ib_register_peer_memory_client`
*   **注销接口**：`ib_unregister_peer_memory_client`
*   **现状**：云脉驱动导出了这两个符号，且函数签名（参数列表）与标准一致，均包含 `invalidate_callback`。

## 3. 适配设计方案

### 3.1 核心策略
基于 **Version 2** 代码进行修改。不依赖系统头文件 `<rdma/peer_mem.h>`（防止系统头文件未更新或版本不匹配），而是**在 NPU 驱动源码内部重新定义完全匹配云脉的结构体**。

### 3.2 数据结构重定义
在代码中显式定义适配云脉的结构体，命名为 `yun_peer_memory_client` 或直接覆盖原定义。

```c
/* 适配云脉定义的结构体 */
struct yun_peer_memory_client {
    char    name[IB_PEER_MEMORY_NAME_MAX];
    char    version[IB_PEER_MEMORY_VER_MAX];
    // ... 原有回调函数保持不变 ...
    int (*acquire)(...);
    int (*get_pages)(...);
    int (*dma_map)(...);
    int (*dma_unmap)(...);
    void (*put_pages)(...);
    unsigned long (*get_page_size)(...);
    void (*release)(...);
    
    /* 新增适配字段 */
    void* (*get_context_private_data)(u64 peer_id);
    void (*put_context_private_data)(void *context);
};
```

### 3.3 新增回调处理
由于 NPU Direct RDMA 的业务逻辑并不依赖 `peer_id` 来获取私有数据（上下文已经在 `acquire` 阶段通过 `client_context` 传递），我们不需要实现这两个新函数的具体逻辑。

*   **策略**：将这两个函数指针显式初始化为 `NULL`。
*   **假设**：云脉驱动在调用这两个回调前会进行非空检查（`if (client->func) client->func(...)`）。这是内核驱动开发的通用规范。

### 3.4 符号查找机制
继续沿用 Version 2 的 `kallsyms_lookup_name` + `kprobes` 机制。
*   **原因**：云脉的符号是由 `xsc` 或类似模块导出的，不一定在 `Module.symvers` 白名单中。动态查找最为稳妥。
*   **注意**：需要验证云脉驱动加载顺序。建议在 `insmod npu_peer_mem.ko` 前确保云脉驱动已加载。

## 4. 实施步骤 (代码修改指南)

以下是针对 **Version 2** 代码的具体修改建议：

### 步骤 1: 移除原有的头文件依赖
注释掉 `#include <rdma/peer_mem.h>`，防止定义冲突。

### 步骤 2: 嵌入云脉版结构体定义
将你提取到的 `peer_mem.h` 内容（结构体部分）直接复制到 `.c` 文件头部，并修改宏定义防止冲突。

### 步骤 3: 修改初始化结构体
在 `roce_peer_mem_client` 初始化时，显式补充最后两个字段。

```c
static struct yun_peer_memory_client roce_peer_mem_client = {
    .name = DRV_ROCE_PEER_MEM_NAME,
    .version = DRV_ROCE_PEER_MEM_VERSION,
    .acquire = NDRPeerMemAcquire,
    .get_pages = NDRPeerMemGetPages,
    .dma_map = NDRPeerMemDmaMap,
    .dma_unmap = NDRPeerMemDmaUnmap,
    .put_pages = NDRPeerMemPutPages,
    .get_page_size = NDRPeerMemGetPageSize,
    .release = NDRPeerMemRelease,
    
    /* 关键适配点：显式初始化为 NULL */
    .get_context_private_data = NULL,
    .put_context_private_data = NULL, 
};
```

### 步骤 4: 强制类型转换注册
由于 `ib_register_peer_memory_client` 的符号定义在内核中可能还是指向旧的结构体（如果我们通过 kallsyms 找到它），或者我们本地定义的结构体名字不同，在调用时进行 `void *` 强转即可，C 语言只看内存布局。

```c
/* 查找符号 */
typedef void *(*ib_register_func_t)(const void *peer_client, void *invalidate_callback);
ib_register_func_t register_func = (ib_register_func_t)g_kallsymsLookupName("ib_register_peer_memory_client");

/* 注册 */
reg_handle = register_func(&roce_peer_mem_client, &mem_invalidate_callback);
```

## 5. 验证与调试

### 5.1 编译检查
确保编译通过，且 `sizeof(struct yun_peer_memory_client)` 的大小比原标准结构体大 16 字节（2个指针大小）。可以在 `init` 函数中打印一下大小：
```c
NDR_PEER_MEM_INFO("Client Struct Size: %lu\n", sizeof(roce_peer_mem_client));
```

### 5.2 运行时检查
1.  加载模块：`insmod npu_peer_mem.ko`。
2.  观察 `dmesg`：
    *   如果没有报错，说明注册成功。
    *   如果出现 Crash，且堆栈指向 `xscale` 驱动内部，说明它可能无条件调用了那两个 NULL 指针。
    *   **Fallback 方案**：如果设置为 NULL 导致崩溃，则需要实现两个空桩函数（Stub Functions），返回 NULL 或 0。

```c
/* 备用桩函数 */
void *NDRStubGetContext(u64 peer_id) { return NULL; }
void NDRStubPutContext(void *context) { return; }
```

## 6. 总结
此次适配的核心风险在于**结构体内存布局不一致**。通过在 NPU 驱动侧手动对齐云脉的结构体定义，并将新增函数指针置空，理论上可以实现无缝迁移。Version 2 代码的逻辑架构优秀，只需修改定义部分即可。