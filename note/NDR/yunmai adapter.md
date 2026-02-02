这是一个非常关键的发现。云脉（MCR）的 `peer_memory_client` 结构体确实比 Mellanox（CX）标准定义的多了两个尾部成员。

基于你提供的 Version 2 代码基础和云脉的头文件定义，我为你编写了一份详细的 **《NPU Direct RDMA 适配云脉/CX网卡 统一设计方案分析文档》**。

---

# NPU Direct RDMA 适配云脉(MCR)与CX网卡 统一设计方案

## 1. 概述
当前 NPU Direct RDMA 代码（Version 2）是基于 Mellanox OFED (MOFED) 的标准 Peer Memory 接口开发的。为了在同一套代码中同时支持 NVIDIA CX 系列网卡和云脉 MCR 系列网卡，需要解决接口定义差异、符号导出位置不同以及结构体二进制兼容性问题。

## 2. 差异分析 (Gap Analysis)

### 2.1. 接口定义差异
通过对比标准 MOFED 头文件与云脉 `peer_mem.h`，发现核心结构体 `struct peer_memory_client` 存在差异：

| 字段/特性 | Mellanox (CX/MOFED) | 云脉 (MCR/XScale) | 差异评估 |
| :--- | :--- | :--- | :--- |
| **基础字段** | `name`, `version`, `acquire`...`release` | 完全一致 | **兼容** |
| **扩展字段** | 无 | `get_context_private_data`<br>`put_context_private_data` | **云脉多了两个函数指针** |
| **定义位置** | 结构体尾部 | 结构体尾部 | **具备向后兼容性条件** |
| **核心上下文** | `u64 core_context` | `u64 core_context` (宏 `PEER_MEM_U64_CORE_CONTEXT`) | **兼容** |

### 2.2. 符号导出差异
*   **Mellanox**: 符号 `ib_register_peer_memory_client` 通常由 `ib_core` 模块导出。
*   **云脉**: 符号由 `peer_mem` (属于 `xscale` 驱动的一部分) 导出。

### 2.3. 风险点
如果直接使用 Mellanox 的结构体定义去注册云脉的驱动，云脉驱动可能会访问结构体尾部越界，读取到野指针并尝试调用，导致 Kernel Panic。

## 3. 统一设计策略

为了实现一套代码同时适配两种网卡，建议采用 **"超集结构体 + 动态符号查找"** 的策略。

### 3.1. 策略核心：使用“超集”结构体
由于 C 语言中结构体内存布局是顺序的，且云脉新增的字段位于尾部。我们可以定义一个包含所有字段的 **“最大公约数”结构体**。
*   **在 CX 网卡上运行时**：驱动只访问前 9 个字段，忽略尾部新增的字段（安全）。
*   **在 MCR 网卡上运行时**：驱动能看到所有字段，我们将新增字段置为 `NULL`，驱动会跳过调用（安全）。

### 3.2. 策略核心：完全动态符号解析
不要在编译时链接 `ib_register_peer_memory_client`（即不要直接调用它），而是通过 `kallsyms_lookup_name` 动态获取该函数的地址。
*   **优点**：编译时不需要依赖云脉或 Mellanox 的特定 `Module.symvers` 文件。
*   **优点**：运行时自动适应，只要内核里有这个符号就能跑，不用管它是 `ib_core` 导出的还是 `xscale` 导出的。

## 4. 代码实现方案 (基于 Version 2 修改)

请基于你之前的 **Version 2** 代码进行以下关键修改。

### 4.1. 修改头文件定义
在代码头部，**不要**包含系统的 `peer_mem.h`（防止冲突），而是把云脉的定义完整复制进来作为通用定义。

```c
/* 定义兼容两者的超集结构体 */
struct peer_memory_client_compat {
    char    name[64];
    char    version[16];
    int (*acquire)(unsigned long addr, size_t size, void *peer_mem_private_data,
               char *peer_mem_name, void **client_context);
    int (*get_pages)(unsigned long addr, size_t size, int write, int force,
             struct sg_table *sg_head, void *client_context, u64 core_context);
    int (*dma_map)(struct sg_table *sg_head, void *client_context,
               struct device *dma_device, int dmasync, int *nmap);
    int (*dma_unmap)(struct sg_table *sg_head, void *client_context,
             struct device  *dma_device);
    void (*put_pages)(struct sg_table *sg_head, void *client_context);
    unsigned long (*get_page_size)(void *client_context);
    void (*release)(void *client_context);
    
    /* 云脉 MCR 扩展字段 - 在 CX 网卡上会被忽略 */
    void* (*get_context_private_data)(u64 peer_id);
    void (*put_context_private_data)(void *context);
};

/* 定义回调函数类型 */
typedef int (*invalidate_peer_memory_t)(void *reg_handle, u64 core_context);
typedef void *(*ib_register_peer_memory_client_t)(const struct peer_memory_client_compat *peer_client,
                                                  invalidate_peer_memory_t *invalidate_callback);
typedef void (*ib_unregister_peer_memory_client_t)(void *reg_handle);
```

### 4.2. 修改全局变量与查找逻辑

在 Version 2 的 `NDRPeerMemKallsymLookup` 函数中，增加对注册函数的查找。

```c
static ib_register_peer_memory_client_t func_ib_register_peer_memory_client = NULL;
static ib_unregister_peer_memory_client_t func_ib_unregister_peer_memory_client = NULL;

static int NDRPeerMemKallsymLookup(void)
{
    // ... 原有的 probe 代码 ...
    
    // 1. 查找 HAL 接口 (保持不变)
    hal_get_pages_func = (typeof(hal_get_pages_t))g_kallsymsLookupName("hal_kernel_p2p_get_pages");
    hal_put_pages_func = (typeof(hal_put_pages_t))g_kallsymsLookupName("hal_kernel_p2p_put_pages");

    // 2. 查找 RDMA Peer Mem 接口 (新增)
    // 这种方式兼容 CX (ib_core) 和 MCR (xscale)
    func_ib_register_peer_memory_client = 
        (ib_register_peer_memory_client_t)g_kallsymsLookupName("ib_register_peer_memory_client");
    
    func_ib_unregister_peer_memory_client = 
        (ib_unregister_peer_memory_client_t)g_kallsymsLookupName("ib_unregister_peer_memory_client");

    if (!func_ib_register_peer_memory_client || !func_ib_unregister_peer_memory_client) {
        NDR_PEER_MEM_ERR("Failed to lookup ib_register/unregister symbols.\n");
        return -ENOENT;
    }
    
    return 0;
}
```

### 4.3. 修改 Client 初始化

初始化时，显式将新增的扩展字段置为 NULL。

```c
static struct peer_memory_client_compat roce_peer_mem_client = {
    .name = DRV_ROCE_PEER_MEM_NAME,
    .version = DRV_ROCE_PEER_MEM_VERSION,
    .acquire = NDRPeerMemAcquire,
    .get_pages = NDRPeerMemGetPages,
    .dma_map = NDRPeerMemDmaMap,
    .dma_unmap = NDRPeerMemDmaUnmap,
    .put_pages = NDRPeerMemPutPages,
    .get_page_size = NDRPeerMemGetPageSize,
    .release = NDRPeerMemRelease,
    /* 关键：显式初始化扩展字段为 NULL */
    .get_context_private_data = NULL,
    .put_context_private_data = NULL,
};
```

### 4.4. 修改加载/卸载函数

使用函数指针调用，而不是直接调用。

```c
static int __init NDRPeerMemAgentInit(void)
{
    // ...
    rc = NDRPeerMemKallsymLookup();
    if (rc) goto dev_reg_err;

    // 使用函数指针调用
    reg_handle = func_ib_register_peer_memory_client(&roce_peer_mem_client, &mem_invalidate_callback);
    // ...
}

static void __exit NDRPeerMemAgentDeInit(void)
{
    if (reg_handle && func_ib_unregister_peer_memory_client) {
        func_ib_unregister_peer_memory_client(reg_handle);
    }
    // ...
}
```

## 5. 验证与 Checklist

### 5.1. 编译验证
*   该方案不需要链接任何外部 RDMA 库 (`Module.symvers`)，仅依赖 Linux Kernel Headers。
*   编译出的 `.ko` 应该能在没有安装网卡驱动的机器上加载（虽然查找符号会失败，但不会报 `Unknown symbol` 错误）。

### 5.2. 运行验证 (云脉环境)
1.  加载云脉驱动。
2.  加载修改后的 NPU 驱动。
3.  查看 `dmesg`。
    *   预期：`NDRPeerMemKallsymLookup` 成功找到符号。
    *   预期：`ib_register` 返回成功句柄。

### 5.3. 运行验证 (CX环境)
*   由于我们定义的结构体前部布局与 CX 完全一致，且 CX 驱动只读取它知道的长度，额外的 NULL 指针不会产生副作用。

## 6. 总结
这个 Gap 实际上是**结构体尾部扩展**，属于良性的兼容性问题。通过**定义超集结构体**并配合**Kprobes 动态符号查找**，可以完美实现“一份源码，两处运行”，无需维护两个分支。

请按照上述 **4.1 ~ 4.4** 的代码片段修改你的 Version 2 代码即可。