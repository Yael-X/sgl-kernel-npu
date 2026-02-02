针对 Linux 内核模块（Kernel Module）编写单元测试（UT）通常有两种流派：

1.  **基于 KUnit / QEMU**：在真实的内核环境中运行，最真实，但环境搭建极其复杂，调试困难，且速度慢。
2.  **基于 User-Space Mocking (推荐)**：在用户态通过 Mock（模拟）内核 API 来测试代码逻辑。这种方式速度快、可调试、易于集成到 CI/CD，是测试**驱动逻辑**（如状态机、锁机制、错误回滚）的最佳实践。

鉴于 Version 2 的代码主要涉及逻辑控制（锁、内存映射管理、回调），我们采用 **GoogleTest (GTest) + Mocking** 的方式在用户态搭建 UT 环境。

---

### 1. 工程目录结构

我们需要构建一个“伪造”的内核环境。请按以下结构创建文件：

```text
npu_rdma_ut/
├── CMakeLists.txt              # 构建脚本
├── src/
│   └── npu_rdma.c              # 你的 Version 2 源代码 (复制过来)
├── test/
│   ├── main.cpp                # 测试入口
│   ├── mock_kernel/            # 模拟的内核头文件
│   │   ├── linux/
│   │   │   ├── init.h
│   │   │   ├── module.h
│   │   │   ├── kernel.h
│   │   │   ├── slab.h
│   │   │   ├── mutex.h
│   │   │   ├── dma-mapping.h
│   │   │   └── ... (其他依赖)
│   │   └── ascend_kernel_hal.h # 模拟的 HAL 头文件
│   └── ndr_test.cpp            # 具体的测试用例
```

---

### 2. 模拟内核头文件 (Mocks)

这是最关键的一步。我们需要定义源代码中用到的宏、结构体和函数桩（Stub）。

**注意**：为了方便演示，我将所有 Mock 逻辑整合到一个核心头文件 `test/mock_env.h` 中，实际工程中通常分散在 `mock_kernel/` 下。

创建一个 `test/mock_env.h` (作为所有 Mock 头的聚合):

```cpp
/* test/mock_env.h */
#pragma once

#include <iostream>
#include <vector>
#include <cstring>
#include <functional>

// ================= 类型定义 =================
typedef uint64_t u64;
typedef uint32_t u32;
typedef uint64_t dma_addr_t;
typedef size_t size_t;
typedef void* (*ib_register_peer_memory_client_t)(void*, void*);
typedef void (*ib_unregister_peer_memory_client_t)(void*);

#define __init
#define __exit
#define KERN_ERR  "ERROR"
#define KERN_INFO "INFO"
#define GFP_KERNEL 0
#define THIS_MODULE 0
#define EXPORT_SYMBOL(x)
#define MODULE_LICENSE(x)
#define module_init(x)
#define module_exit(x)

// ================= 结构体模拟 =================
struct mutex { int locked; };
struct rw_semaphore { int read_locked; int write_locked; };
struct device {};
struct task_struct { int tgid; int pid; };
struct p2p_page_table {
    u32 page_num;
    u64 page_size;
    struct { u64 pa; } *pages_info;
};
struct scatterlist {
    dma_addr_t dma_address;
    unsigned int length;
};
struct sg_table {
    struct scatterlist *sgl;
    unsigned int nents;
};
struct kprobe {
    const char *symbol_name;
    void *addr;
};
struct peer_memory_client {
    const char* name;
    const char* version;
    void* acquire;
    void* get_pages;
    void* dma_map;
    void* dma_unmap;
    void* put_pages;
    void* get_page_size;
    void* release;
};

// 全局模拟状态控制
extern struct task_struct* current;
extern bool g_fail_dma_map; // 控制 DMA 映射是否失败
extern int g_dma_map_fail_index; // 控制第几次映射失败

// ================= 函数模拟 (C Linkage) =================
extern "C" {
    // 打印
    int printk(const char *fmt, ...);
    
    // 内存
    void *kzalloc(size_t size, int flags);
    void *kcalloc(size_t n, size_t size, int flags);
    void kfree(const void *objp);
    
    // 锁
    void mutex_init(struct mutex *lock);
    void mutex_lock(struct mutex *lock);
    void mutex_unlock(struct mutex *lock);
    void init_rwsem(struct rw_semaphore *sem);
    void down_read(struct rw_semaphore *sem);
    void up_read(struct rw_semaphore *sem);
    void down_write(struct rw_semaphore *sem);
    void up_write(struct rw_semaphore *sem);
    
    // SG Table
    int sg_alloc_table(struct sg_table *table, unsigned int nents, int gfp_mask);
    void sg_free_table(struct sg_table *table);
    
    // DMA
    dma_addr_t dma_map_resource(struct device *dev, u64 phys_addr, size_t size, int dir, unsigned long attrs);
    void dma_unmap_resource(struct device *dev, dma_addr_t addr, size_t size, int dir, unsigned long attrs);
    int dma_mapping_error(struct device *dev, dma_addr_t dma_addr);
    
    // Module
    void __module_get(void *module);
    void module_put(void *module);
    
    // KProbes
    int register_kprobe(struct kprobe *p);
    void unregister_kprobe(struct kprobe *p);
    
    // IB (Mocking external symbols)
    void* ib_register_peer_memory_client(const struct peer_memory_client *client, void *cb);
    void ib_unregister_peer_memory_client(void *reg_handle);
    
    // SecureC (Mock)
    int memset_s(void *dest, size_t destMax, int c, size_t count);
}

// 辅助宏，用于遍历 SG
#define for_each_sg(sgl, sg, nents, i) \
    for ((i) = 0, (sg) = (sgl); (i) < (nents); (i)++, (sg)++)

#define sg_dma_address(sg) ((sg)->dma_address)
#define sg_dma_len(sg)     ((sg)->length)
#define sg_set_page(sg, page, len, offset)
#define WARN_ON(x) (x)
#define WARN_ONCE(x, ...) (x)
#define dev_err(...)
#define dev_name(x) "mock_dev"
```

---

### 3. 实现 Mock 逻辑 (`test/mock_impl.cpp`)

这里实现上述声明的函数，用来追踪调用、注入错误。

```cpp
#include "mock_env.h"

// 全局变量定义
struct task_struct g_current_task = {100, 101};
struct task_struct* current = &g_current_task;

bool g_fail_dma_map = false;
int g_dma_map_fail_index = -1;
int g_dma_map_call_count = 0;
int g_dma_unmap_call_count = 0; // 用于验证回滚

extern "C" {
    int printk(const char *fmt, ...) { return 0; }
    
    void *kzalloc(size_t size, int flags) { return calloc(1, size); }
    void *kcalloc(size_t n, size_t size, int flags) { return calloc(n, size); }
    void kfree(const void *objp) { free((void*)objp); }
    
    void mutex_init(struct mutex *lock) { lock->locked = 0; }
    void mutex_lock(struct mutex *lock) { lock->locked = 1; }
    void mutex_unlock(struct mutex *lock) { lock->locked = 0; }
    
    // 简化 rwsem，测试中只要编译通过即可，单线程测试不需要真实阻塞
    void init_rwsem(struct rw_semaphore *sem) {}
    void down_read(struct rw_semaphore *sem) {}
    void up_read(struct rw_semaphore *sem) {}
    void down_write(struct rw_semaphore *sem) {}
    void up_write(struct rw_semaphore *sem) {}
    
    int sg_alloc_table(struct sg_table *table, unsigned int nents, int gfp_mask) {
        table->nents = nents;
        table->sgl = (struct scatterlist*)calloc(nents, sizeof(struct scatterlist));
        return 0;
    }
    
    void sg_free_table(struct sg_table *table) {
        if(table->sgl) free(table->sgl);
        table->sgl = NULL;
    }
    
    dma_addr_t dma_map_resource(struct device *dev, u64 phys_addr, size_t size, int dir, unsigned long attrs) {
        if (g_fail_dma_map && g_dma_map_call_count == g_dma_map_fail_index) {
            g_dma_map_call_count++;
            return 0xFFFFFFFFFFFFFFFF; // Error
        }
        g_dma_map_call_count++;
        return (dma_addr_t)(phys_addr + 0x10000); // Fake DMA address
    }
    
    int dma_mapping_error(struct device *dev, dma_addr_t dma_addr) {
        return dma_addr == 0xFFFFFFFFFFFFFFFF;
    }
    
    void dma_unmap_resource(struct device *dev, dma_addr_t addr, size_t size, int dir, unsigned long attrs) {
        g_dma_unmap_call_count++;
    }
    
    void __module_get(void *module) {}
    void module_put(void *module) {}
    
    // 这里是关键：模拟 kprobe 注册，返回我们自定义的 HAL 函数地址
    // 前向声明 Mock HAL
    int mock_hal_get_pages(u64 addr, u64 len, void (*cb)(void*), void *data, struct p2p_page_table **pt);
    int mock_hal_put_pages(struct p2p_page_table *pt);

    int register_kprobe(struct kprobe *p) {
        // 简单粗暴：直接把地址指向一个通用查找函数
        // 在真实代码中 g_kallsymsLookupName = (uint64_t(*)(const char *))kp.addr;
        // 我们让 kp.addr 指向一个能处理字符串并返回 mock_hal 函数指针的函数
        return 0;
    }
    void unregister_kprobe(struct kprobe *p) {}
    
    // 这是一个特殊的 Helper，用于模拟 kallsyms_lookup_name
    u64 mock_kallsyms_lookup(const char *name) {
        if (strcmp(name, "hal_kernel_p2p_get_pages") == 0) return (u64)mock_hal_get_pages;
        if (strcmp(name, "hal_kernel_p2p_put_pages") == 0) return (u64)mock_hal_put_pages;
        return 0;
    }
    
    void* ib_register_peer_memory_client(const struct peer_memory_client *client, void *cb) {
        return (void*)0x1234; // Return a fake handle
    }
    void ib_unregister_peer_memory_client(void *reg_handle) {}
    
    int memset_s(void *dest, size_t destMax, int c, size_t count) {
        memset(dest, c, count);
        return 0;
    }
}
```

---

### 4. 编写 HAL 桩代码与测试用例 (`test/ndr_test.cpp`)

这里使用 GTest 编写具体的测试逻辑。需要一些 Trick 来包含源代码。

```cpp
#include <gtest/gtest.h>
#include "mock_env.h"

// ==========================================
// 核心技巧：通过宏定义替换，直接包含 .c 源代码
// 这样我们就可以测试 static 函数，并且使用我们的 Mock 头文件
// ==========================================

// 1. 强制让源代码 include 我们的 mock 头文件，而不是系统头文件
// 在 CMakeLists.txt 中通过 include_directories 实现，
// 或者在这里定义宏来阻止标准头文件被处理（如果 mock 做得不够彻底）

// 2. 包含 mock 实现
#include "mock_impl.cpp"

// 3. 定义 HAL Mock 逻辑
struct p2p_page_table* g_mock_pt = nullptr;

extern "C" {
    int mock_hal_get_pages(u64 addr, u64 len, void (*cb)(void*), void *data, struct p2p_page_table **pt) {
        // 创建一个假的页表
        g_mock_pt = (struct p2p_page_table*)calloc(1, sizeof(struct p2p_page_table));
        g_mock_pt->page_size = 4096;
        g_mock_pt->page_num = len / 4096;
        g_mock_pt->pages_info = (decltype(g_mock_pt->pages_info))calloc(g_mock_pt->page_num, sizeof(u64));
        
        for(u32 i=0; i<g_mock_pt->page_num; ++i) {
            g_mock_pt->pages_info[i].pa = 0x10000000 + i * 4096;
        }
        
        *pt = g_mock_pt;
        return 0; // Success
    }

    int mock_hal_put_pages(struct p2p_page_table *pt) {
        if (pt) {
            if (pt->pages_info) free(pt->pages_info);
            free(pt);
        }
        g_mock_pt = nullptr;
        return 0;
    }
}

// 4. Hack kallsyms 的处理
// 原代码中：g_kallsymsLookupName = (uint64_t(*)(const char *))kp.addr;
// 我们需要修改源代码或者在这里做手脚。
// 最简单的办法是：在编译源代码时，让它链接到我们的 mock_register_kprobe，
// 并在 mock_register_kprobe 中把 kp.addr 赋值为 &mock_kallsyms_lookup

// 重写 mock_impl.cpp 中的 register_kprobe 以适应代码逻辑
extern "C" {
    u64 mock_kallsyms_lookup(const char *name); // Forward decl
    
    // 覆盖掉 mock_impl.cpp 中的定义 (C++ 允许覆盖弱符号，或者是直接修改 impl)
    // 这里为了演示方便，假设 mock_impl.cpp 的代码是直接写在这里的
}

// 5. 包含被测源文件
// 注意：需要将源文件中的 #include <...> 修改为能被 Mock 拦截的形式，
// 或者在 CMake 中配置 include path。
// 假设我们通过 -I 使得 #include <linux/...> 指向了 test/mock_kernel/linux/...

// 为了能编译源文件，需要定义一些缺失的宏
#define DMA_BIDIRECTIONAL 0
#define ENOENT 2
#define EINVAL 22
#define ENOMEM 12
#define EBUSY 16
#define EFAULT 14

// 包含被测代码！
// 包含之前先 undef 冲突的宏
#undef module_init
#undef module_exit

// 这里的路径取决于你的实际放置位置
#include "../src/npu_rdma.c"

// 修正：源文件中的 g_kallsymsLookupName 是函数指针变量。
// 我们在 TestFixture Setup 中手动给它赋值，或者依赖 Mock 的 kprobe 逻辑。

class NDRTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 重置全局状态
        g_fail_dma_map = false;
        g_dma_map_fail_index = -1;
        g_dma_map_call_count = 0;
        g_dma_unmap_call_count = 0;
        
        // 关键：为了让代码能跑，我们需要手动初始化 kallsyms 指针，
        // 或者调用 NDRPeerMemAgentInit() 来走一遍初始化流程
        // 为了单元测试的纯粹性，我们也可以直接给函数指针赋值
        hal_get_pages_func = mock_hal_get_pages;
        hal_put_pages_func = mock_hal_put_pages;
    }

    void TearDown() override {
        // 清理
        if (g_mock_pt) {
             mock_hal_put_pages(g_mock_pt);
        }
    }
    
    // Helper to get client context
    void* acquire_context(size_t size) {
        void* ctx = nullptr;
        NDRPeerMemAcquire(0x10000000, size, nullptr, nullptr, &ctx);
        return ctx;
    }
};

// 测试用例 1: 获取内存上下文 (Acquire)
TEST_F(NDRTest, AcquireSuccess) {
    void* ctx = nullptr;
    size_t size = 4096 * 10;
    
    int ret = NDRPeerMemAcquire(0x10000000, size, nullptr, nullptr, &ctx);
    
    EXPECT_EQ(ret, true);
    EXPECT_NE(ctx, nullptr);
    
    // 验证内部状态
    struct svm_agent_context* s_ctx = (struct svm_agent_context*)ctx;
    EXPECT_EQ(s_ctx->len, size);
    EXPECT_EQ(s_ctx->inited_flag, 1);
    EXPECT_NE(s_ctx->page_table, nullptr);
    
    NDRPeerMemRelease(ctx);
}

// 测试用例 2: DMA 映射成功
TEST_F(NDRTest, DmaMapSuccess) {
    void* ctx = acquire_context(4096 * 5); // 5 Pages
    struct sg_table sg_head;
    sg_alloc_table(&sg_head, 5, 0); // Mock alloc
    int nmap = 0;
    struct device dev;
    
    int ret = NDRPeerMemDmaMap(&sg_head, ctx, &dev, 0, &nmap);
    
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(nmap, 5);
    EXPECT_EQ(g_dma_map_call_count, 5);
    
    // Clean up
    NDRPeerMemDmaUnmap(&sg_head, ctx, &dev);
    NDRPeerMemRelease(ctx);
    sg_free_table(&sg_head);
}

// 测试用例 3: DMA 映射部分失败触发回滚 (Rollback)
// 这是 Version 2 代码中最精彩的部分，必须测
TEST_F(NDRTest, DmaMapPartialFailureRollback) {
    void* ctx = acquire_context(4096 * 10); // 10 Pages
    struct sg_table sg_head;
    sg_alloc_table(&sg_head, 10, 0);
    int nmap = 0;
    struct device dev;
    
    // 设定：第 7 次映射失败 (Index 6)
    g_fail_dma_map = true;
    g_dma_map_fail_index = 6; 
    
    int ret = NDRPeerMemDmaMap(&sg_head, ctx, &dev, 0, &nmap);
    
    EXPECT_EQ(ret, -EFAULT); // 期望返回错误
    
    // 验证：应该调用了 map 7 次 (0-5 成功, 6 失败)
    EXPECT_EQ(g_dma_map_call_count, 7);
    
    // 关键验证：应该调用 unmap 6 次 (回滚前 6 个成功的)
    EXPECT_EQ(g_dma_unmap_call_count, 6);
    
    NDRPeerMemRelease(ctx);
    sg_free_table(&sg_head);
}

// 测试用例 4: 重复 Map 检测
TEST_F(NDRTest, DoubleMapError) {
    void* ctx = acquire_context(4096);
    struct sg_table sg1, sg2;
    sg_alloc_table(&sg1, 1, 0);
    sg_alloc_table(&sg2, 1, 0);
    struct device dev;
    int nmap;
    
    // 第一次 Map
    EXPECT_EQ(NDRPeerMemDmaMap(&sg1, ctx, &dev, 0, &nmap), 0);
    
    // 第二次 Map (应该失败)
    EXPECT_EQ(NDRPeerMemDmaMap(&sg2, ctx, &dev, 0, &nmap), -EBUSY);
    
    NDRPeerMemDmaUnmap(&sg1, ctx, &dev);
    NDRPeerMemRelease(ctx);
}
```

---

### 5. `CMakeLists.txt` 构建脚本

使用 FetchContent 自动下载 GoogleTest。

```cmake
cmake_minimum_required(VERSION 3.14)
project(npu_rdma_ut)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)

# 包含源代码目录 (但不直接编译 .c 文件，而是通过 include 包含在 test cpp 中)
include_directories(src)
include_directories(test)
include_directories(test/mock_kernel) # 指向你的模拟头文件目录

# GoogleTest 依赖
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
)
FetchContent_MakeAvailable(googletest)

# 定义宏以适配内核代码在用户态编译
add_definitions(
    -D__KERNEL__ 
    -DMODULE 
    -DKBUILD_MODNAME="ndr_peer_mem"
)

# 测试可执行文件
add_executable(run_tests 
    test/ndr_test.cpp
    test/mock_impl.cpp 
)

# 链接 GTest
target_link_libraries(run_tests gtest_main)
```

### 6. 如何运行

1.  **准备环境**：安装 CMake, g++, make。
2.  **创建目录**：按照第 1 步的结构创建目录和文件。
3.  **补充头文件**：你需要创建 `test/mock_kernel/linux/` 目录，并在里面放上空的或者包含基本定义的 `.h` 文件（`init.h`, `module.h`, `dma-mapping.h` 等）。你可以把 `mock_env.h` 的内容拆分进去，或者在每个头文件里写 `#include "../../mock_env.h"`。
    *   *简便方法*：在 `ndr_test.cpp` 里包含源文件之前，手动定义所有需要的宏，跳过 include 检查，但这比较脏。推荐建立 mock header 目录。
4.  **编译与运行**：

```bash
mkdir build
cd build
cmake ..
make
./run_tests
```

### 总结

这个方案的核心在于：
1.  **用户态编译**：通过 `-D__KERNEL__` 和 Mock 头文件，把内核模块当作普通 C++ 程序编译。
2.  **直接 Include 源文件**：`#include "../src/npu_rdma.c"` 允许我们访问 `static` 函数和变量，这对于测试内部状态（如 `svm_agent_context`）至关重要。
3.  **Mock 关键路径**：重点 Mock 了 `dma_map_resource`（用于测试回滚）和 `hal_get_pages`（用于解除对硬件的依赖）。
4.  **测试逻辑覆盖**：特别是 Version 2 中优秀的错误回滚逻辑 (`DmaMapPartialFailureRollback` 测试用例)，这是必须要覆盖的。