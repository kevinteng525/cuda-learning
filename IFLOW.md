# iFlow 项目上下文 (IFLOW.md)

## 项目概述

此目录 (`cuda-learning`) 是一个用于学习和实践 CUDA 编程的代码库。它包含基础的 CUDA 示例代码、本地 C++ 开发环境配置指南以及远程 CUDA 调试环境的完整搭建指南。

项目旨在为开发者提供一个从本地开发到云端 GPU 实例调试的完整学习路径，特别适用于 macOS 用户通过 VS Code 进行 CUDA 应用的开发与调试。

## 目录结构

```
.
├── Makefile                      # 顶层 Makefile，用于构建和运行 CUDA 示例
├── build_and_run.sh             # 用于构建和运行 CUDA 示例的 Bash 脚本
├── sample/                      # CUDA 示例代码目录
│   └── vector_add.cu            # 基础的 CUDA 向量加法示例
├── cpp-sample/                  # 本地 C++ 示例代码目录 (用于 VS Code 本地调试配置)
│   ├── Makefile
│   ├── main.cpp
│   └── .vscode/                 # 本地 C++ 调试配置
│       ├── tasks.json
│       └── launch.json
├── .vscode/                     # 顶层 VS Code 配置 (用于 CUDA 示例)
│   ├── tasks.json
│   └── launch.json
├── CUDA远程调试环境搭建指南.md    # 远程 CUDA 调试环境搭建完整指南 (阿里云 + VS Code)
└── VSCode-CPP-Setup.md          # 本地 C++ 开发环境配置指南 (macOS)
```

## 项目类型

这是一个以 CUDA 和 C++ 为核心的代码学习项目，专注于并行计算和 GPU 编程。项目同时提供了本地和远程开发/调试的配置说明。

## 构建与运行

### CUDA 示例 (位于 `sample/`)

**使用 Makefile:**
```bash
make build    # 构建 sample/vector_add.cu
make run      # 构建并运行
make clean    # 清理生成的二进制文件
```

**使用脚本:**
```bash
./build_and_run.sh
```

**手动编译 (使用 `nvcc`):**
```bash
nvcc -g -G sample/vector_add.cu -o sample/vector_add
sample/sample/vector_add
```

### C++ 示例 (位于 `cpp-sample/`)

**使用 Makefile:**
```bash
cd cpp-sample
make          # 构建
make run      # 运行
make clean    # 清理
```

**使用 VS Code Task:**
在 VS Code 中，按 `F1` -> `Tasks: Run Build Task` -> 选择 `build: cpp-sample`。

## 开发与调试

### 本地 C++ 开发 (macOS)

1.  安装 Xcode 命令行工具: `xcode-select --install`
2.  在 VS Code 中安装 `C/C++` 和 `CodeLLDB` 扩展。
3.  打开 `cpp-sample` 目录，使用 `Makefile` 或 VS Code Task 构建。
4.  在 `main.cpp` 中设置断点，通过 VS Code 的 Run 面板启动调试。

### 远程 CUDA 开发与调试 (阿里云 GPU 实例 + VS Code)

1.  按照 `CUDA远程调试环境搭建指南.md` 购买和配置阿里云 GPU 实例。
2.  在本地 VS Code 中安装 `Remote - SSH` 和 `C/C++` 扩展。
3.  使用 `Remote - SSH` 连接到阿里云实例。
4.  在远端仓库中，使用 `.vscode/tasks.json` 构建 CUDA 代码。
5.  使用 `.vscode/launch.json` 配置，通过 `cuda-gdb` 启动调试会话。

## 开发约定

*   **CUDA 代码**: 放置于 `sample/` 目录下，每个示例为一个独立的 `.cu` 文件。
*   **C++ 代码**: 放置于 `cpp-sample/` 目录下，用于演示本地开发环境配置。
*   **文档**: 中文文档用于详细说明环境搭建和使用方法。
*   **构建系统**: 顶层 `Makefile` 和 `build_and_run.sh` 脚本用于快速构建和运行 CUDA 示例。