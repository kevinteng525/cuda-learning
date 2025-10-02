# VS Code 本地 C++ 调试配置指南 (macOS)

本指南帮助你在 macOS 上把刚装好的 Visual Studio Code 配置为可以编辑、构建并调试本地 C++ 程序。文档包含扩展安装、编译器准备、VS Code 的 tasks/launch 示例，以及如何运行仓库中提供的示例。

适用场景：本地 macOS（使用 clang/lldb）。如果你要在远端使用 Remote-SSH，请参考仓库中的 `CUDA远程调试环境搭建指南.md`（已包含远端 CUDA/VS Code 指南）。

---

## 1. 必要组件

- macOS（已安装 Xcode 命令行工具）
  - 安装：
    ```zsh
    xcode-select --install
    ```
- VS Code（已安装）
- 推荐扩展：
  - C/C++ (ms-vscode.cpptools)
  - CodeLLDB (vadimcn.vscode-lldb) — 推荐用于 LLDB 的原生支持
  - CMake Tools (ms-vscode.cmake-tools) — 如果你使用 CMake

安装顺序建议：先安装 C/C++，然后 CodeLLDB。

## 2. 检查编译器与调试器

在终端运行：
```zsh
clang --version
which lldb
```
如果 lldb 不存在，请安装 Xcode 命令行工具。

## 3. 仓库中已包含的示例

我在仓库中添加了一个 C++ 示例目录 `cpp-sample/`，包含：
- `cpp-sample/main.cpp` —— 示例源文件
- `cpp-sample/Makefile` —— 构建规则
- `cpp-sample/.vscode/tasks.json` —— VS Code 的构建任务
- `cpp-sample/.vscode/launch.json` —— 用 CodeLLDB 的调试配置

示例目录结构：
```
cpp-sample/
├─ .vscode/
│  ├─ tasks.json
│  └─ launch.json
├─ Makefile
└─ main.cpp
```

## 4. 在 VS Code 中打开并运行示例

1. 打开 VS Code，选择菜单 `File -> Open...` 并打开仓库根目录（或直接打开 `cpp-sample` 文件夹）。
2. 确保已安装扩展：C/C++ 和 CodeLLDB。
3. 在 VS Code 中打开 `cpp-sample/main.cpp`。

构建（两种方式）：
- 使用 Makefile（推荐）:
  - 在终端运行：
    ```zsh
    cd cpp-sample
    make
    ```
  - 或在 VS Code 终端运行同样命令。
- 使用 VS Code Task：按 F1 -> Tasks: Run Build Task -> 选择 `build: cpp-sample`。

运行：
```zsh
./cpp-sample/main
```

## 5. 在 VS Code 中调试

1. 在 `main.cpp` 中添加断点（点击左侧行号）。
2. 打开左侧 Run 面板，确保选中配置 `LLDB: Launch`（位于 `cpp-sample/.vscode/launch.json` 中）。
3. 点击绿色播放按钮或按 F5 启动调试。

你应能在断点处暂停，查看变量、调用栈和控制单步执行。

## 6. 文件内容摘要（便于参考）

- `cpp-sample/main.cpp`：演示函数调用和向量，适合设置多个断点。  
- `cpp-sample/Makefile`：提供 `make`、`make run`、`make clean`。
- `cpp-sample/.vscode/tasks.json`：自动化 clang++ 构建任务。  
- `cpp-sample/.vscode/launch.json`：CodeLLDB 的 launch 配置。

## 7. 常见问题排查

- 如果 F5 无法启动或提示 `The debug type is not recognized`：确认已安装 CodeLLDB 或 C/C++ 扩展，并重启 VS Code。  
- 如果编译失败提示找不到头文件：检查 Include 路径或在 `tasks.json`/Makefile 中添加需要的 `-I`。  
- 如果需要使用 Microsoft 的 `cppdbg`（ms-vscode.cpptools）：可以把 `launch.json` 改为 `type: "cppdbg"` 并设置 `MIMode: "lldb"`，但此时确保已安装 C/C++ 扩展。

---

如果你愿意，我可以把本说明追加到仓库根的 `README.md` 中，或把 `cpp-sample` 的构建/调试演示录制成更详细的步骤（含截图或命令输出示例）。
