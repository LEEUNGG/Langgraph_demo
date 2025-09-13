# LangChain 和 LangGraph 示例集

这个目录包含了一系列专注于 LangChain 和 LangGraph 核心功能的示例文件，帮助你学习和掌握这些强大的LLM应用开发框架。每个文件都聚焦于特定的功能点，便于理解和实践。

## 文件列表及功能说明

### 1. `langchain_model_calls.py`
**专注于 LangChain 的模型调用方法**
- 直接导入特定提供商模型类的方法
- 使用 `init_chat_model` 自动识别模型的方法
- 明确指定提供商的 `init_chat_model` 调用方法
- 各种调用方式的参数配置和使用场景

### 2. `chat_formats.py`
**专注于不同的聊天记录格式**
- LangChain Message 对象格式（SystemMessage、HumanMessage、AIMessage）
- OpenAI 风格格式（字典列表）
- 高级封装格式（携带额外元数据和上下文信息）
- 各种格式的优缺点和适用场景对比

### 3. `thread_management.py`
**专注于对话历史管理和线程管理的对比**
- 手动维护对话历史的方法和示例
- 使用 LangGraph Checkpointer 进行线程管理的方法
- 多用户、多会话场景下的会话隔离实现
- 自定义 LangGraph 状态管理
- 手动维护与线程管理的详细对比

### 4. `chat_window_control.py`
**专注于聊天窗口大小控制**
- 手动控制聊天窗口大小的实现
- 使用自定义状态类控制窗口大小
- 通过配置化方式动态调整窗口大小
- 不同窗口大小控制方法的优缺点对比
- 窗口大小设置的最佳实践建议

### 5. `advanced_format_and_langgraph.py`
**专注于高级封装格式和 LangGraph 高级用法**
- 高级封装格式的完整应用示例
- LangGraph 复杂工作流的构建和使用
- 条件分支和工具调用的集成
- 高级封装格式与 LangGraph 的结合应用
- 企业级应用的最佳实践

## 如何使用这些示例

1. **前提条件**
   - 安装 Python 3.8+ 和 pip
   - 安装所需依赖：`pip install langchain langgraph python-dotenv`
   - 准备 `.env` 文件，放在项目根目录，包含 API 密钥等配置

2. **运行示例**
   ```bash
   # 运行特定的示例文件
   python agent_tool_examples/文件名.py
   ```

3. **学习建议**
   - 从简单的示例开始：先了解模型调用和基本的聊天格式
   - 逐步深入：学习对话历史管理和窗口控制
   - 高级应用：探索 LangGraph 的复杂工作流和高级封装格式
   - 对比学习：比较不同实现方式的优缺点，选择适合自己项目的方案

## 学习路径推荐

1. **基础阶段**
   - `langchain_model_calls.py` - 掌握模型调用的基础知识
   - `chat_formats.py` - 了解不同的消息格式

2. **进阶阶段**
   - `thread_management.py` - 学习对话历史管理
   - `chat_window_control.py` - 掌握窗口大小控制

3. **高级阶段**
   - `advanced_format_and_langgraph.py` - 探索复杂应用场景

## 最佳实践总结

- 在 LangChain 环境中，优先使用 LangChain Message 对象
- 需要跨平台兼容性时，使用 OpenAI 风格格式
- 小型项目或快速验证概念时，使用手动维护对话历史
- 生产环境、多用户应用、需要复杂状态管理时，使用 LangGraph 线程管理
- 窗口大小设置建议：简单应用 3-5 轮，中等应用 5-10 轮，复杂应用 10-20 轮
- 结合摘要技术和窗口控制，优化长时间运行的对话

## 注意事项
- 示例中的 API 调用需要有效的 API 密钥
- 某些示例可能需要安装额外的依赖
- 生产环境中请合理管理和保护用户数据和会话信息
- 根据实际项目需求调整示例中的参数和配置