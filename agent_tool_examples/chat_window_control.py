# -*- coding: utf-8 -*-
"""
聊天窗口大小控制
专注于展示如何在LangGraph中控制对话上下文长度
"""

import os
from dotenv import load_dotenv
from typing import List
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState, START, StateGraph, MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, add_messages
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构造上层目录的.env文件路径
env_path = os.path.join(current_dir, '..', '.env')
load_dotenv(dotenv_path=env_path)

print("=" * 80)
print("聊天窗口大小控制")
print("=" * 80)

# ===== 初始化模型 =====
print("\n初始化模型")
print("-" * 60)

try:
    # 初始化模型用于测试不同格式
    model = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek",
        temperature=0.7,
        max_tokens=200
    )
    print("✓ 模型初始化成功")
    
except Exception as e:
    print(f"✗ 模型初始化失败: {e}")
    exit(1)

# ===== 1. 手动控制聊天窗口大小 =====
print("\n\n1. 手动控制聊天窗口大小")
print("-" * 60)

try:
    # 定义窗口大小
    WINDOW_SIZE = 4  # 只保留最近4条消息 (2轮对话)
    
    # 初始化对话历史
    manual_conversation_history = [
        SystemMessage(content="你是一个专业的助手，回答要简洁明了。")
    ]
    
    print(f"  使用固定窗口大小: {WINDOW_SIZE} 条消息")
    print("  开始多轮对话示例:")
    
    # 模拟多轮对话
    for i in range(5):
        user_query = f"这是第{i+1}轮对话，我想了解如何控制聊天窗口大小。"
        
        # 添加用户消息
        manual_conversation_history.append(HumanMessage(content=user_query))
        
        # 控制窗口大小（保留系统消息和最近的WINDOW_SIZE条消息）
        if len(manual_conversation_history) > WINDOW_SIZE + 1:  # +1 是为了保留系统消息
            # 保留第一条系统消息 + 最近的WINDOW_SIZE条消息
            manual_conversation_history = [manual_conversation_history[0]] + \
                                         manual_conversation_history[-(WINDOW_SIZE):]
        
        print(f"\n  第{i+1}轮对话:")
        print(f"  用户: {user_query}")
        print(f"  当前对话历史长度: {len(manual_conversation_history)} 条消息")
        
        # 调用模型
        response = model.invoke(manual_conversation_history)
        print(f"  助手: {response.content}")
        
        # 添加助手回复到历史记录
        manual_conversation_history.append(AIMessage(content=response.content))
        
        # 再次控制窗口大小
        if len(manual_conversation_history) > WINDOW_SIZE + 1:
            manual_conversation_history = [manual_conversation_history[0]] + \
                                         manual_conversation_history[-(WINDOW_SIZE):]
    
    print("\n  手动控制聊天窗口大小的特点:")
    print("  - 简单直接，实现灵活")
    print("  - 需要自己实现截断逻辑")
    print("  - 适合简单场景")
    
except Exception as e:
    print(f"✗ 手动控制聊天窗口大小时出错: {e}")

# ===== 2. LangGraph 自定义状态类控制窗口大小 =====
print("\n\n2. LangGraph 自定义状态类控制窗口大小")
print("-" * 60)

try:
    from typing import Annotated, List, TypedDict
    
    # 定义自定义状态类，扩展MessagesState以支持窗口大小控制
    class WindowedMessagesState(MessagesState):
        message_window_size: int = 4  # 默认窗口大小为4条消息
    
    # 定义节点函数，实现窗口大小控制
    def chat_node(state: WindowedMessagesState):
        # 获取当前消息和窗口大小
        current_messages = state.messages
        window_size = state.message_window_size
        
        # 实现窗口大小控制逻辑
        if len(current_messages) > window_size:
            # 只保留最近的window_size条消息
            trimmed_messages = current_messages[-window_size:]
        else:
            trimmed_messages = current_messages
        
        # 添加系统消息
        system_msg = SystemMessage(content="你是一个专业助手，回答要简洁明了。")
        messages_for_model = [system_msg] + trimmed_messages
        
        # 调用模型
        response = model.invoke(messages_for_model)
        
        # 返回更新后的状态（注意：这里不添加到state.messages，而是由add_messages处理）
        return {"messages": [response]}
    
    # 创建状态图
    workflow = StateGraph(WindowedMessagesState)
    
    # 添加节点
    workflow.add_node("chat", chat_node)
    
    # 设置边
    workflow.add_edge(START, "chat")
    
    # 编译图，添加检查点
    app = workflow.compile(checkpointer=InMemorySaver())
    
    # 定义线程ID
    thread_id = "window_control_thread_001"
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"  使用自定义状态类控制窗口大小，默认窗口大小: 4 条消息")
    print("  开始多轮对话示例:")
    
    # 模拟多轮对话
    for i in range(5):
        user_query = f"这是第{i+1}轮对话，测试自定义状态类的窗口大小控制功能。"
        
        # 运行图
        result = app.invoke(
            {"messages": [HumanMessage(content=user_query)]},
            config=config
        )
        
        print(f"\n  第{i+1}轮对话:")
        print(f"  用户: {user_query}")
        print(f"  当前窗口中消息数量: {len(result['messages'])} 条")
        print(f"  助手: {result['messages'][-1].content}")
    
    print("\n  自定义状态类控制窗口大小的特点:")
    print("  - 结构清晰，封装性好")
    print("  - 可以在状态中灵活配置窗口大小")
    print("  - 适合需要复杂状态管理的场景")
    
except Exception as e:
    print(f"✗ LangGraph自定义状态类控制窗口大小时出错: {e}")

# ===== 3. LangGraph 配置化窗口大小控制 =====
print("\n\n3. LangGraph 配置化窗口大小控制")
print("-" * 60)

try:
    # 使用标准MessagesState
    # 定义节点函数，使用messages_add函数处理消息
    def config_chat_node(state: MessagesState):
        # 调用模型
        response = model.invoke([
            SystemMessage(content="你是一个专业助手，回答要简洁明了。")
        ] + state.messages)
        
        return {"messages": [response]}
    
    # 创建状态图
    workflow = StateGraph(MessagesState)
    
    # 添加节点
    workflow.add_node("chat", config_chat_node)
    
    # 设置边
    workflow.add_edge(START, "chat")
    
    # 编译图，添加检查点
    app = workflow.compile(checkpointer=InMemorySaver())
    
    # 定义不同窗口大小的配置
    thread_id_config = "config_window_thread_001"
    
    # 定义一个辅助函数，使用messages_add和窗口大小配置
    def send_with_window_control(user_query: str, thread_id: str, window_size: int):
        # 配置包含窗口大小
        config = {
            "configurable": {
                "thread_id": thread_id,
                "window_size": window_size
            }
        }
        
        # 使用add_messages函数处理消息，它会根据配置的窗口大小管理消息
        result = app.invoke(
            {"messages": [HumanMessage(content=user_query)]},
            config=config,
            # 使用add_messages处理消息
            patch_state={"messages": add_messages(window_size=window_size)}
        )
        
        return result
    
    print(f"  使用配置化方式控制窗口大小")
    print(f"  第一轮对话使用窗口大小: 2 条消息")
    print(f"  第二轮对话使用窗口大小: 4 条消息")
    
    # 第一轮对话，使用小窗口
    result1 = send_with_window_control(
        "你好，我想了解配置化窗口大小控制。", 
        thread_id_config, 
        window_size=2
    )
    
    print("\n  第一轮对话:")
    print(f"  用户: 你好，我想了解配置化窗口大小控制。")
    print(f"  助手: {result1['messages'][-1].content}")
    print(f"  当前窗口中消息数量: {len(result1['messages'])} 条")
    
    # 第二轮对话，增加窗口大小
    result2 = send_with_window_control(
        "能详细解释一下如何通过配置来调整窗口大小吗？", 
        thread_id_config, 
        window_size=4
    )
    
    print("\n  第二轮对话:")
    print(f"  用户: 能详细解释一下如何通过配置来调整窗口大小吗？")
    print(f"  助手: {result2['messages'][-1].content}")
    print(f"  当前窗口中消息数量: {len(result2['messages'])} 条")
    
    # 第三轮对话，验证窗口大小
    result3 = send_with_window_control(
        "如果对话轮数很多，如何避免上下文过长？", 
        thread_id_config, 
        window_size=3
    )
    
    print("\n  第三轮对话:")
    print(f"  用户: 如果对话轮数很多，如何避免上下文过长？")
    print(f"  助手: {result3['messages'][-1].content}")
    print(f"  当前窗口中消息数量: {len(result3['messages'])} 条")
    
    print("\n  配置化窗口大小控制的特点:")
    print("  - 灵活配置，无需修改代码")
    print("  - 可以为不同用户或场景设置不同窗口大小")
    print("  - 内置的messages_add函数简化了实现")
    print("  - 适合生产环境和多用户场景")
    
except Exception as e:
    print(f"✗ LangGraph配置化窗口大小控制时出错: {e}")

# ===== 聊天窗口大小控制最佳实践 =====
print("\n" + "=" * 80)
print("聊天窗口大小控制最佳实践")
print("=" * 80)
print("\n实现方法对比:")
print("1. 手动控制:")
print("   - 实现方式: 直接操作消息列表，自行实现截断逻辑")
print("   - 优点: 简单直接，无需额外依赖")
print("   - 缺点: 代码复用性差，维护困难")
print("")
print("2. 自定义状态类法:")
print("   - 实现方式: 扩展MessagesState，在状态类中添加窗口大小属性")
print("   - 优点: 结构清晰，封装性好")
print("   - 缺点: 需要定义新的状态类")
print("")
print("3. 配置化窗口法:")
print("   - 实现方式: 使用messages_add函数和configurable配置")
print("   - 优点: 灵活配置，无需修改代码")
print("   - 缺点: 需要了解LangGraph的配置机制")
print("")
print("窗口大小设置建议:")
print("- 小型对话应用: 3-5轮对话 (6-10条消息)")
print("- 中型应用: 5-10轮对话 (10-20条消息)")
print("- 复杂应用: 10-20轮对话 (20-40条消息)")
print("")
print("高级策略:")
print("1. 动态调整: 根据对话内容重要性和token数量动态调整窗口大小")
print("2. 优先级保留: 重要信息（如系统提示、关键指令）始终保留在窗口中")
print("3. 摘要压缩: 对旧消息进行摘要压缩，保留关键信息")
print("4. 分层管理: 不同类型的信息使用不同的窗口策略")
print("")
print("注意事项:")
print("- 窗口太小会导致上下文丢失，影响对话连贯性")
print("- 窗口太大会增加token消耗和响应时间")
print("- 需要根据具体模型的上下文窗口限制调整策略")
print("- 对于长时间运行的对话，建议结合摘要技术使用")
print("=" * 80)