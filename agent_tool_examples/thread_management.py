# -*- coding: utf-8 -*-
"""
对话历史管理方式对比
专注于展示手动维护对话历史与使用LangGraph Checkpointer线程管理的区别
"""

import os
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构造上层目录的.env文件路径
env_path = os.path.join(current_dir, '..', '.env')
load_dotenv(dotenv_path=env_path)

print("=" * 80)
print("对话历史管理方式对比")
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

# ===== 1. 手动维护对话历史 =====
print("\n\n1. 手动维护对话历史")
print("-" * 60)

try:
    # 创建一个列表来存储对话历史
    manual_conversation_history = [
        SystemMessage(content="你是一个专业的助手，回答要简洁明了。")
    ]
    
    print("  开始多轮对话示例:")
    
    # 第一轮对话
    user_query_1 = "你好，我想了解一下机器学习。"
    manual_conversation_history.append(HumanMessage(content=user_query_1))
    
    print(f"  用户: {user_query_1}")
    response_1 = model.invoke(manual_conversation_history)
    print(f"  助手: {response_1.content}")
    
    # 将助手的回复添加到对话历史
    manual_conversation_history.append(AIMessage(content=response_1.content))
    
    # 第二轮对话，测试上下文理解
    user_query_2 = "它和深度学习有什么关系？"
    manual_conversation_history.append(HumanMessage(content=user_query_2))
    
    print(f"  用户: {user_query_2}")
    response_2 = model.invoke(manual_conversation_history)
    print(f"  助手: {response_2.content}")
    
    # 将助手的回复添加到对话历史
    manual_conversation_history.append(AIMessage(content=response_2.content))
    
    # 检查对话历史长度
    print(f"\n  当前对话历史长度: {len(manual_conversation_history)} 条消息")
    print(f"  前3条消息示例:")
    for i, msg in enumerate(manual_conversation_history[:3]):
        msg_type = type(msg).__name__
        print(f"  {i+1}. [{msg_type}] {msg.content[:30]}...")
    
    print("\n  手动维护对话历史的特点:")
    print("  - 简单直接，适合小型脚本和简单应用")
    print("  - 开发者需要手动管理消息的添加和移除")
    print("  - 多用户场景下需要自己实现会话隔离")
    print("  - 适合快速原型开发")
    
except Exception as e:
    print(f"✗ 手动维护对话历史时出错: {e}")

# ===== 2. LangGraph 线程管理 (Checkpointer) =====
print("\n\n2. LangGraph 线程管理 (Checkpointer)")
print("-" * 60)

try:
    # 定义一个简单的工具
    @tool
    def search_tool(query: str) -> str:
        """搜索工具，用于获取关于特定主题的最新信息"""
        return f"关于'{query}'的搜索结果: 这是一个示例搜索结果，实际应用中会调用真实的搜索引擎。"
    
    tools = [search_tool]
    
    # 创建一个检查点存储，用于管理对话历史
    checkpointer = InMemorySaver()
    
    # 创建一个反应式智能体
    agent = create_react_agent(model, tools, checkpointer=checkpointer)
    
    # 定义两个不同的线程ID，模拟不同用户或不同会话
    thread_id_1 = "user_123_session_001"
    thread_id_2 = "user_456_session_002"
    
    # 为线程1设置配置
    config_1 = {"configurable": {"thread_id": thread_id_1}}
    config_2 = {"configurable": {"thread_id": thread_id_2}}
    
    print(f"  使用线程1 ({thread_id_1}):")
    
    # 线程1的第一轮对话
    response_agent_1 = agent.invoke(
        {"messages": [HumanMessage(content="你好，我是用户1，能介绍一下LangGraph吗？")]},
        config=config_1
    )
    
    print(f"  用户1: 你好，我是用户1，能介绍一下LangGraph吗？")
    print(f"  助手: {response_agent_1['messages'][-1].content}")
    
    # 线程1的第二轮对话
    response_agent_1b = agent.invoke(
        {"messages": [HumanMessage(content="它和普通的LangChain有什么区别？")]},
        config=config_1
    )
    
    print(f"  用户1: 它和普通的LangChain有什么区别？")
    print(f"  助手: {response_agent_1b['messages'][-1].content}")
    
    print(f"\n  使用线程2 ({thread_id_2}):")
    
    # 线程2的第一轮对话（独立上下文）
    response_agent_2 = agent.invoke(
        {"messages": [HumanMessage(content="你好，我是用户2，我想了解Python编程。")]},
        config=config_2
    )
    
    print(f"  用户2: 你好，我是用户2，我想了解Python编程。")
    print(f"  助手: {response_agent_2['messages'][-1].content}")
    
    # 回到线程1，验证上下文是否独立
    print(f"\n  回到线程1 ({thread_id_1}):")
    response_agent_1c = agent.invoke(
        {"messages": [HumanMessage(content="能举个简单的例子说明LangGraph的应用场景吗？")]},
        config=config_1
    )
    
    print(f"  用户1: 能举个简单的例子说明LangGraph的应用场景吗？")
    print(f"  助手: {response_agent_1c['messages'][-1].content}")
    
    print("\n  LangGraph线程管理的特点:")
    print("  - 自动管理对话历史，无需手动添加和移除消息")
    print("  - 通过thread_id实现多用户/多会话隔离")
    print("  - 支持会话的持久化和恢复")
    print("  - 集成了工具调用和状态管理")
    print("  - 适合复杂的多用户应用场景")
    
except Exception as e:
    print(f"✗ LangGraph线程管理时出错: {e}")

# ===== 3. 自定义LangGraph状态管理 =====
print("\n\n3. 自定义LangGraph状态管理")
print("-" * 60)

try:
    # 定义一个简单的状态类
    class CustomState(MessagesState):
        session_id: str = ""
        user_name: str = ""
        
    # 定义一个节点函数
    def chat_node(state: CustomState):
        # 可以访问和修改状态中的各种属性
        user_message = state.messages[-1].content if state.messages else ""
        
        # 构造带有个性化问候的系统消息
        system_msg = SystemMessage(
            content=f"你是一个专业助手，用户名叫{state.user_name}，会话ID是{state.session_id}。"
        )
        
        # 创建消息列表，包含系统消息和用户消息
        messages = [system_msg] + state.messages
        
        # 调用模型
        response = model.invoke(messages)
        
        # 返回更新后的状态
        return {"messages": [response]}
    
    # 创建状态图
    workflow = StateGraph(CustomState)
    
    # 添加节点
    workflow.add_node("chat", chat_node)
    
    # 设置边
    workflow.add_edge(START, "chat")
    
    # 编译图，添加检查点
    app = workflow.compile(checkpointer=InMemorySaver())
    
    # 设置自定义配置
    custom_config = {
        "configurable": {
            "thread_id": "custom_thread_001",
            "session_id": "custom_session_123",
            "user_name": "小明"
        }
    }
    
    # 运行图
    print(f"  使用自定义状态和线程ID: custom_thread_001")
    response_custom = app.invoke(
        {"messages": [HumanMessage(content="你好，你知道我是谁吗？")]},
        config=custom_config
    )
    
    print(f"  用户: 你好，你知道我是谁吗？")
    print(f"  助手: {response_custom['messages'][-1].content}")
    
    print("\n  自定义LangGraph状态管理的特点:")
    print("  - 可以定义个性化的状态结构")
    print("  - 支持更复杂的业务逻辑集成")
    print("  - 结合了线程管理和自定义状态的优势")
    print("  - 适合企业级应用开发")
    
except Exception as e:
    print(f"✗ 自定义LangGraph状态管理时出错: {e}")

# ===== 手动维护与线程管理的对比总结 =====
print("\n" + "=" * 80)
print("手动维护对话历史 vs LangGraph线程管理")
print("=" * 80)
print("\n对比维度:")
print("1. 历史记录管理方式:")
print("   - 手动维护: 开发者直接操作消息列表，添加/删除消息")
print("   - 线程管理: Checkpointer自动管理，通过thread_id访问")
print("")
print("2. 多会话处理:")
print("   - 手动维护: 需要自行实现会话隔离逻辑")
print("   - 线程管理: 内置多会话支持，通过thread_id区分")
print("")
print("3. 调用方式:")
print("   - 手动维护: 直接调用模型接口，传入消息列表")
print("   - 线程管理: 通过agent.invoke()调用，传入配置")
print("")
print("4. 扩展性:")
print("   - 手动维护: 扩展功能需要自己实现")
print("   - 线程管理: 集成了工具调用、状态管理等高级功能")
print("")
print("5. 持久化:")
print("   - 手动维护: 需要自行实现持久化逻辑")
print("   - 线程管理: 支持多种持久化后端")
print("")
print("6. 适用场景:")
print("   - 手动维护: 简单脚本、快速原型开发")
print("   - 线程管理: 复杂应用、多用户系统、企业级产品")
print("")
print("最佳实践建议:")
print("- 小型项目或快速验证概念时，使用手动维护方式")
print("- 生产环境、多用户应用、需要复杂状态管理时，使用LangGraph线程管理")
print("- 可根据项目规模和复杂度逐步从手动维护过渡到线程管理")
print("=" * 80)