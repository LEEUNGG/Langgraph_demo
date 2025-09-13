#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek 模型调用测试示例
展示三种不同的模型调用方法，并实际向模型提问
同时演示不同的聊天记录格式
"""

import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构造上层目录的.env文件路径
env_path = os.path.join(current_dir, '..', '.env')
load_dotenv(dotenv_path=env_path)

print("=" * 80)
print("DeepSeek 模型调用测试示例")
print("=" * 80)
print("展示三种不同模型调用方法和多种聊天记录格式")
print("=" * 80)

# ===== 方法1: 直接导入特定提供商的模型类 =====
print("\n1. 直接导入特定提供商的模型类")
print("-" * 60)

question = "hi"
# 准备更复杂的对话历史示例
conversation_history = [
    {"role": "user", "content": "你好！"},
    {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"},
    {"role": "user", "content": "请简单介绍一下你自己。"}
]

# 方法1: 直接导入特定提供商的模型类
try:
    # DeepSeek 模型
    from langchain_deepseek.chat_models import ChatDeepSeek
    from langchain.schema import HumanMessage, AIMessage, SystemMessage
    
    deepseek_model_direct = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0.7,
        max_tokens=100,
        api_key=os.getenv("DEEPSEEK_API_KEY")
    )
    print("✓ DeepSeek 模型 (直接导入):")
    print(f"  类型: {type(deepseek_model_direct)}")
    print(f"  模型名: {deepseek_model_direct.model_name}")
    print(f"  温度: {deepseek_model_direct.temperature}")
    
    # 格式1: 使用 LangChain 的 Message 对象（HumanMessage）
    print(f"\n  格式1: 使用 HumanMessage 单轮提问")
    print(f"  提问: {question}")
    response = deepseek_model_direct.invoke([HumanMessage(content=question)])
    print(f"  响应: {response.content}")
    
    # 格式2: 使用 LangChain 的 Message 对象链（包含对话历史）
    print(f"\n  格式2: 使用 LangChain Message 对象链（包含对话历史）")
    langchain_messages = []
    for msg in conversation_history:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=msg["content"]))
    
    # 添加最新问题
    new_question = "你刚才说的内容能详细一点吗？"
    langchain_messages.append(HumanMessage(content=new_question))
    
    print(f"  对话历史长度: {len(langchain_messages) - 1}")
    print(f"  最新提问: {new_question}")
    response = deepseek_model_direct.invoke(langchain_messages)
    print(f"  响应: {response.content}")
    
    # 格式3: 使用 OpenAI 风格的消息格式
    print(f"\n  格式3: 使用 OpenAI 风格的消息格式")
    # 直接传入OpenAI格式的消息列表
    openai_style_messages = conversation_history.copy()
    openai_style_messages.append({"role": "user", "content": new_question})
    
    print(f"  对话历史长度: {len(openai_style_messages) - 1}")
    print(f"  最新提问: {new_question}")
    response = deepseek_model_direct.invoke(openai_style_messages)
    print(f"  响应: {response.content}")
    
except ImportError as e:
    print(f"✗ DeepSeek 模型导入失败: {e}")
except Exception as e:
    print(f"✗ 调用模型时出错: {e}")

# ===== 方法2: 使用 init_chat_model 统一初始化 =====
print("\n\n2. 使用 init_chat_model 统一初始化")
print("-" * 60)

try:
    from langchain.chat_models import init_chat_model
    from langchain.schema import HumanMessage, AIMessage
    
    # 方式2a: 通过模型名自动识别提供商
    print("方式2a: 自动识别提供商")
    deepseek_model_init_auto = init_chat_model(
        model="deepseek-chat",
        temperature=0.7,
        max_tokens=100,
        api_key=os.getenv("DEEPSEEK_API_KEY")
    )
    print("✓ DeepSeek 模型 (自动识别):")
    print(f"  类型: {type(deepseek_model_init_auto)}")
    print(f"  模型名: {deepseek_model_init_auto.model_name if hasattr(deepseek_model_init_auto, 'model_name') else getattr(deepseek_model_init_auto, 'model', '未知')}")
    
    # 格式1: 使用 HumanMessage 单轮提问
    print(f"\n  格式1: 使用 HumanMessage 单轮提问")
    print(f"  提问: {question}")
    response = deepseek_model_init_auto.invoke([HumanMessage(content=question)])
    print(f"  响应: {response.content}")
    
    # 格式2: 带系统提示的多轮对话
    print(f"\n  格式2: 带系统提示的多轮对话")
    # 准备带系统提示的多轮对话
    system_prompt = "你是一个专业的助手，回答问题要简洁明了。"
    system_message = {"role": "system", "content": system_prompt}
    
    multi_turn_messages = [
        system_message,
        {"role": "user", "content": "什么是人工智能？"},
        {"role": "assistant", "content": "人工智能是模拟人类智能的技术。"},
        {"role": "user", "content": "能举个例子吗？"}
    ]
    
    print(f"  系统提示: {system_prompt}")
    print(f"  对话轮数: {len(multi_turn_messages) - 1}")
    response = deepseek_model_init_auto.invoke(multi_turn_messages)
    print(f"  响应: {response.content}")
    
    # 方式2b: 明确指定提供商
    print("\n方式2b: 明确指定提供商")
    deepseek_model_init_explicit = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek",
        temperature=0.5,
        max_tokens=100
        # 测试不直接传入api_key，让它自动从环境变量读取
        # api_key=os.getenv("DEEPSEEK_API_KEY")
    )
    print("✓ DeepSeek 模型 (指定提供商):")
    print(f"  类型: {type(deepseek_model_init_explicit)}")
    print(f"  模型名: {deepseek_model_init_explicit.model_name if hasattr(deepseek_model_init_explicit, 'model_name') else getattr(deepseek_model_init_explicit, 'model', '未知')}")
    
    # 格式1: 使用 invoke 方法直接传入消息列表
    print(f"\n  格式1: 使用 HumanMessage 单轮提问")
    print(f"  提问: {question}")
    response = deepseek_model_init_explicit.invoke([HumanMessage(content=question)])
    print(f"  响应: {response.content}")
    
    # 格式2: 使用高级封装的消息格式
    print(f"\n  格式2: 使用高级封装的消息格式")
    print(f"  这种格式在复杂框架中非常有用，可以同时携带多种上下文信息")
    
    # 示例1: 携带会话ID和元数据的高级封装
    print(f"  \n  示例1: 携带会话ID和元数据")
    advanced_message_with_context = {
        "messages": [
            {"role": "system", "content": "你是一个专业的AI助手。"},
            {"role": "user", "content": "解释什么是机器学习？"}
        ],
        # 额外参数1: 会话元数据
        "session_id": "user_123_session_456",
        "user_id": "user_123",
        # 额外参数2: 模型配置参数
        "model_config": {
            "temperature": 0.7,
            "max_tokens": 200,
            "top_p": 0.9
        },
        # 额外参数3: 上下文信息
        "context": {
            "user_preferences": {"language": "中文", "detail_level": "中等"},
            "chat_history_id": "history_789"
        }
    }
    
    print(f"  高级封装格式包含的额外参数:")
    print(f"  - 会话ID: {advanced_message_with_context['session_id']}")
    print(f"  - 用户ID: {advanced_message_with_context['user_id']}")
    print(f"  - 模型配置: 温度={advanced_message_with_context['model_config']['temperature']}, ")
    print(f"              最大tokens={advanced_message_with_context['model_config']['max_tokens']}")
    print(f"  - 用户偏好: {advanced_message_with_context['context']['user_preferences']}")
    print(f"  - 系统提示: {advanced_message_with_context['messages'][0]['content']}")
    print(f"  - 提问: {advanced_message_with_context['messages'][1]['content']}")
    
    # 在真实框架中，这些额外参数会被框架处理
    # 这里我们只提取messages部分进行模型调用
    response = deepseek_model_init_explicit.invoke(advanced_message_with_context['messages'])
    print(f"  响应: {response.content}")
    
    # 示例2: LangGraph风格的高级封装
    print(f"  \n  示例2: LangGraph风格的高级封装")
    langgraph_style_message = {
        "messages": [
            {"role": "system", "content": "你是一个助手，回答要简洁明了。"},
            {"role": "user", "content": "什么是Python？"}
        ],
        # LangGraph风格的额外参数
        "agent_state": {
            "step": "response_generation",
            "thought_process": "用户询问关于Python的定义，需要给出简洁解释",
            "available_tools": ["search", "code_interpreter"]
        },
        "chat_metadata": {
            "timestamp": "2023-10-01T12:00:00Z",
            "source": "web_interface",
            "client_info": {"browser": "Chrome", "version": "118"}
        }
    }
    
    print(f"  LangGraph风格封装的特点:")
    print(f"  - 包含智能体状态: {langgraph_style_message['agent_state']['step']}")
    print(f"  - 思考过程: {langgraph_style_message['agent_state']['thought_process']}")
    print(f"  - 可用工具: {', '.join(langgraph_style_message['agent_state']['available_tools'])}")
    print(f"  - 提问: {langgraph_style_message['messages'][1]['content']}")
    
    # 调用模型
    response = deepseek_model_init_explicit.invoke(langgraph_style_message['messages'])
    print(f"  响应: {response.content}")
    
except ImportError as e:
    print(f"✗ init_chat_model 导入失败: {e}")
except Exception as e:
    print(f"✗ 初始化或调用模型时出错: {e}")

# ===== 不同聊天格式总结 =====
print("\n" + "=" * 80)
print("不同聊天记录格式总结")
print("=" * 80)
print("1. LangChain Message 对象格式:")
print("   - HumanMessage: 用户消息")
print("   - AIMessage: 助手回复")
print("   - SystemMessage: 系统提示")
print("   - 优点: 类型明确，便于代码处理和类型检查")
print("   - 适用: LangChain生态系统内的开发")
print("")
print("2. OpenAI 风格格式 (字典列表):")
print("   - 格式: [{\"role\": \"user/assistant/system\", \"content\": \"消息内容\"}]")
print("   - 优点: 通用性强，兼容多种模型API")
print("   - 适用: 跨平台、跨模型的开发场景")
print("")
print("3. 高级封装格式:")
print("   - 格式: {\"messages\": [OpenAI风格消息列表], \"session_id\": \"xxx\", \"metadata\": {...}, ...}")
print("   - 优点: 可以携带丰富的上下文信息、会话元数据、模型配置等额外参数")
print("   - 特点: 结构化设计，便于在复杂框架中传递多种信息")
print("   - 典型应用:")
print("     - 会话管理: 追踪session_id、user_id")
print("     - 上下文增强: 传递用户偏好、历史记录引用")
print("     - 动态配置: 运行时调整模型参数")
print("     - 框架集成: 与LangGraph等高级框架无缝集成")
print("   - 适用: LangGraph等高级框架中的智能体开发，企业级多用户系统")
print("")
print("最佳实践:")
print("- 在LangChain环境中，优先使用LangChain Message对象")
print("- 需要跨平台兼容性时，使用OpenAI风格格式")
print("- 在复杂框架中，使用高级封装格式")
print("=" * 80)
print("模型调用测试完成！")
print("=" * 80)


# ===== 对话历史管理方式对比 =====
print("\n" + "=" * 80)
print("对话历史管理方式对比")
print("=" * 80)

print("\n1. 手动维护对话历史")
print("-" * 60)

# 手动维护对话历史示例
try:
    # 初始化模型
    model_manual = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek",
        temperature=0.7
    )
    
    # 手动创建和维护对话历史列表
    manual_conversation_history = []
    
    # 第一轮对话
    print("  第一轮对话:")
    user_query_1 = "你好，我叫陈明，好久不见！"
    manual_conversation_history.append({"role": "user", "content": user_query_1})
    response_1 = model_manual.invoke(manual_conversation_history)
    manual_conversation_history.append({"role": "assistant", "content": response_1.content})
    
    print(f"    用户: {user_query_1}")
    print(f"    助手: {response_1.content}")
    
    # 第二轮对话 - 需要手动将所有历史消息传递给模型
    print("\n  第二轮对话:")
    user_query_2 = "还记得我刚才告诉你的名字吗？"
    manual_conversation_history.append({"role": "user", "content": user_query_2})
    response_2 = model_manual.invoke(manual_conversation_history)
    manual_conversation_history.append({"role": "assistant", "content": response_2.content})
    
    print(f"    用户: {user_query_2}")
    print(f"    助手: {response_2.content}")
    
    print("\n  手动维护的特点:")
    print("  - 开发者需要手动创建和管理对话历史列表")
    print("  - 每次调用模型时必须传递完整的历史记录")
    print(f"  - 当前历史记录长度: {len(manual_conversation_history)}")
    
    # 模拟历史记录管理的复杂性
    print("\n  潜在挑战:")
    print("  - 需要手动处理消息的添加、更新和清理")
    print("  - 在多用户环境中需要额外的数据结构区分不同用户")
    print("  - 没有内置的会话状态恢复机制")
    
except Exception as e:
    print(f"✗ 手动维护对话历史时出错: {e}")

print("\n\n2. 使用 LangGraph 的 Checkpointer 线程管理")
print("-" * 60)

try:
    # 定义一个简单的工具函数作为示例
    def mock_get_weather(location: str) -> str:
        """模拟获取天气信息的工具函数"""
        return f"{location}的天气晴朗，气温25°C"
    
    # 初始化模型
    model_threaded = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek",
        temperature=0.7
    )
    
    # 创建内存中的检查点存储（线程管理）
    checkpointer = InMemorySaver()
    
    # 定义工具列表
    tools = [mock_get_weather]
    
    # 创建支持线程管理的智能体
    agent = create_react_agent(
        model=model_threaded,
        tools=tools,
        checkpointer=checkpointer
    )
    
    # 定义第一个线程的配置
    thread_config_1 = {
        "configurable": {
            "thread_id": "user_chenming_thread_1"
        }
    }
    
    # 第一轮对话 - 使用线程管理
    print("  第一轮对话 (线程1):")
    user_query_thread_1 = "你好，我叫陈明，好久不见！"
    response_thread_1 = agent.invoke(
        {"messages": [{"role": "user", "content": user_query_thread_1}]},
        thread_config_1
    )
    
    print(f"    用户: {user_query_thread_1}")
    print(f"    助手: {response_thread_1['messages'][-1].content}")
    
    # 第二轮对话 - 不需要手动传递历史，由checkpointer管理
    print("\n  第二轮对话 (线程1):")
    user_query_thread_2 = "还记得我刚才告诉你的名字吗？"
    response_thread_2 = agent.invoke(
        {"messages": [{"role": "user", "content": user_query_thread_2}]},
        thread_config_1
    )
    
    print(f"    用户: {user_query_thread_2}")
    print(f"    助手: {response_thread_2['messages'][-1].content}")
    
    # 创建第二个线程，模拟多用户场景
    print("\n  同时处理第二个线程 (用户李强):")
    thread_config_2 = {
        "configurable": {
            "thread_id": "user_liqiang_thread_2"
        }
    }
    
    user_query_thread_3 = "你好，我是李强，初次见面！"
    response_thread_3 = agent.invoke(
        {"messages": [{"role": "user", "content": user_query_thread_3}]},
        thread_config_2
    )
    
    print(f"    用户: {user_query_thread_3}")
    print(f"    助手: {response_thread_3['messages'][-1].content}")
    
    # 返回第一个线程继续对话
    print("\n  返回第一个线程继续对话:")
    user_query_thread_4 = "你能帮我查一下北京的天气吗？"
    response_thread_4 = agent.invoke(
        {"messages": [{"role": "user", "content": user_query_thread_4}]},
        thread_config_1
    )
    
    print(f"    用户: {user_query_thread_4}")
    print(f"    助手: {response_thread_4['messages'][-1].content}")
    
    print("\n  LangGraph 线程管理的特点:")
    print("  - 通过 thread_id 自动隔离和管理不同会话")
    print("  - checkpointer 自动保存和恢复会话状态")
    print("  - 无需手动传递完整对话历史")
    print("  - 支持多用户、多会话场景")
    print("  - 集成了工具使用能力")
    
except Exception as e:
    print(f"✗ 使用线程管理时出错: {e}")

# ===== Checkpointer线程管理中控制聊天窗口大小 =====
print("\n" + "=" * 80)
print("在Checkpointer线程管理中定义聊天窗口大小")
print("=" * 80)

print("\n1. 通过自定义状态类控制聊天窗口大小")
print("-" * 60)

try:
    from typing import List, Dict, Any
    from langgraph.graph import StateGraph, State, MessagesState
    from langgraph.graph.message import add_messages
    
    # 定义一个自定义状态类，包含消息窗口大小限制
    class WindowedMessagesState(MessagesState):
        # 可选的消息窗口大小字段
        message_window_size: int = 10  # 默认窗口大小为10条消息
    
    # 创建一个简单的图来演示窗口控制
    def create_windowed_chat_graph(model, window_size=5):
        """创建一个带消息窗口控制的简单聊天图"""
        
        # 定义图的构建函数
        def chat_node(state: WindowedMessagesState) -> Dict[str, Any]:
            # 调用模型获取响应
            response = model.invoke(state["messages"])
            
            # 计算新的消息列表
            new_messages = state["messages"] + [response]
            
            # 应用窗口大小限制
            if len(new_messages) > state["message_window_size"]:
                # 只保留最新的message_window_size条消息
                # 注意：通常我们希望保留系统提示，所以这里从索引1开始
                if new_messages[0].role == "system":
                    # 保留第一条系统消息，其余按窗口大小截断
                    new_messages = [new_messages[0]] + new_messages[-(state["message_window_size"]-1):]
                else:
                    new_messages = new_messages[-state["message_window_size"]:]
            
            return {"messages": new_messages}
        
        # 创建状态图
        graph_builder = StateGraph(WindowedMessagesState)
        graph_builder.add_node("chat", chat_node)
        graph_builder.set_entry_point("chat")
        graph_builder.set_finish_point("chat")
        
        # 创建图实例
        return graph_builder.compile(checkpointer=InMemorySaver())
    
    # 初始化模型
    model_windowed = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek",
        temperature=0.7
    )
    
    # 创建带窗口控制的聊天图，设置窗口大小为4（小窗口以便快速看到效果）
    windowed_chat = create_windowed_chat_graph(model_windowed, window_size=4)
    
    # 定义线程配置
    windowed_thread_config = {
        "configurable": {
            "thread_id": "windowed_chat_thread"
        }
    }
    
    # 发送第一条消息（包含系统提示）
    print("  第一条消息 (包含系统提示):")
    initial_messages = [
        {"role": "system", "content": "你是一个助手，回答要简洁明了。"},
        {"role": "user", "content": "你好，我叫陈明，好久不见！"}
    ]
    
    response_window_1 = windowed_chat.invoke(
        {"messages": initial_messages, "message_window_size": 4},
        windowed_thread_config
    )
    
    print(f"    用户: {initial_messages[1]['content']}")
    print(f"    助手: {response_window_1['messages'][-1].content}")
    print(f"    当前消息数量: {len(response_window_1['messages'])}")
    
    # 继续发送消息，直到超出窗口大小
    print("\n  连续发送多条消息，测试窗口控制:")
    queries = [
        "今天天气怎么样？",
        "能给我讲个笑话吗？",
        "帮我计算一下1+1等于多少？",
        "你能记住我刚才问了什么问题吗？"
    ]
    
    for i, query in enumerate(queries):
        print(f"  消息 {i+2}:")
        print(f"    用户: {query}")
        
        # 只发送新消息，Checkpointer会自动维护历史
        response = windowed_chat.invoke(
            {"messages": [{"role": "user", "content": query}]},
            windowed_thread_config
        )
        
        print(f"    助手: {response['messages'][-1].content}")
        print(f"    当前消息数量: {len(response['messages'])}")
        print(f"    消息内容预览: {[msg.role for msg in response['messages']]}")
        print()
    
    print("\n  窗口控制特点:")
    print("  - 通过自定义状态类可以设置最大消息数量")
    print("  - 当消息数量超过窗口大小时，自动截断最早的消息")
    print("  - 可以选择保留系统提示消息")
    print("  - 适用于控制上下文长度，防止token超出限制")
    
except Exception as e:
    print(f"✗ 控制聊天窗口大小时出错: {e}")

print("\n\n2. 使用内置的消息处理函数")
print("-" * 60)

try:
    from langgraph.graph.message import add_messages, messages_add
    
    # 定义一个简单的节点，使用内置函数处理消息窗口
    def windowed_agent_node(state: MessagesState, config: Dict[str, Any]) -> Dict[str, Any]:
        # 获取配置的窗口大小
        window_size = config.get("configurable", {}).get("window_size", 5)
        
        # 调用模型获取响应
        response = model_windowed.invoke(state["messages"])
        
        # 使用内置函数添加消息
        updated_messages = messages_add(state["messages"], [response])
        
        # 应用窗口大小限制
        if len(updated_messages) > window_size:
            # 保留最新的window_size条消息
            if updated_messages[0].role == "system":
                updated_messages = [updated_messages[0]] + updated_messages[-(window_size-1):]
            else:
                updated_messages = updated_messages[-window_size:]
        
        return {"messages": updated_messages}
    
    # 创建使用内置函数的图
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("agent", windowed_agent_node)
    graph_builder.set_entry_point("agent")
    graph_builder.set_finish_point("agent")
    
    # 编译图
    windowed_agent_graph = graph_builder.compile(checkpointer=InMemorySaver())
    
    # 定义带窗口大小配置的线程配置
    windowed_agent_config = {
        "configurable": {
            "thread_id": "windowed_agent_thread",
            "window_size": 3  # 更严格的窗口大小
        }
    }
    
    print("  使用内置函数和配置化窗口大小:")
    print("  窗口大小设置为: 3")
    
    # 发送初始消息
    initial_messages = [
        {"role": "system", "content": "你是一个简洁的助手。"},
        {"role": "user", "content": "你好，测试窗口控制！"}
    ]
    
    response_agent_1 = windowed_agent_graph.invoke(
        {"messages": initial_messages},
        windowed_agent_config
    )
    
    print(f"    用户: {initial_messages[1]['content']}")
    print(f"    助手: {response_agent_1['messages'][-1].content}")
    print(f"    当前消息数量: {len(response_agent_1['messages'])}")
    
    # 发送更多消息测试
    test_queries = ["测试消息1", "测试消息2", "测试消息3"]
    
    for query in test_queries:
        response = windowed_agent_graph.invoke(
            {"messages": [{"role": "user", "content": query}]},
            windowed_agent_config
        )
        
        print(f"    用户: {query}")
        print(f"    助手: {response['messages'][-1].content}")
        print(f"    当前消息数量: {len(response['messages'])}")
        print(f"    消息内容预览: {[msg.role for msg in response['messages']]}")
        print()
    
    print("\n  使用内置函数的优势:")
    print("  - 可以通过配置动态调整窗口大小")
    print("  - 利用LangGraph内置的消息处理功能")
    print("  - 可以根据不同线程设置不同的窗口大小")
    
except Exception as e:
    print(f"✗ 使用内置函数控制窗口大小时出错: {e}")

# ===== 聊天窗口大小控制总结 =====
print("\n" + "=" * 80)
print("聊天窗口大小控制最佳实践")
print("=" * 80)
print("\n实现方法:")
print("1. 自定义状态类法:")
print("   - 继承MessagesState添加窗口大小字段")
print("   - 在节点函数中实现消息截断逻辑")
print("   - 适合需要固定窗口大小的场景")
print("")
print("2. 配置化窗口法:")
print("   - 通过configurable配置传递窗口大小")
print("   - 使用内置的消息处理函数")
print("   - 支持为不同线程设置不同窗口")
print("")
print("3. 高级策略:")
print("   - 保留系统提示消息不被截断")
print("   - 可以基于token数量而非消息数量进行截断")
print("   - 考虑在不同阶段使用不同的窗口大小")
print("")
print("注意事项:")
print("- 窗口大小过小可能导致模型失去上下文理解能力")
print("- 窗口大小过大会增加token消耗和响应时间")
print("- 对于需要长期记忆的场景，可以结合向量数据库等外部存储")
print("- 推荐窗口大小: 根据模型上下文限制和应用需求，通常设置为5-20轮对话")

# ===== 两种方式的详细对比总结 =====
print("\n" + "=" * 80)
print("手动维护对话历史 VS LangGraph 线程管理")
print("=" * 80)
print("\n对比维度:")
print("1. 历史记录管理方式:")
print("   - 手动维护: 开发者需要创建列表并手动添加/更新每一条消息")
print("   - 线程管理: 由checkpointer自动保存和恢复，通过thread_id索引")
print("")
print("2. 多会话处理:")
print("   - 手动维护: 需要额外的数据结构(如字典)来区分不同会话")
print("   - 线程管理: 内置支持多会话，只需切换thread_id")
print("")
print("3. 调用方式:")
print("   - 手动维护: 每次调用必须传递完整历史记录")
print("   - 线程管理: 只需传递新消息，系统自动关联历史")
print("")
print("4. 功能集成:")
print("   - 手动维护: 仅提供基本对话能力，需要自己实现工具调用等功能")
print("   - 线程管理: 与LangGraph生态集成，支持工具调用、状态管理等高级功能")
print("")
print("5. 适用场景:")
print("   - 手动维护: 简单应用、单用户场景、快速原型开发")
print("   - 线程管理: 复杂应用、多用户系统、生产环境、需要工具调用的智能体")
print("")
print("6. 可扩展性:")
print("   - 手动维护: 扩展困难，需要手动处理更多复杂性")
print("   - 线程管理: 易于扩展，可与其他LangGraph组件无缝集成")
print("")
print("最佳实践建议:")
print("- 简单脚本或测试: 手动维护对话历史更直接")
print("- 生产级应用或智能体: 使用LangGraph的线程管理更可靠、可扩展")
print("- 多用户场景: 强烈推荐使用线程管理，避免会话混乱")
print("=" * 80)
print("所有测试完成！")
print("=" * 80)