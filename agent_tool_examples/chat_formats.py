# -*- coding: utf-8 -*-
"""
LangChain 聊天记录格式示例
专注于展示不同的消息格式和聊天历史管理方式
"""

import os
from dotenv import load_dotenv

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构造上层目录的.env文件路径
env_path = os.path.join(current_dir, '..', '.env')
load_dotenv(dotenv_path=env_path)

print("=" * 80)
print("LangChain 聊天记录格式示例")
print("=" * 80)

# ===== 初始化模型 =====
print("\n初始化模型")
print("-" * 60)

try:
    from langchain.chat_models import init_chat_model
    from langchain.schema import HumanMessage, AIMessage, SystemMessage
    
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

# ===== 格式1: LangChain Message 对象格式 =====
print("\n\n1. LangChain Message 对象格式")
print("-" * 60)

try:
    # 准备对话历史
    conversation_history = [
        SystemMessage(content="你是一个专业的助手，回答要简洁明了。"),
        HumanMessage(content="你好！"),
        AIMessage(content="你好！有什么我可以帮助你的吗？"),
        HumanMessage(content="请简单介绍一下你自己。")
    ]
    
    print("  消息对象类型:")
    for i, msg in enumerate(conversation_history):
        print(f"  {i+1}. {type(msg).__name__}: {msg.content[:30]}...")
    
    # 发送新问题
    new_question = "你刚才说的内容能详细一点吗？"
    conversation_history.append(HumanMessage(content=new_question))
    
    print(f"\n  发送新问题: {new_question}")
    response = model.invoke(conversation_history)
    print(f"  模型响应: {response.content}")
    
    print("\n  LangChain Message 对象格式的特点:")
    print("  - 类型明确，便于代码处理和类型检查")
    print("  - 支持IDE自动补全和类型提示")
    print("  - 在LangChain生态系统中集成良好")
    
except Exception as e:
    print(f"✗ 使用LangChain Message对象格式时出错: {e}")

# ===== 格式2: OpenAI 风格格式 =====
print("\n\n2. OpenAI 风格格式 (字典列表)")
print("-" * 60)

try:
    # 准备OpenAI风格的对话历史
    openai_style_messages = [
        {"role": "system", "content": "你是一个专业的助手，回答要简洁明了。"},
        {"role": "user", "content": "你好！"},
        {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"},
        {"role": "user", "content": "请简单介绍一下你自己。"}
    ]
    
    print("  消息格式:")
    for i, msg in enumerate(openai_style_messages):
        print(f"  {i+1}. {msg['role']}: {msg['content'][:30]}...")
    
    # 发送新问题
    new_question = "你刚才说的内容能详细一点吗？"
    openai_style_messages.append({"role": "user", "content": new_question})
    
    print(f"\n  发送新问题: {new_question}")
    response = model.invoke(openai_style_messages)
    print(f"  模型响应: {response.content}")
    
    print("\n  OpenAI风格格式的特点:")
    print("  - 通用性强，兼容多种模型API")
    print("  - 格式简单直观，易于理解")
    print("  - 便于序列化和存储")
    print("  - 跨平台、跨模型兼容性好")
    
except Exception as e:
    print(f"✗ 使用OpenAI风格格式时出错: {e}")

# ===== 格式3: 高级封装格式 =====
print("\n\n3. 高级封装格式")
print("-" * 60)

try:
    # 示例1: 携带会话ID和元数据的高级封装
    print("  示例1: 携带会话ID和元数据")
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
    
    print("  高级封装格式包含的额外参数:")
    print(f"  - 会话ID: {advanced_message_with_context['session_id']}")
    print(f"  - 用户ID: {advanced_message_with_context['user_id']}")
    print(f"  - 模型配置: 温度={advanced_message_with_context['model_config']['temperature']}, ")
    print(f"              最大tokens={advanced_message_with_context['model_config']['max_tokens']}")
    print(f"  - 用户偏好: {advanced_message_with_context['context']['user_preferences']}")
    print(f"  - 系统提示: {advanced_message_with_context['messages'][0]['content']}")
    print(f"  - 提问: {advanced_message_with_context['messages'][1]['content']}")
    
    # 提取messages部分进行模型调用
    response = model.invoke(advanced_message_with_context['messages'])
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
    response = model.invoke(langgraph_style_message['messages'])
    print(f"  响应: {response.content}")
    
except Exception as e:
    print(f"✗ 使用高级封装格式时出错: {e}")

# ===== 不同聊天格式总结 =====
print("\n" + "=" * 80)
print("不同聊天记录格式总结")
print("=" * 80)
print("1. LangChain Message 对象格式:")
print("   - 格式: SystemMessage, HumanMessage, AIMessage 对象")
print("   - 优点: 类型明确，便于代码处理和类型检查")
print("   - 适用: LangChain生态系统内的开发")
print("")
print("2. OpenAI 风格格式 (字典列表):")
print('   - 格式: [{"role": "user/assistant/system", "content": "消息内容"}]')
print("   - 优点: 通用性强，兼容多种模型API")
print("   - 适用: 跨平台、跨模型的开发场景")
print("")
print("3. 高级封装格式:")
print('   - 格式: {"messages": [OpenAI风格消息列表], "session_id": "xxx", "metadata": {...}, ...}')
print("   - 优点: 可以携带丰富的上下文信息、会话元数据、模型配置等额外参数")
print("   - 适用: LangGraph等高级框架中的智能体开发，企业级多用户系统")
print("")
print("最佳实践:")
print("- 在LangChain环境中，优先使用LangChain Message对象")
print("- 需要跨平台兼容性时，使用OpenAI风格格式")
print("- 在复杂框架中，使用高级封装格式")
print("=" * 80)