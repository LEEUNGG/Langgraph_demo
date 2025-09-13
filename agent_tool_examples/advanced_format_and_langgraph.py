# -*- coding: utf-8 -*-
"""
高级封装格式与LangGraph高级用法
专注于展示高级封装格式的应用和LangGraph的复杂功能
"""

import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState, START, StateGraph, END, CompiledGraph
from langgraph.prebuilt import create_react_agent, ToolNode
from langchain.tools import tool, StructuredTool, tool
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构造上层目录的.env文件路径
env_path = os.path.join(current_dir, '..', '.env')
load_dotenv(dotenv_path=env_path)

print("=" * 80)
print("高级封装格式与LangGraph高级用法")
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
        max_tokens=500
    )
    print("✓ 模型初始化成功")
    
except Exception as e:
    print(f"✗ 模型初始化失败: {e}")
    exit(1)

# ===== 1. 高级封装格式的完整应用 =====
print("\n\n1. 高级封装格式的完整应用")
print("-" * 60)

try:
    # 定义一个高级封装格式的处理器类
    class AdvancedMessageProcessor:
        def __init__(self, model):
            self.model = model
            
        def process_message(self, advanced_message: Dict[str, Any]) -> Dict[str, Any]:
            """处理高级封装格式的消息"""
            # 提取消息内容
            messages = advanced_message.get("messages", [])
            
            # 提取额外参数
            session_id = advanced_message.get("session_id", "default_session")
            user_id = advanced_message.get("user_id", "anonymous")
            model_config = advanced_message.get("model_config", {})
            context = advanced_message.get("context", {})
            
            print(f"  处理消息: 会话ID={session_id}, 用户ID={user_id}")
            print(f"  上下文信息: {context}")
            
            # 应用模型配置（如果有）
            if model_config:
                # 在实际应用中，这里会根据model_config动态调整模型参数
                print(f"  应用模型配置: {model_config}")
            
            # 调用模型
            response = self.model.invoke(messages)
            
            # 构建响应对象
            result = {
                "response": response.content,
                "metadata": {
                    "session_id": session_id,
                    "user_id": user_id,
                    "processing_time": "示例: 1.2秒",
                    "context_used": context
                }
            }
            
            return result
    
    # 创建处理器实例
    processor = AdvancedMessageProcessor(model)
    
    # 示例1: 携带完整元数据的高级封装
    print("  示例1: 携带完整元数据的高级封装")
    complete_message = {
        "messages": [
            {"role": "system", "content": "你是一个专业的AI助手，擅长数据分析。"},
            {"role": "user", "content": "请分析一下2023年全球AI发展趋势。"}
        ],
        "session_id": "analytics_session_789",
        "user_id": "data_scientist_456",
        "model_config": {
            "temperature": 0.8,
            "max_tokens": 500,
            "top_p": 0.95
        },
        "context": {
            "user_profile": {"domain": "数据科学", "expertise": "高级"},
            "preferences": {"language": "中文", "output_format": "要点式"},
            "session_history": {
                "previous_topics": ["机器学习基础", "深度学习框架"],
                "conversation_length": 3
            }
        }
    }
    
    # 处理消息
    result1 = processor.process_message(complete_message)
    
    print(f"  \n  模型响应: {result1['response'][:150]}...")
    print(f"  响应元数据: {result1['metadata']}")
    
    # 示例2: 多轮对话场景下的高级封装
    print("\n  示例2: 多轮对话场景下的高级封装")
    
    # 第一轮消息
    first_message = {
        "messages": [
            {"role": "system", "content": "你是一个编程助手。"},
            {"role": "user", "content": "如何用Python实现一个简单的计算器？"}
        ],
        "session_id": "coding_session_123",
        "user_id": "developer_789",
        "context": {
            "skill_level": "初级",
            "preferred_language": "Python"
        }
    }
    
    result2_1 = processor.process_message(first_message)
    print(f"  第一轮 - 用户: 如何用Python实现一个简单的计算器？")
    print(f"  第一轮 - 助手: {result2_1['response'][:100]}...")
    
    # 第二轮消息，使用第一轮的结果作为上下文
    second_message = {
        "messages": [
            {"role": "system", "content": "你是一个编程助手。"},
            {"role": "user", "content": "如何用Python实现一个简单的计算器？"},
            {"role": "assistant", "content": result2_1['response']},
            {"role": "user", "content": "如何扩展这个计算器，让它支持科学计算功能？"}
        ],
        "session_id": "coding_session_123",  # 使用相同的会话ID
        "user_id": "developer_789",
        "context": {
            "skill_level": "初级",
            "preferred_language": "Python",
            "conversation_stage": "深入讨论",
            "previous_response_id": "response_001"
        }
    }
    
    result2_2 = processor.process_message(second_message)
    print(f"  第二轮 - 用户: 如何扩展这个计算器，让它支持科学计算功能？")
    print(f"  第二轮 - 助手: {result2_2['response'][:100]}...")
    
    print("\n  高级封装格式的优势:")
    print("  - 结构化设计，便于扩展和维护")
    print("  - 支持丰富的上下文信息和元数据")
    print("  - 适合复杂的多轮对话场景")
    print("  - 便于与其他系统集成")
    
except Exception as e:
    print(f"✗ 高级封装格式应用时出错: {e}")

# ===== 2. LangGraph 高级用法 - 复杂工作流 =====
print("\n\n2. LangGraph 高级用法 - 复杂工作流")
print("-" * 60)

try:
    # 定义工具
    @tool
    def search_weather(location: str) -> str:
        """搜索指定位置的天气信息"""
        return f"{location}的天气: 晴朗，温度25°C，风力2级"
    
    @tool
    def calculate(expression: str) -> str:
        """计算数学表达式，只支持简单的加减乘除"""
        try:
            # 安全计算（实际应用中需要更严格的安全检查）
            result = eval(expression)
            return f"计算结果: {expression} = {result}"
        except:
            return f"无法计算表达式: {expression}"
    
    tools = [search_weather, calculate]
    
    # 定义自定义状态
    class AdvancedState(MessagesState):
        query_type: str = ""
        confidence_score: float = 0.0
        processing_stage: str = "initial"
        
    # 定义节点函数
    def classify_query(state: AdvancedState) -> Dict[str, Any]:
        """分类用户查询"""
        last_message = state.messages[-1].content
        
        # 简单的查询分类逻辑
        if any(keyword in last_message.lower() for keyword in ["天气", "温度", "下雨"]):
            query_type = "weather"
            confidence = 0.9
        elif any(keyword in last_message.lower() for keyword in ["计算", "加", "减", "乘", "除"]):
            query_type = "calculation"
            confidence = 0.9
        else:
            query_type = "general"
            confidence = 0.8
        
        print(f"  分类查询: {query_type}, 置信度: {confidence}")
        
        return {
            "query_type": query_type,
            "confidence_score": confidence,
            "processing_stage": "classified"
        }
    
    def process_with_tools(state: AdvancedState) -> Dict[str, Any]:
        """使用工具处理查询"""
        # 创建工具节点
        tool_node = ToolNode(tools)
        
        # 使用工具节点处理消息
        result = tool_node.invoke(state)
        
        return {
            "messages": result["messages"],
            "processing_stage": "tools_applied"
        }
    
    def process_general(state: AdvancedState) -> Dict[str, Any]:
        """处理一般查询"""
        # 调用模型处理一般查询
        response = model.invoke([
            SystemMessage(content="你是一个专业助手，回答要简洁明了。")
        ] + state.messages)
        
        return {
            "messages": [response],
            "processing_stage": "completed"
        }
    
    def decide_next_step(state: AdvancedState) -> str:
        """决定下一步操作"""
        if state.query_type in ["weather", "calculation"]:
            return "use_tools"
        else:
            return "general_process"
    
    # 创建状态图
    workflow = StateGraph(AdvancedState)
    
    # 添加节点
    workflow.add_node("classify", classify_query)
    workflow.add_node("use_tools", process_with_tools)
    workflow.add_node("general_process", process_general)
    
    # 设置边
    workflow.add_edge(START, "classify")
    workflow.add_conditional_edges("classify", decide_next_step)
    workflow.add_edge("use_tools", END)
    workflow.add_edge("general_process", END)
    
    # 编译图，添加检查点
    app = workflow.compile(checkpointer=InMemorySaver())
    
    # 测试不同类型的查询
    print("  测试复杂工作流处理不同类型的查询:")
    
    # 测试天气查询
    print("\n  测试1: 天气查询")
    weather_result = app.invoke({
        "messages": [HumanMessage(content="北京今天的天气怎么样？")]
    })
    
    print(f"  用户: 北京今天的天气怎么样？")
    print(f"  助手: {weather_result['messages'][-1].content}")
    print(f"  处理阶段: {weather_result['processing_stage']}")
    
    # 测试计算查询
    print("\n  测试2: 计算查询")
    calc_result = app.invoke({
        "messages": [HumanMessage(content="计算一下123乘以456等于多少？")]
    })
    
    print(f"  用户: 计算一下123乘以456等于多少？")
    print(f"  助手: {calc_result['messages'][-1].content}")
    print(f"  处理阶段: {calc_result['processing_stage']}")
    
    # 测试一般查询
    print("\n  测试3: 一般查询")
    general_result = app.invoke({
        "messages": [HumanMessage(content="请解释什么是人工智能？")]
    })
    
    print(f"  用户: 请解释什么是人工智能？")
    print(f"  助手: {general_result['messages'][-1].content[:100]}...")
    print(f"  处理阶段: {general_result['processing_stage']}")
    
    print("\n  LangGraph复杂工作流的优势:")
    print("  - 可视化的工作流设计")
    print("  - 支持条件分支和循环")
    print("  - 集成工具调用能力")
    print("  - 支持状态持久化和恢复")
    print("  - 适合构建复杂的多步骤智能体")
    
except Exception as e:
    print(f"✗ LangGraph高级用法应用时出错: {e}")

# ===== 3. 高级封装格式与LangGraph的结合 =====
print("\n\n3. 高级封装格式与LangGraph的结合")
print("-" * 60)

try:
    # 定义一个结合高级封装格式和LangGraph的处理类
    class AdvancedAgent:
        def __init__(self, model, tools=None):
            self.model = model
            self.tools = tools or []
            self.checkpointer = InMemorySaver()
            self.graph = self._build_graph()
            
        def _build_graph(self) -> CompiledGraph:
            """构建LangGraph图"""
            class AgentState(MessagesState):
                agent_metadata: Dict[str, Any] = {}
                
            def process_node(state: AgentState):
                """处理节点"""
                # 提取元数据
                metadata = state.agent_metadata
                
                # 构造系统消息，包含元数据信息
                system_content = "你是一个专业助手。"
                if metadata.get("user_preferences"):
                    system_content += f" 用户偏好: {metadata['user_preferences']}"
                
                system_msg = SystemMessage(content=system_content)
                
                # 调用模型
                response = model.invoke([system_msg] + state.messages)
                
                return {"messages": [response]}
            
            # 创建状态图
            workflow = StateGraph(AgentState)
            workflow.add_node("process", process_node)
            workflow.add_edge(START, "process")
            workflow.add_edge("process", END)
            
            # 编译图
            return workflow.compile(checkpointer=self.checkpointer)
        
        def process_advanced_message(self, advanced_message: Dict[str, Any]) -> Dict[str, Any]:
            """处理高级封装格式的消息"""
            # 提取信息
            messages = advanced_message.get("messages", [])
            session_id = advanced_message.get("session_id", "default")
            metadata = advanced_message.get("metadata", {})
            
            # 准备LangGraph输入
            input_data = {
                "messages": [HumanMessage(content=msg["content"]) for msg in messages if msg["role"] == "user"][-1:],
                "agent_metadata": metadata
            }
            
            # 准备配置
            config = {"configurable": {"thread_id": session_id}}
            
            # 运行图
            result = self.graph.invoke(input_data, config=config)
            
            # 构建响应
            response = {
                "response": result["messages"][-1].content,
                "session_id": session_id,
                "processed_metadata": metadata
            }
            
            return response
    
    # 创建智能体实例
    agent = AdvancedAgent(model)
    
    # 测试结合使用
    print("  测试高级封装格式与LangGraph的结合:")
    
    advanced_combined_message = {
        "messages": [
            {"role": "user", "content": "什么是LangGraph？请用简洁的语言解释。"}
        ],
        "session_id": "advanced_combination_001",
        "metadata": {
            "user_preferences": {"language": "中文", "detail_level": "中等"},
            "session_info": {"source": "web", "timestamp": "2023-10-01T12:00:00Z"},
            "user_profile": {"expertise": "初级", "domain": "AI开发"}
        }
    }
    
    combined_result = agent.process_advanced_message(advanced_combined_message)
    
    print(f"  用户: {advanced_combined_message['messages'][0]['content']}")
    print(f"  助手: {combined_result['response']}")
    print(f"  使用的会话ID: {combined_result['session_id']}")
    print(f"  处理的元数据: {combined_result['processed_metadata']}")
    
    print("\n  高级封装格式与LangGraph结合的优势:")
    print("  - 结合了高级封装的灵活性和LangGraph的强大工作流能力")
    print("  - 支持复杂的元数据处理和状态管理")
    print("  - 便于构建企业级的智能体应用")
    print("  - 提供了清晰的模块化结构")
    
except Exception as e:
    print(f"✗ 高级封装格式与LangGraph结合时出错: {e}")

# ===== 最佳实践总结 =====
print("\n" + "=" * 80)
print("高级封装格式与LangGraph最佳实践总结")
print("=" * 80)
print("\n高级封装格式的最佳实践:")
print("1. 结构化设计:")
print("   - 使用统一的JSON结构进行消息封装")
print("   - 明确定义必选字段和可选字段")
print("   - 为元数据设计清晰的层次结构")
print("")
print("2. 上下文管理:")
print("   - 在多轮对话中有效传递上下文信息")
print("   - 合理使用会话ID和用户ID进行会话管理")
print("   - 根据用户偏好调整响应风格")
print("")
print("3. LangGraph应用策略:")
print("   - 简单应用: 使用预构建的智能体和检查点")
print("   - 中等复杂度: 自定义状态和简单工作流")
print("   - 复杂应用: 构建多节点工作流和条件分支")
print("")
print("4. 集成建议:")
print("   - 在企业应用中，结合数据库存储会话历史和元数据")
print("   - 使用缓存优化频繁访问的数据")
print("   - 实现错误处理和重试机制")
print("")
print("5. 性能优化:")
print("   - 合理控制上下文窗口大小")
print("   - 使用异步处理提高并发能力")
print("   - 对大型工作流进行性能监控和调优")
print("=" * 80)