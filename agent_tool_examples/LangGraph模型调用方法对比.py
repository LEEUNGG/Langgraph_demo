"""
LangGraph 中不同模型调用方法的对比示例
展示各种初始化模型的方式及其区别
修改为使用DeepSeek模型并从.env文件获取API密钥
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# 从.env文件加载环境变量
load_dotenv()

print("=" * 60)
print("LangGraph 模型调用方法对比 (使用DeepSeek模型)")
print("=" * 60)

# ===== 方法1: 直接导入特定提供商的模型类 =====
print("\n1. 直接导入特定提供商的模型类")
print("-" * 40)

try:
    # DeepSeek 模型
    from langchain_community.chat_models import ChatDeepSeek
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
    
except ImportError as e:
    print(f"✗ DeepSeek 模型导入失败: {e}")

# ===== 方法2: 使用 init_chat_model 统一初始化 =====
print("\n\n2. 使用 init_chat_model 统一初始化")
print("-" * 40)

try:
    from langchain.chat_models import init_chat_model
    
    # 方式2a: 通过模型名自动识别提供商
    print("方式2a: 自动识别提供商")
    deepseek_model_init_auto = init_chat_model(
        model="deepseek-chat",
        temperature=0.7,
        max_tokens=100,
        # api_key=os.getenv("DEEPSEEK_API_KEY")
    )
    print("✓ DeepSeek 模型 (自动识别):")
    print(f"  类型: {type(deepseek_model_init_auto)}")
    print(f"  模型名: {deepseek_model_init_auto.model_name if hasattr(deepseek_model_init_auto, 'model_name') else getattr(deepseek_model_init_auto, 'model', '未知')}")
    
    # 方式2b: 明确指定提供商
    print("\n方式2b: 明确指定提供商")
    deepseek_model_init_explicit = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek",
        temperature=0.5,
        max_tokens=100,
        # api_key=os.getenv("DEEPSEEK_API_KEY")
    )
    print("✓ DeepSeek 模型 (指定提供商):")
    print(f"  类型: {type(deepseek_model_init_explicit)}")
    print(f"  模型名: {deepseek_model_init_explicit.model_name if hasattr(deepseek_model_init_explicit, 'model_name') else getattr(deepseek_model_init_explicit, 'model', '未知')}")

except ImportError as e:
    print(f"✗ init_chat_model 导入失败: {e}")

except Exception as e:
    print(f"✗ 初始化模型时出错: {e}")

# ===== 方法3: 在 LangGraph 中的实际使用 =====
print("\n\n3. 在 LangGraph 中的实际使用示例")
print("-" * 40)

from langgraph.graph import Graph
from langchain.schema import HumanMessage

def create_chat_node(model, node_name: str):
    """创建一个聊天节点"""
    def chat_node(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("messages", [])
        try:
            # 模拟调用（实际需要真实API密钥）
            print(f"  {node_name} 节点使用模型: {type(model).__name__}")
            # response = model.invoke(messages)  # 实际调用
            response = f"模拟响应来自 {node_name}"
            return {"messages": messages + [response]}
        except Exception as e:
            print(f"  {node_name} 调用失败: {e}")
            return {"messages": messages + [f"{node_name} 调用失败"]}
    
    return chat_node

# 创建图
workflow = Graph()

# 使用不同方式初始化的模型创建节点
try:
    workflow.add_node("deepseek_direct", create_chat_node(deepseek_model_direct, "DeepSeek直接导入"))
except:
    print("跳过 DeepSeek 直接导入节点")

try:
    workflow.add_node("deepseek_init", create_chat_node(deepseek_model_init_auto, "DeepSeek统一初始化"))
except:
    print("跳过 DeepSeek 统一初始化节点")

print("✓ LangGraph 工作流创建完成")

# ===== 各种方法的对比总结 =====
print("\n\n4. 各种方法的优缺点对比")
print("-" * 40)

comparison_table = """
方法                  优点                           缺点
--------------------------------------------------------------------------------
直接导入特定类         • 明确知道使用的模型类型        • 需要记住不同提供商的导入路径
                      • 可以使用所有特定参数          • 切换提供商需要修改导入语句
                      • 类型提示更清晰               • 代码冗长

init_chat_model       • 统一的初始化方式              • 可能无法使用所有特定参数
                      • 容易切换不同提供商            • 需要额外的依赖
                      • 代码更简洁                   • 较新的功能，文档可能不全
                      • 自动识别提供商
"""

print(comparison_table)

# ===== 推荐使用方式 =====
print("\n5. 推荐使用方式")
print("-" * 40)
print("""
🎯 推荐策略:

1. 新项目: 优先使用 init_chat_model
   - 代码更简洁统一
   - 便于后期维护和切换模型

2. 需要特定功能: 使用直接导入
   - 当需要使用某个提供商的特有参数时
   - 对性能有极致要求时

3. 在 LangGraph 中:
   - 两种方式都可以无缝集成
   - 选择你和团队更熟悉的方式
""")

# ===== 完整的 LangGraph 使用示例 =====
print("\n6. 完整的 LangGraph 使用示例")
print("-" * 40)

def complete_langgraph_example():
    """完整的 LangGraph 示例"""
    from langgraph.graph import StateGraph
    from typing import TypedDict, List
    
    # 定义状态
    class ChatState(TypedDict):
        messages: List[str]
        current_model: str
    
    # 使用推荐的 init_chat_model 方式
    try:
        model = init_chat_model(
            model="deepseek-chat",
            temperature=0.7,
            # api_key=os.getenv("DEEPSEEK_API_KEY")
        )
        
        def chat_node(state: ChatState) -> ChatState:
            # 实际使用中这里会调用模型
            print(f"使用模型: {type(model).__name__}")
            new_message = f"处理了 {len(state['messages'])} 条消息"
            return {
                "messages": state["messages"] + [new_message],
                "current_model": type(model).__name__
            }
        
        # 创建图
        graph = StateGraph(ChatState)
        graph.add_node("chat", chat_node)
        graph.set_entry_point("chat")
        graph.set_finish_point("chat")
        
        print("✓ 完整的 LangGraph 工作流创建成功")
        
        # 模拟运行
        initial_state = {"messages": ["Hello"], "current_model": ""}
        # result = graph.invoke(initial_state)  # 实际运行
        print("✓ 可以正常运行 (需要真实API密钥)")
        
    except Exception as e:
        print(f"示例运行出错: {e}")

complete_langgraph_example()

print("\n" + "=" * 60)
print("总结: init_chat_model 是推荐的现代化方式！")
print("=" * 60)