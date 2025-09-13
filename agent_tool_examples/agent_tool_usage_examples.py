# agent_tool_usage_examples.py
from typing import Dict, Any
import random
import os

# LangGraph / LangChain 相关
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv

# 加载.env文件（从项目根目录）
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# -----------------------
# 模拟工具函数
# -----------------------
def get_weather(loc: str) -> str:
    """模拟查询天气"""
    conditions = ["晴天", "多云", "小雨", "雷阵雨", "阴天"]
    temp = random.randint(20, 35)
    return f"{loc} 今天天气：{random.choice(conditions)}，气温 {temp}°C"


# -----------------------
# 用 bind_tools 手写 Agent Loop
# -----------------------
def run_bind_tools_example():
    print("=== 方法一：使用 bind_tools 手动实现 Agent Loop ===")
    # 用 init_chat_model 配置模型
    model = init_chat_model(model="deepseek-chat", model_provider="deepseek").bind_tools(
        tools=[{"name": "get_weather", "func": get_weather}]
    )

    # 初始化对话历史
    history = [HumanMessage(content="广州今天的天气怎么样？")]

    while True:
        # 模型生成
        ai_response = model.invoke(history)
        print("\n方法一 - ai_response 内容：")
        print(f"类型: {type(ai_response)}")
        print(f"工具调用: {hasattr(ai_response, 'tool_calls') and ai_response.tool_calls}")
        if hasattr(ai_response, 'tool_calls'):
            print(f"工具调用详情: {ai_response.tool_calls}")
        print(f"内容: {getattr(ai_response, 'content', '无内容')}")
        history.append(ai_response)

        # 如果触发了工具调用
        if hasattr(ai_response, "tool_calls") and ai_response.tool_calls:
            for tool_call in ai_response.tool_calls:
                if tool_call["name"] == "get_weather":
                    # 检查参数格式并确保正确提取loc参数
                    args = tool_call["args"]
                    # 安全地获取loc参数，处理可能的嵌套结构
                    loc = args.get("loc", "")
                    if not loc:
                        # 尝试从其他可能的参数结构中获取
                        for key, value in args.items():
                            if key.lower() in ["loc", "location", "城市"]:
                                loc = value
                                break
                    result = get_weather(loc) if loc else "参数错误：缺少位置信息"

                    # 把工具结果作为 ToolMessage 加入历史
                    history.append(
                        ToolMessage(
                            content=result,
                            tool_call_id=tool_call["id"]
                        )
                    )
        else:
            # 没有工具调用，说明模型给出最终答案，结束
            print("模型回答：", ai_response.content)
            break


# -----------------------
# 用 create_react_agent 快速实现
# -----------------------
def run_create_react_agent_example():
    print("\n=== 方法二：使用 create_react_agent 快速实现 ===")
    # 配置模型
    model = init_chat_model(model="deepseek-chat", model_provider="deepseek")

    # 定义工具
    tools = [{"name": "get_weather", "func": get_weather}]

    # 创建 ReAct agent
    agent = create_react_agent(model, tools=tools)

    # 输入一次
    result = agent.invoke({"messages": [HumanMessage(content="广州今天的天气怎么样？")]})
    print("\n方法二 - agent.invoke 返回结果：")
    print(f"结果类型: {type(result)}")
    print(f"结果结构: {list(result.keys())}")
    print(f"消息列表长度: {len(result['messages'])}")
    for i, msg in enumerate(result['messages']):
        print(f"消息 {i} 类型: {type(msg)}")
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"消息 {i} 工具调用: {msg.tool_calls}")
        print(f"消息 {i} 内容: {getattr(msg, 'content', '无内容')}")

    # 输出结果
    final_message = result["messages"][-1]
    print("模型回答：", final_message.content)


# -----------------------
# 主入口
# -----------------------
if __name__ == "__main__":
    run_bind_tools_example()
    run_create_react_agent_example()