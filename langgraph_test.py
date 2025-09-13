# weather_agent_demo_init.py
from typing import Dict, Any
import random

# LangGraph / LangChain 相关
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv

load_dotenv()

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
    print("=== bind_tools 示例 ===")
    # 用 init_chat_model 替代 ChatOpenAI
    model = init_chat_model(model="deepseek-chat", model_provider="deepseek").bind_tools(
        tools=[{"name": "get_weather", "func": get_weather}]
    )

    # 初始化对话历史
    history = [HumanMessage(content="广州今天的天气怎么样？")]

    while True:
        # 模型生成
        ai_response = model.invoke(history)
        history.append(ai_response)

        # 如果触发了工具调用
        if hasattr(ai_response, "tool_calls") and ai_response.tool_calls:
            for tool_call in ai_response.tool_calls:
                if tool_call["name"] == "get_weather":
                    args: Dict[str, Any] = tool_call["args"]
                    result = get_weather(**args)

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
# 用 create_react_agent 示例
# -----------------------
def run_create_react_agent_example():
    print("\n=== create_react_agent 示例 ===")
    # 这里直接用 init_chat_model
    model = init_chat_model(model="deepseek-chat", model_provider="deepseek")

    # 定义工具
    tools = [{"name": "get_weather", "func": get_weather}]

    # 创建 ReAct agent
    agent = create_react_agent(model, tools=tools)

    # 输入一次
    result = agent.invoke({"messages": [HumanMessage(content="广州今天的天气怎么样？")]})

    # 输出结果
    final_message = result["messages"][-1]
    print("模型回答：", final_message.content)


# -----------------------
# 主入口
# -----------------------
if __name__ == "__main__":
    run_bind_tools_example()
    run_create_react_agent_example()
