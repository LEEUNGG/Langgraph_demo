import streamlit as st
import asyncio
import nest_asyncio
import json
import os
import platform
# ----- 1. 页面和CSS美化 -----

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Apply nest_asyncio: Allow nested calls within an already running event loop
nest_asyncio.apply()

# Create and reuse global event loop (create once and continue using)
if "event_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    st.session_state.event_loop = loop
    asyncio.set_event_loop(loop)

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from utils import astream_graph, random_uuid
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_deepseek import ChatDeepSeek

# Load environment variables (get API keys and settings from .env file)
load_dotenv(override=True)

# config.json file path setting
CONFIG_FILE_PATH = "config.json"


# 从 JSON 文件中加载设置
def load_config_from_json():
    """
    Loads settings from config.json file.
    Creates a file with default settings if it doesn't exist.

    Returns:
        dict: Loaded settings
    """
    default_config = {
        "get_current_time": {
            "command": "python",
            "args": ["./mcp_server_time.py"],
            "transport": "stdio"
        }
    }
    
    try:
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            # Create file with default settings if it doesn't exist
            save_config_to_json(default_config)
            return default_config
    except Exception as e:
        st.error(f"Error loading settings file: {str(e)}")
        return default_config

# 将设置保存到 JSON 文件
def save_config_to_json(config):
    """
    Saves settings to config.json file.

    Args:
        config (dict): Settings to save
    
    Returns:
        bool: Save success status
    """
    try:
        with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error saving settings file: {str(e)}")
        return False

# 初始化登录 session 变量
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# 检查是否需要登录
use_login = os.environ.get("USE_LOGIN", "false").lower() == "true"

# 根据登录状态更改页面设置
if use_login and not st.session_state.authenticated:
    # 登录页面使用默认（窄）布局
    st.set_page_config(page_title="LangGraph Agent MCP Tools", page_icon="🧠")
else:
    # 主应用使用宽布局
    st.set_page_config(page_title="LangGraph Agent MCP Tools", page_icon="🧠", layout="wide")


# 登录页面CSS美化
def inject_css():
    st.markdown("""
        <style>
        /* 背景渐变 */
        body {
            background: linear-gradient(120deg, #f8fafc 0%, #dbeafe 100%) !important;
        }
        .main {
            background-color: rgba(255,255,255,0.95);
            border-radius: 24px;
            box-shadow: 0 4px 32px rgba(30, 64, 175, 0.10);
            padding: 2.5em 2em 2em 2em;
            max-width: 380px;
            margin: 3em auto 0 auto;
        }
        header, .st-emotion-cache-1avcm0n {display: none !important;}

        .stTextInput>div>div>input {
            border-radius: 1em;
            padding: 0.6em 0.9em;
        }
        .login-logo {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1.2em;
        }
        .login-logo img {
            height: 48px;
            margin-right: 12px;
        }
        .login-tip {
            color: #64748b;
            margin-bottom: 1.5em;
            font-size: 1.05em;
            text-align: center;
        }
                
        /* 让按钮自适应宽度，并居中显示 */
        .stForm .stButton {
            display: flex;
            justify-content: center;
        }

        .stButton > button {
            width: 140px !important;     /* 自己设宽度，比如140px，也可以80%、fit-content等 */
            margin: 0 auto;
        }

        </style>
    """, unsafe_allow_html=True)

inject_css()

# ----- 2. 登录逻辑 -----
use_login = True  # 按需设置

if use_login and not st.session_state.get("authenticated", False):
    with st.container():
        # logo和标题
        st.markdown("""
        <div class="login-logo">
            <img src="https://img.icons8.com/ios-filled/50/lock-2.png"/>
            <span style="font-size:1.6em;font-weight:700;">LangGraph MCP by 九天Hector</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="login-tip">请输入用户名和密码以进入系统。</div>', unsafe_allow_html=True)

        # 表单
        with st.form("login_form"):
            username = st.text_input("用户名", placeholder="输入您的用户名")
            password = st.text_input("密码", type="password", placeholder="输入您的密码")
            submit_button = st.form_submit_button("登录")

            if submit_button:
                expected_username = os.environ.get("USER_ID")
                expected_password = os.environ.get("USER_PASSWORD")
                if username == expected_username and password == expected_password:
                    st.session_state.authenticated = True
                    st.success("✅ 登录成功！请稍候自动进入主页...")
                else:
                    st.error("❌ 用户名或密码不正确，请重试。")

        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


st.sidebar.divider()  

# 已登录页面标题和描述
st.title("💬 LangGraph MCP by 九天Hector")
st.markdown("✨ 可以自定义接入和使用MCP工具的ReAct代理")


# 设置系统提示词
SYSTEM_PROMPT = """<ROLE>
You are a smart agent with an ability to use tools. 
You will be given a question and you will use the tools to answer the question.
Pick the most relevant tool to answer the question. 
If you are failed to answer the question, try different tools to get context.
Your answer should be very polite and professional.
</ROLE>

----

<INSTRUCTIONS>
Step 1: Analyze the question
- Analyze user's question and final goal.
- If the user's question is consist of multiple sub-questions, split them into smaller sub-questions.

Step 2: Pick the most relevant tool
- Pick the most relevant tool to answer the question.
- If you are failed to answer the question, try different tools to get context.

Step 3: Answer the question
- Answer the question in the same language as the question.
- Your answer should be very polite and professional.

Step 4: Provide the source of the answer(if applicable)
- If you've used the tool, provide the source of the answer.
- Valid sources are either a website(URL) or a document(PDF, etc).

Guidelines:
- If you've used the tool, your answer should be based on the tool's output(tool's output is more important than your own knowledge).
- If you've used the tool, and the source is valid URL, provide the source(URL) of the answer.
- Skip providing the source if the source is not URL.
- Answer in the same language as the question.
- Answer should be concise and to the point.
- Avoid response your output with any other information than the answer and the source.  
</INSTRUCTIONS>

----

<OUTPUT_FORMAT>
(concise answer to the question)

**Source**(if applicable)
- (source1: valid URL)
- (source2: valid URL)
- ...
</OUTPUT_FORMAT>
"""


# 初始化会话状态
if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = False  
    st.session_state.agent = None  
    st.session_state.history = []  
    st.session_state.mcp_client = None  
    st.session_state.timeout_seconds = (
        120  
    )
    st.session_state.selected_model = (
        "deepseek-chat"   # 默认的模型是 DeepSeek v3
    )
    st.session_state.recursion_limit = 100  # 递归调用限制，默认100

if "thread_id" not in st.session_state:
    st.session_state.thread_id = random_uuid()


# --- 工具函数定义 ---
async def cleanup_mcp_client():
    """
    Safely terminates the existing MCP client.

    Properly releases resources if an existing client exists.
    """
    if "mcp_client" in st.session_state and st.session_state.mcp_client is not None:
        try:
            # 新版本不需要手动调用__aexit__，直接设置为None即可
            st.session_state.mcp_client = None
        except Exception as e:
            import traceback
            st.error(f"清理MCP客户端时出错: {str(e)}")
            st.session_state.mcp_client = None

def print_message():
    """
    Displays chat history on the screen.

    Distinguishes between user and assistant messages on the screen,
    and displays tool call information within the assistant message container.
    """
    i = 0
    while i < len(st.session_state.history):
        message = st.session_state.history[i]
        if message["role"] == "user":
            st.chat_message("user", avatar="🧑‍💻").markdown(message["content"])
            i += 1
        elif message["role"] == "assistant":
            # 创建助手消息容器
            with st.chat_message("assistant", avatar="🤖"):
                # 显示助手消息内容
                st.markdown(message["content"])

                # 检查下一个消息是否是工具调用信息
                if (
                    i + 1 < len(st.session_state.history)
                    and st.session_state.history[i + 1]["role"] == "assistant_tool"
                ):
                    # 在同一个容器中显示工具调用信息
                    with st.expander("🔧 Tool Call Information", expanded=False):
                        st.markdown(st.session_state.history[i + 1]["content"])
                    i += 2  # 递增2，因为我们处理了两个消息
                else:
                    i += 1  # 递增1，因为我们只处理了一个常规消息
        else:
            # 跳过助手工具消息，因为它们已经在上面处理了
            i += 1


def get_streaming_callback(text_placeholder, tool_placeholder):
    """
    Creates a streaming callback function.

    This function creates a callback function to display responses generated from the LLM in real-time.
    It displays text responses and tool call information in separate areas.

    Args:
        text_placeholder: Streamlit component to display text responses
        tool_placeholder: Streamlit component to display tool call information

    Returns:
        callback_func: Streaming callback function
        accumulated_text: List to store accumulated text responses
        accumulated_tool: List to store accumulated tool call information
    """
    accumulated_text = []
    accumulated_tool = []

    def callback_func(message: dict):
        nonlocal accumulated_text, accumulated_tool
        message_content = message.get("content", None)

        if isinstance(message_content, AIMessageChunk):
            content = message_content.content
            if (
                hasattr(message_content, "tool_calls")
                and message_content.tool_calls
                and len(message_content.tool_calls[0]["name"]) > 0
            ):
                tool_call_info = message_content.tool_calls[0]
                accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                with tool_placeholder.expander(
                    "🔧 Tool Call Information", expanded=True
                ):
                    st.markdown("".join(accumulated_tool))
            # 如果内容是字符串类型
            elif isinstance(content, str):
                accumulated_text.append(content)
                text_placeholder.markdown("".join(accumulated_text))
            # 如果存在无效的工具调用信息
            elif (
                hasattr(message_content, "invalid_tool_calls")
                and message_content.invalid_tool_calls
            ):
                tool_call_info = message_content.invalid_tool_calls[0]
                accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                with tool_placeholder.expander(
                    "🔧 Tool Call Information (Invalid)", expanded=True
                ):
                    st.markdown("".join(accumulated_tool))
            # 如果tool_call_chunks属性存在
            elif (
                hasattr(message_content, "tool_call_chunks")
                and message_content.tool_call_chunks
            ):
                tool_call_chunk = message_content.tool_call_chunks[0]
                accumulated_tool.append(
                    "\n```json\n" + str(tool_call_chunk) + "\n```\n"
                )
                with tool_placeholder.expander(
                    "🔧 Tool Call Information", expanded=True
                ):
                    st.markdown("".join(accumulated_tool))
            # 如果tool_calls存在additional_kwargs中（支持各种模型兼容性）
            elif (
                hasattr(message_content, "additional_kwargs")
                and "tool_calls" in message_content.additional_kwargs
            ):
                tool_call_info = message_content.additional_kwargs["tool_calls"][0]
                accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                with tool_placeholder.expander(
                    "🔧 Tool Call Information", expanded=True
                ):
                    st.markdown("".join(accumulated_tool))
        # 如果消息是工具消息（工具响应）
        elif isinstance(message_content, ToolMessage):
            accumulated_tool.append(
                "\n```json\n" + str(message_content.content) + "\n```\n"
            )
            with tool_placeholder.expander("🔧 Tool Call Information", expanded=True):
                st.markdown("".join(accumulated_tool))
        return None
    return callback_func, accumulated_text, accumulated_tool


async def process_query(query, text_placeholder, tool_placeholder, timeout_seconds=60):
    """
    Processes user questions and generates responses.

    This function passes the user's question to the agent and streams the response in real-time.
    Returns a timeout error if the response is not completed within the specified time.

    Args:
        query: Text of the question entered by the user
        text_placeholder: Streamlit component to display text responses
        tool_placeholder: Streamlit component to display tool call information
        timeout_seconds: Response generation time limit (seconds)

    Returns:
        response: Agent's response object
        final_text: Final text response
        final_tool: Final tool call information
    """
    try:
        if st.session_state.agent:
            streaming_callback, accumulated_text_obj, accumulated_tool_obj = (
                get_streaming_callback(text_placeholder, tool_placeholder)
            )
            try:
                response = await asyncio.wait_for(
                    astream_graph(
                        st.session_state.agent,
                        {"messages": [HumanMessage(content=query)]},
                        callback=streaming_callback,
                        config=RunnableConfig(
                            recursion_limit=st.session_state.recursion_limit,
                            thread_id=st.session_state.thread_id,
                        ),
                    ),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                error_msg = f"⏱️ 请求时间超过 {timeout_seconds} 秒. 请稍后再试."
                return {"error": error_msg}, error_msg, ""

            final_text = "".join(accumulated_text_obj)
            final_tool = "".join(accumulated_tool_obj)
            return response, final_text, final_tool
        else:
            return (
                {"error": "🚫 代理未初始化."},
                "🚫 代理未初始化.",
                "",
            )
    except Exception as e:
        import traceback

        error_msg = f"❌ 查询处理时发生错误: {str(e)}\n{traceback.format_exc()}"
        return {"error": error_msg}, error_msg, ""


async def initialize_session(mcp_config=None):
    """
    Initializes MCP session and agent.

    Args:
        mcp_config: MCP tool configuration information (JSON). Uses default settings if None

    Returns:
        bool: Initialization success status
    """
    with st.spinner("🔄 连接到MCP服务器..."):
        # 首先安全地清理现有的客户端
        await cleanup_mcp_client()

        if mcp_config is None:
            # 从config.json文件加载设置
            mcp_config = load_config_from_json()
        
        try:
            # 使用新的API方式创建客户端
            client = MultiServerMCPClient(mcp_config)
            
            # 方法1: 直接获取工具
            tools = await client.get_tools()
            st.session_state.tool_count = len(tools)
            st.session_state.mcp_client = client

            # 根据选择初始化适当的模型
            selected_model = st.session_state.selected_model

            model = ChatDeepSeek(
                model=selected_model,
                temperature=0.1,

            )
            agent = create_react_agent(
                model,
                tools,
                checkpointer=MemorySaver(),
                prompt=SYSTEM_PROMPT,
            )
            st.session_state.agent = agent
            st.session_state.session_initialized = True
            return True
            
        except Exception as e:
            st.error(f"❌ MCP客户端初始化失败: {str(e)}")
            st.error("请检查MCP服务器配置是否正确")
            return False


# --- Sidebar: 系统设置 ---
with st.sidebar:
    st.subheader("⚙️ 系统设置")

    # 模型选择功能
    # 创建可用模型的列表
    available_models = []

    has_deepseek_key = os.environ.get("DEEPSEEK_API_KEY") is not None
    if has_deepseek_key:
        available_models.extend(["deepseek-chat"])

  
    if not available_models:
        st.warning(
            "⚠️ API key 没有配置. 请在 .env 文件中添加 DEEPSEEK_API_KEY."
        )
        available_models = ["deepseek-chat"]

    # 模型选择下拉框
    previous_model = st.session_state.selected_model
    st.session_state.selected_model = st.selectbox(
        "🤖 选择模型",
        options=available_models,
        index=(
            available_models.index(st.session_state.selected_model)
            if st.session_state.selected_model in available_models
            else 0
        ),
        help="DeepSeek 模型需要 DEEPSEEK_API_KEY 作为环境变量.",
    )
    if (
        previous_model != st.session_state.selected_model
        and st.session_state.session_initialized
    ):
        st.warning(
            "⚠️ 模型已更改. 点击 '应用设置' 按钮以应用更改."
        )

    # 添加超时设置滑块
    st.session_state.timeout_seconds = st.slider(
        "⏱️ 响应生成时间限制 (秒)",
        min_value=60,
        max_value=300,
        value=st.session_state.timeout_seconds,
        step=10,
        help="设置代理生成响应的最大时间. 复杂任务可能需要更多时间.",
    )

    st.session_state.recursion_limit = st.slider(
        "⏱️ 递归调用限制 (次数)",
        min_value=10,
        max_value=200,
        value=st.session_state.recursion_limit,
        step=10,
        help="设置递归调用限制. 设置过高的值可能会导致内存问题.",
    )

    st.divider() 

    # 工具设置部分
    st.subheader("🔧 工具设置")

    # 管理扩展器状态
    if "mcp_tools_expander" not in st.session_state:
        st.session_state.mcp_tools_expander = False

    # MCP 工具添加界面
    with st.expander("🧰 添加 MCP 工具", expanded=st.session_state.mcp_tools_expander):
        # 从config.json文件加载设置
        loaded_config = load_config_from_json()
        default_config_text = json.dumps(loaded_config, indent=2, ensure_ascii=False)
        
        # 根据现有的mcp_config_text创建pending config
        if "pending_mcp_config" not in st.session_state:
            try:
                st.session_state.pending_mcp_config = loaded_config
            except Exception as e:
                st.error(f"Failed to set initial pending config: {e}")

        # 添加单个工具的UI
        st.subheader("添加工具(JSON格式)")
        st.markdown(
            """
        请插入 **一个工具** 在 JSON 格式中.
        ⚠️ **重要**: JSON 必须用花括号 (`{}`) 包裹.
        """
        )
        # 提供更清晰的示例
        example_json = {
            "github": {
                "command": "npx",
                "args": [
                    "-y",
                    "@smithery/cli@latest",
                    "run",
                    "@smithery-ai/github",
                    "--config",
                    '{"githubPersonalAccessToken":"your_token_here"}',
                ],
                "transport": "stdio",
            }
        }

        default_text = json.dumps(example_json, indent=2, ensure_ascii=False)

        new_tool_json = st.text_area(
            "Tool JSON",
            default_text,
            height=250,
        )

        # 添加按钮
        if st.button(
            "Add Tool",
            type="primary",
            key="add_tool_button",
            use_container_width=True,
        ):
            try:
                # 验证输入
                if not new_tool_json.strip().startswith(
                    "{"
                ) or not new_tool_json.strip().endswith("}"):
                    st.error("JSON must start and end with curly braces ({}).")
                    st.markdown('Correct format: `{ "tool_name": { ... } }`')
                else:
                    # 解析 JSON
                    parsed_tool = json.loads(new_tool_json)

                    # 检查是否在mcpServers格式中，并相应处理
                    if "mcpServers" in parsed_tool:
                        # 将mcpServers的内容移动到顶层
                        parsed_tool = parsed_tool["mcpServers"]
                        st.info(
                            "'mcpServers' format detected. Converting automatically."
                        )

                    # 检查输入的工具数量
                    if len(parsed_tool) == 0:
                        st.error("Please enter at least one tool.")
                    else:
                        # 处理所有工具
                        success_tools = []
                        for tool_name, tool_config in parsed_tool.items():
                            # 检查URL字段并设置transport
                            if "url" in tool_config:
                                # 如果URL存在，则设置transport为"sse"
                                tool_config["transport"] = "sse"
                                st.info(
                                    f"URL detected in '{tool_name}' tool, setting transport to 'sse'."
                                )
                            elif "transport" not in tool_config:
                                # 如果URL不存在且transport未指定，则设置默认值"stdio"
                                tool_config["transport"] = "stdio"

                            # 检查必需字段
                            if (
                                "command" not in tool_config
                                and "url" not in tool_config
                            ):
                                st.error(
                                    f"'{tool_name}' tool configuration requires either 'command' or 'url' field."
                                )
                            elif "command" in tool_config and "args" not in tool_config:
                                st.error(
                                    f"'{tool_name}' tool configuration requires 'args' field."
                                )
                            elif "command" in tool_config and not isinstance(
                                tool_config["args"], list
                            ):
                                st.error(
                                    f"'args' field in '{tool_name}' tool must be an array ([]) format."
                                )
                            else:
                                # 将工具添加到pending_mcp_config
                                st.session_state.pending_mcp_config[tool_name] = (
                                    tool_config
                                )
                                success_tools.append(tool_name)

                        # 成功消息
                        if success_tools:
                            if len(success_tools) == 1:
                                st.success(
                                    f"{success_tools[0]} 工具已添加. 点击 '应用设置' 按钮以应用."
                                )
                            else:
                                tool_names = ", ".join(success_tools)
                                st.success(
                                    f"总共 {len(success_tools)} 个工具 ({tool_names}) 已添加. 点击 '应用设置' 按钮以应用."
                                )
                            # 添加工具后折叠扩展器
                            st.session_state.mcp_tools_expander = False
                            st.rerun()

            except json.JSONDecodeError as e:
                st.error(f"JSON 解析错误: {e}")
                st.markdown(
                    f"""
                **如何修复**:
                1. 检查您的JSON格式是否正确.
                2. 所有键必须用双引号包裹 (").
                3. 字符串值也必须用双引号包裹 (").
                4. 在字符串中使用双引号时，必须转义 (\").
                """
                )
            except Exception as e:
                st.error(f"发生错误: {e}")

    # 显示已注册的工具列表并添加删除按钮
    with st.expander("📋 已注册的工具列表", expanded=True):
        try:
            pending_config = st.session_state.pending_mcp_config
        except Exception as e:
            st.error("不是一个有效的MCP工具配置.")
        else:
            # 遍历pending config中的键（工具名称）
            for tool_name in list(pending_config.keys()):
                col1, col2 = st.columns([8, 2])
                col1.markdown(f"- **{tool_name}**")
                if col2.button("Delete", key=f"delete_{tool_name}"):
                    # 从pending config中删除工具（不立即应用）
                    del st.session_state.pending_mcp_config[tool_name]
                    st.success(
                        f"{tool_name} 工具已删除. 点击 '应用设置' 按钮以应用."
                    )

    st.divider() 

# --- Sidebar: 系统信息和操作按钮部分 ---
with st.sidebar:
    st.subheader("📊 系统信息")
    st.write(
        f"🛠️ MCP 工具数量: {st.session_state.get('tool_count', '初始化中...')}"
    )
    selected_model_name = st.session_state.selected_model
    st.write(f"🧠 当前模型: {selected_model_name}")

    # 移动应用设置按钮
    if st.button(
        "应用设置",
        key="apply_button",
        type="primary",
        use_container_width=True,
    ):
     
        apply_status = st.empty()
        with apply_status.container():
            st.warning("🔄 应用更改. 请稍候...")
            progress_bar = st.progress(0)

            # 保存设置
            st.session_state.mcp_config_text = json.dumps(
                st.session_state.pending_mcp_config, indent=2, ensure_ascii=False
            )

            # 将设置保存到config.json文件
            save_result = save_config_to_json(st.session_state.pending_mcp_config)
            if not save_result:
                st.error("❌ 保存设置文件失败.")
            
            progress_bar.progress(15)

            # 准备会话初始化
            st.session_state.session_initialized = False
            st.session_state.agent = None

            # 更新进度
            progress_bar.progress(30)

            # 运行初始化
            success = st.session_state.event_loop.run_until_complete(
                initialize_session(st.session_state.pending_mcp_config)
            )

            # 更新进度
            progress_bar.progress(100)

            if success:
                st.success("✅ 新设置已应用.")
                # 折叠工具添加扩展器
                if "mcp_tools_expander" in st.session_state:
                    st.session_state.mcp_tools_expander = False
            else:
                st.error("❌ Failed to apply settings.")

        # 刷新页面
        st.rerun()

    st.divider() 

    # 操作按钮部分
    st.subheader("🔄 操作")

    # 重置对话按钮
    if st.button("重置对话", use_container_width=True, type="primary"):
        # 重置thread_id
        st.session_state.thread_id = random_uuid()

        # 重置对话历史
        st.session_state.history = []

        # 通知消息
        st.success("✅ Conversation has been reset.")

        # 刷新页面
        st.rerun()

    # 显示注销按钮（仅当登录功能启用时）
    if use_login and st.session_state.authenticated:
        st.divider() 
        if st.button("注销", use_container_width=True, type="secondary"):
            st.session_state.authenticated = False
            st.success("✅ 已注销.")
            st.rerun()

# --- Initialize default session (if not initialized) ---
if not st.session_state.session_initialized:
    st.info(
        "MCP 服务器和代理未初始化. 请点击左侧边栏的 '应用设置' 按钮以初始化."
    )


# --- 打印对话历史 ---
print_message()

# --- 用户输入和处理 ---
user_query = st.chat_input("💬 Enter your question")
if user_query:
    if st.session_state.session_initialized:
        st.chat_message("user", avatar="🧑‍💻").markdown(user_query)
        with st.chat_message("assistant", avatar="🤖"):
            tool_placeholder = st.empty()
            text_placeholder = st.empty()
            resp, final_text, final_tool = (
                st.session_state.event_loop.run_until_complete(
                    process_query(
                        user_query,
                        text_placeholder,
                        tool_placeholder,
                        st.session_state.timeout_seconds,
                    )
                )
            )
        if "error" in resp:
            st.error(resp["error"])
        else:
            st.session_state.history.append({"role": "user", "content": user_query})
            st.session_state.history.append(
                {"role": "assistant", "content": final_text}
            )
            if final_tool.strip():
                st.session_state.history.append(
                    {"role": "assistant_tool", "content": final_tool}
                )
            st.rerun()
    else:
        st.warning(
            "⚠️ MCP 服务器和代理未初始化. 请点击左侧边栏的 '应用设置' 按钮以初始化."
        )
