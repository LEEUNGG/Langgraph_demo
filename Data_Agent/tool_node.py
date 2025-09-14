#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data Agent 工具节点模块
   提供各类工具函数，包括数据库查询、数据提取、代码执行和可视化等功能
"""
import os
import json
import pymysql
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv 
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# 加载环境变量
def load_environment():
    """加载环境变量配置"""
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"),override=True)

# 初始化加载环境变量
load_environment()


# ==============================================================================
# 数据库查询工具
# ==============================================================================
class SQLQuerySchema(BaseModel):
    sql_query: str = Field(description="""
当用户需要进行数据库查询工作时，请调用该函数。
该函数用于在指定MySQL服务器上运行一段SQL代码，完成数据查询相关工作，
并且当前函数是使用pymysql连接MySQL数据库。
本函数只负责运行SQL代码并进行数据查询，若要进行数据提取，则使用另一个extract_data函数。
""")

@tool(args_schema=SQLQuerySchema)
def sql_query_tool(sql_query: str):
    """
    当用户需要进行数据库查询工作时，请调用该函数。
    该函数用于在指定MySQL服务器上运行一段SQL代码，完成数据查询相关工作，
    并且当前函数是使用pymysql连接MySQL数据库。
    
    :param sql_query: 字符串形式的SQL查询语句，用于执行对MySQL中telco_db数据库中各张表进行查询，并获得各表中的各类相关信息
    :return: sql_query在MySQL中的运行结果（JSON格式字符串）
    """
    # 从环境变量获取数据库连接信息
    host = os.getenv('HOST')
    user = os.getenv('USER')
    mysql_pw = os.getenv('MYSQL_PW')
    db = os.getenv('DB_NAME')
    port = os.getenv('PORT')
    
    # 创建数据库连接
    connection = pymysql.connect(
        host=host,
        user=user,
        passwd=mysql_pw,
        db=db,
        port=int(port),
        charset='utf8'
    )
    
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql_query)
            results = cursor.fetchall()
    finally:
        connection.close()

    # 将结果以 JSON 字符串形式返回
    return json.dumps(results, ensure_ascii=False)


# ==============================================================================
# 数据提取工具
# ==============================================================================
class ExtractQuerySchema(BaseModel):
    sql_query: str = Field(description="用于从 MySQL 提取数据的 SQL 查询语句。")
    df_name: str = Field(description="指定用于保存结果的 pandas 变量名称（字符串形式）。")

@tool(args_schema=ExtractQuerySchema)
def extract_data(sql_query: str, df_name: str) -> str:
    """
    用于在MySQL数据库中提取一张表到当前Python环境中，注意，本函数只负责数据表的提取，
    并不负责数据查询，若需要在MySQL中进行数据查询，请使用sql_query_tool函数。
    
    :param sql_query: 字符串形式的SQL查询语句，用于提取MySQL中的某张表。
    :param df_name: 将MySQL数据库中提取的表格进行本地保存时的变量名，以字符串形式表示。
    :return: 表格读取和保存结果（成功/失败信息）
    """
    print("正在调用 extract_data 工具运行 SQL 查询...")
    
    load_dotenv(override=True)
    host = os.getenv('HOST')
    user = os.getenv('USER')
    mysql_pw = os.getenv('MYSQL_PW')
    db = os.getenv('DB_NAME')
    port = os.getenv('PORT')

    # 创建数据库连接
    connection = pymysql.connect(
        host=host,
        user=user,
        passwd=mysql_pw,
        db=db,
        port=int(port),
        charset='utf8'
    )

    try:
        # 执行 SQL 并保存为全局变量
        df = pd.read_sql(sql_query, connection)
        globals()[df_name] = df
        return f"✅ 成功创建 pandas 对象 `{df_name}`，包含从 MySQL 提取的数据。"
    except Exception as e:
        return f"❌ 执行失败：{e}"
    finally:
        connection.close()


# ==============================================================================
# Python代码执行工具
# ==============================================================================
class PythonCodeInput(BaseModel):
    py_code: str = Field(description="一段合法的 Python 代码字符串，例如 '2 + 2' 或 'x = 3\ny = x * 2'")

@tool(args_schema=PythonCodeInput)
def python_inter(py_code):
    """
    当用户需要编写Python程序并执行时，请调用该函数。
    该函数可以执行一段Python代码并返回最终结果，需要注意，本函数只能执行非绘图类的代码，若是绘图相关代码，则需要调用fig_inter函数运行。
    
    :param py_code: 要执行的Python代码字符串
    :return: 代码执行结果
    """
    g = globals()
    try:
        # 尝试如果是表达式，则返回表达式运行结果
        return str(eval(py_code, g))
    # 若报错，则先测试是否是对相同变量重复赋值
    except Exception as e:
        global_vars_before = set(g.keys())
        try:
            exec(py_code, g)
        except Exception as e:
            return f"代码执行时报错{e}"
        global_vars_after = set(g.keys())
        new_vars = global_vars_after - global_vars_before
        # 若存在新变量
        if new_vars:
            result = {var: g[var] for var in new_vars}
            return str(result)
        else:
            return "已经顺利执行代码"


# ==============================================================================
# 可视化绘图工具
# ==============================================================================
class FigCodeInput(BaseModel):
    py_code: str = Field(description="要执行的 Python 绘图代码，必须使用 matplotlib/seaborn 创建图像并赋值给变量")
    fname: str = Field(description="图像对象的变量名，例如 'fig'，用于从代码中提取并保存为图片")

@tool(args_schema=FigCodeInput)
def fig_inter(py_code: str, fname: str) -> str:
    """
    当用户需要使用 Python 进行可视化绘图任务时，请调用该函数。
    该函数会执行用户提供的 Python 绘图代码，并自动将生成的图像对象保存为图片文件并展示。
    
    :param py_code: 要执行的Python绘图代码
    :param fname: 图像对象的变量名
    :return: 图像生成和保存结果（成功/失败信息）
    """
    print("正在调用fig_inter工具运行Python代码...")

    current_backend = matplotlib.get_backend()
    matplotlib.use('Agg')

    local_vars = {"plt": plt, "pd": pd, "sns": sns}
    
    # 设置图像保存路径（当前项目的相对路径）
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 向上两级目录（LangGraph_Demo目录）
    base_dir = os.path.abspath(os.path.join(current_dir, '..', 'Part 2.LangGraph+MCP智能体开发', '前端项目源码', 'agent-chat-ui', 'public'))
    images_dir = os.path.join(base_dir, "images")
    os.makedirs(images_dir, exist_ok=True)  # 自动创建 images 文件夹（如不存在）

    try:
        g = globals()
        exec(py_code, g, local_vars)
        g.update(local_vars)

        fig = local_vars.get(fname, None)
        if fig:
            image_filename = f"{fname}.png"
            abs_path = os.path.join(images_dir, image_filename)  # 绝对路径
            rel_path = os.path.join("images", image_filename)    # 返回相对路径（给前端用）

            fig.savefig(abs_path, bbox_inches='tight')
            return f"✅ 图片已保存，路径为: {rel_path}"
        else:
            return "⚠️ 图像对象未找到，请确认变量名正确并为 matplotlib 图对象。"
    except Exception as e:
        return f"❌ 执行失败：{e}"
    finally:
        plt.close('all')
        matplotlib.use(current_backend)

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]