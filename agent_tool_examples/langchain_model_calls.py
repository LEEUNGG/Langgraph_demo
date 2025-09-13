# -*- coding: utf-8 -*-
"""
LangChain 模型调用方法示例
专注于展示不同的模型初始化和调用方式
"""

import os
from dotenv import load_dotenv

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构造上层目录的.env文件路径
env_path = os.path.join(current_dir, '..', '.env')
load_dotenv(dotenv_path=env_path)

print("=" * 80)
print("LangChain 模型调用方法示例")
print("=" * 80)

# ===== 方法1: 直接导入特定提供商的模型类 =====
print("\n1. 直接导入特定提供商的模型类")
print("-" * 60)

try:
    # DeepSeek 模型
    from langchain_deepseek.chat_models import ChatDeepSeek
    from langchain.schema import HumanMessage
    
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
    
    # 使用 HumanMessage 提问
    question = "hi"
    print(f"\n  提问: {question}")
    response = deepseek_model_direct.invoke([HumanMessage(content=question)])
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
    from langchain.schema import HumanMessage
    
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
    
    # 使用 HumanMessage 提问
    question = "hi"
    print(f"\n  提问: {question}")
    response = deepseek_model_init_auto.invoke([HumanMessage(content=question)])
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
    
    # 使用 HumanMessage 提问
    print(f"\n  提问: {question}")
    response = deepseek_model_init_explicit.invoke([HumanMessage(content=question)])
    print(f"  响应: {response.content}")
    
except ImportError as e:
    print(f"✗ init_chat_model 导入失败: {e}")
except Exception as e:
    print(f"✗ 初始化或调用模型时出错: {e}")

print("\n" + "=" * 80)
print("模型调用方法总结")
print("=" * 80)
print("1. 直接导入特定提供商模型类:")
print("   - 优点: 可以直接访问提供商特有的功能和参数")
print("   - 缺点: 与特定提供商绑定，更换模型时需要修改更多代码")
print("")
print("2. 使用 init_chat_model 自动识别提供商:")
print("   - 优点: 统一的接口，便于切换不同模型")
print("   - 缺点: 可能无法访问某些提供商特有的高级功能")
print("")
print("3. 使用 init_chat_model 明确指定提供商:")
print("   - 优点: 统一接口的同时避免识别错误")
print("   - 缺点: 仍然可能无法访问某些提供商特有的高级功能")
print("")
print("最佳实践:")
print("- 简单应用: 使用 init_chat_model 简化代码")
print("- 需要高级功能: 直接导入特定提供商的模型类")
print("- 考虑未来扩展性: 优先使用 init_chat_model 统一接口")
print("=" * 80)