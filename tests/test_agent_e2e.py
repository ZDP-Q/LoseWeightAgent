import asyncio
import sys
from pathlib import Path

# 将后端根目录添加到 sys.path，以便正确导入 src 和 LoseWeightAgent
backend_root = Path(__file__).parent.parent.parent
sys.path.append(str(backend_root))

from LoseWeightAgent.src.agent import LoseWeightAgent  # noqa: E402
from src.core.config import get_settings  # noqa: E402
from src.core.database import engine  # noqa: E402
from sqlmodel import Session  # noqa: E402

async def test_chat_and_tools():
    settings = get_settings()
    
    # 打印配置信息（隐藏 API Key 部分）
    masked_key = f"{settings.llm.api_key[:5]}...{settings.llm.api_key[-5:]}" if settings.llm.api_key else "None"
    print("--- 正在初始化 LoseWeightAgent ---")
    print(f"Model: {settings.llm.model}")
    print(f"Base URL: {settings.llm.base_url}")
    print(f"API Key: {masked_key}")
    print(f"Milvus: {settings.milvus.host}:{settings.milvus.port}")
    print("----------------------------------\n")

    agent = LoseWeightAgent(
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        model=settings.llm.model,
        milvus_host=settings.milvus.host,
        milvus_port=settings.milvus.port,
        milvus_collection=settings.milvus.collection,
        embedding_model=settings.embedding.model,
        embedding_dimension=settings.embedding.dimension,
        session_factory=lambda: Session(engine),
    )

    test_cases = [
        {
            "name": "TDEE 计算测试",
            "message": "帮我算一下我的 TDEE。我是男的，25岁，身高180cm，体重75kg，每周运动3-5次。",
            "expected_tool": "calculate_tdee"
        },
        {
            "name": "食物热量 RAG 测试",
            "message": "100克苹果和100克香蕉哪个热量更高？",
            "expected_tool": "search_food_nutrition"
        },
        {
            "name": "饮食记录测试",
            "message": "我刚刚吃了一个 200 大卡的苹果，帮我记一下。",
            "expected_tool": "record_food"
        },
        {
            "name": "体重记录测试",
            "message": "我今天体重 74.5kg，帮我记一下。",
            "expected_tool": "record_weight"
        },
        {
            "name": "综合查询测试",
            "message": "帮我看看我最近的饮食和体重记录。",
            "expected_tool": ["query_food_log", "query_weight_history"]
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"CASE {i}: {case['name']}")
        print(f"User: {case['message']}")
        
        full_text = ""
        tools_called = []
        
        async for event in agent.chat_stream(case["message"]):
            if event["event"] == "text":
                text = event["data"]
                full_text += text
                print(text, end="", flush=True)
            elif event["event"] == "action_result":
                action = event["data"].get("action")
                tools_called.append(action)
                print(f"\n[Tool Call: {action}] -> Result: {event['data'].get('message', 'Success')}")
            elif event["event"] == "error":
                print(f"\n[ERROR]: {event['data']}")
        
        print("\n\n--- 验证结果 ---")
        expected = case["expected_tool"]
        if isinstance(expected, str):
            expected = [expected]
            
        success = all(tool in tools_called for tool in expected)
        if success:
            print("✅ 成功：正确调用了预期的工具。")
        else:
            print(f"❌ 失败：未能在调用列表中找到预期的工具 {expected}。实际调用: {tools_called}")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    try:
        asyncio.run(test_chat_and_tools())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n程序发生错误: {e}")
        import traceback
        traceback.print_exc()
