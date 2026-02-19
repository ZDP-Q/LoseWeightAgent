import asyncio
import sys
from pathlib import Path

# 添加后端根目录以便导入
backend_root = Path(__file__).parent.parent.parent
sys.path.append(str(backend_root))

from LoseWeightAgent.src.agent import LoseWeightAgent  # noqa: E402
from src.core.config import get_settings  # noqa: E402
from src.core.database import engine  # noqa: E402
from sqlmodel import Session  # noqa: E402

async def test_guidance():
    settings = get_settings()
    agent = LoseWeightAgent(
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        model=settings.llm.model,
        session_factory=lambda: Session(engine),
    )
    print("--- 测试减重指导 (工具调用版) ---")
    
    # 确保用户存在 (使用真实数据库)
    username = "test_user_guidance"
    agent.register_user(username, 75, 175, 30, "male", "moderate")
    
    # 模拟一个带工具调用的问题
    question = "我今天吃了 100g 炸鸡，帮我记一下。另外，100g 炸鸡有多少热量？"
    
    print(f"用户提问: {question}")
    # 使用 get_guidance_direct，它支持工具调用但返回单一文本
    guidance = await agent.get_guidance_direct(question, user_info=f"用户姓名: {username}")
    
    print(f"教练回复:\n{guidance}")

if __name__ == "__main__":
    asyncio.run(test_guidance())
