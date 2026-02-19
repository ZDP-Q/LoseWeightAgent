import asyncio
import json
import sys
from pathlib import Path

# 添加后端根目录以便导入
backend_root = Path(__file__).parent.parent.parent
sys.path.append(str(backend_root))

from LoseWeightAgent.src.agent import LoseWeightAgent  # noqa: E402
from src.core.config import get_settings  # noqa: E402

async def test_meal_plan():
    settings = get_settings()
    agent = LoseWeightAgent(
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        model=settings.llm.model,
        database_url=settings.database.url
    )
    print("--- 测试餐食规划 ---")
    
    # 准备测试数据
    username = "test_user_meal"
    agent.register_user(username, 75, 175, 30, "male", "moderate")
    agent.add_ingredient(username, "鸡蛋")
    agent.add_ingredient(username, "燕麦")
    agent.add_ingredient(username, "牛肉")
    
    plan = await agent.plan_daily_meals(username)
    print("今日餐食规划:")
    if plan:
        print(json.dumps(plan.model_dump(), indent=2, ensure_ascii=False))
    else:
        print("规划生成失败")

if __name__ == "__main__":
    asyncio.run(test_meal_plan())
