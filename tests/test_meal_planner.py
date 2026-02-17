import asyncio
import json
from src.agent import LoseWeightAgent

async def test_meal_plan():
    agent = LoseWeightAgent()
    print("--- 测试餐食规划 (异步) ---")
    
    # 准备测试数据
    agent.register_user("test_user", 75, 175, 30, "male", "moderate")
    agent.add_ingredient("test_user", "鸡蛋")
    agent.add_ingredient("test_user", "燕麦")
    agent.add_ingredient("test_user", "牛肉")
    
    plan = await agent.plan_daily_meals("test_user")
    print("今日餐食规划:")
    if plan:
        # 使用 model_dump() 转换为字典后再 dump
        print(json.dumps(plan.model_dump(), indent=2, ensure_ascii=False))
    else:
        print("规划生成失败")

if __name__ == "__main__":
    asyncio.run(test_meal_plan())
