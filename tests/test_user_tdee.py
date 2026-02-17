import asyncio
from src.agent import LoseWeightAgent

async def test_tdee():
    agent = LoseWeightAgent()
    print("--- 测试用户注册与 TDEE 计算 ---")
    user_data = agent.register_user(
        username="test_user",
        weight=75.0,
        height=175.0,
        age=30,
        gender="male",
        activity_level="moderate"
    )
    print(f"注册结果: {user_data}")
    
    tdee = agent.get_user_tdee("test_user")
    print(f"查询 TDEE: {tdee} kcal")

if __name__ == "__main__":
    asyncio.run(test_tdee())
