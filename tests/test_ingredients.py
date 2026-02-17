import asyncio
from src.agent import LoseWeightAgent

async def test_ingredients():
    agent = LoseWeightAgent()
    print("--- 测试食材管理 ---")
    
    # 确保用户存在
    agent.register_user("test_user", 75, 175, 30, "male", "moderate")
    
    agent.add_ingredient("test_user", "鸡胸肉")
    agent.add_ingredient("test_user", "西兰花")
    agent.add_ingredient("test_user", "糙米")
    
    ingredients = agent.get_ingredients("test_user")
    print(f"用户食材列表: {ingredients}")

if __name__ == "__main__":
    asyncio.run(test_ingredients())
