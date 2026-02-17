import asyncio
from src.agent import LoseWeightAgent

async def test_guidance():
    agent = LoseWeightAgent()
    print("--- 测试减重指导 (异步) ---")
    
    # 确保用户存在
    agent.register_user("test_user", 75, 175, 30, "male", "moderate")
    
    question = "我今天特别想吃炸鸡，怎么办？"
    guidance = await agent.get_guidance("test_user", question)
    
    print(f"用户提问: {question}")
    print(f"教练回复:\n{guidance}")

if __name__ == "__main__":
    asyncio.run(test_guidance())
