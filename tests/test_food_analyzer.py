import asyncio
import json
from src.agent import LoseWeightAgent
from src.schemas import FoodRecognitionResponse

async def test_food_analysis():
    agent = LoseWeightAgent()
    image_path = "t.jpg"
    print(f"--- 测试图片识别 (3路并发平均值): {image_path} ---")
    
    result = await agent.analyze_food(image_path, username="test_user")
    
    print("识别结果:")
    if isinstance(result, FoodRecognitionResponse):
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    else:
        print(result)

if __name__ == "__main__":
    asyncio.run(test_food_analysis())
