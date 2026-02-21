import asyncio
import json
import sys
from pathlib import Path

# 添加后端根目录以便导入
backend_root = Path(__file__).parent.parent.parent
sys.path.append(str(backend_root))

from LoseWeightAgent.src.agent import LoseWeightAgent  # noqa: E402
from LoseWeightAgent.src.schemas import FoodRecognitionResponse  # noqa: E402
from src.core.config import get_settings  # noqa: E402


async def test_food_analysis():
    settings = get_settings()
    agent = LoseWeightAgent(
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        model=settings.llm.model,
    )
    # 注意：运行此测试需要真实存在的图片 t.jpg
    image_path = "t.jpg"
    print(f"--- 测试图片识别 (3路并发平均值): {image_path} ---")

    if not Path(image_path).exists():
        print(f"⚠️ 跳过测试: 未找到 {image_path}")
        return

    result = await agent.analyze_food(image_path, username="test_user")

    print("识别结果:")
    if isinstance(result, FoodRecognitionResponse):
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    else:
        print(result)


if __name__ == "__main__":
    asyncio.run(test_food_analysis())
