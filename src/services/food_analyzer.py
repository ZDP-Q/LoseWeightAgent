import base64
import logging
from typing import Any, Union

from openai import AsyncOpenAI

from ..prompt_manager import PromptManager
from ..schemas import FoodAnalysisResult
from ..utils import parse_llm_json

logger = logging.getLogger("loseweight.agent.food_analyzer")


class FoodAnalyzer:
    def __init__(
        self,
        client: AsyncOpenAI,
        prompt_manager: PromptManager,
        model: str = "qwen3.5-plus",
        stream: bool = False,
    ):
        self.client = client
        self.prompt_manager = prompt_manager
        self.model = model
        self.stream = stream

    async def analyze_food_image(
        self, image_path: str
    ) -> Union[FoodAnalysisResult, dict[str, Any], None]:
        """从文件路径读取图片并识别食物。"""
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        return await self._analyze(base64_image)

    async def analyze_food_image_bytes(
        self, image_bytes: bytes
    ) -> Union[FoodAnalysisResult, dict[str, Any], None]:
        """从字节数据识别食物（供后端 API 使用）。"""
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        return await self._analyze(base64_image)

    async def _analyze(
        self, base64_image: str
    ) -> Union[FoodAnalysisResult, dict[str, Any], None]:
        """内部分析方法，接受 base64 编码的图片。"""
        prompt = self.prompt_manager.render("food_analyzer.j2")

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                temperature=0.1,
                stream=False,  # 食物分析始终非流式
            )

            content = response.choices[0].message.content
            if not content or not content.strip():
                logger.warning("LLM 返回空内容，食物识别失败")
                return None

            return parse_llm_json(content, model=FoodAnalysisResult)

        except Exception as e:
            logger.error("食物识别 API 调用失败: %s", e)
            return None
