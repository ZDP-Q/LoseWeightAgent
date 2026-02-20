import base64
import logging
import io
from typing import Any, Union
from PIL import Image

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

    def _resize_image(self, image_bytes: bytes, max_size: int = 1024) -> bytes:
        """保持比例缩放图片，确保最大边不超过 max_size。"""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            orig_size = len(image_bytes)
            
            # 如果是 RGBA 模式，转换为 RGB
            if img.mode == 'RGBA':
                img = img.convert('RGB')
                
            width, height = img.size
            if width > max_size or height > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                output = io.BytesIO()
                img.save(output, format="JPEG", quality=85)
                resized_bytes = output.getvalue()
                logger.info(f"图片已压缩: {orig_size/1024:.1f}KB -> {len(resized_bytes)/1024:.1f}KB (尺寸: {width}x{height} -> {new_width}x{new_height})")
                return resized_bytes
            
            logger.info(f"图片无需压缩: {orig_size/1024:.1f}KB (尺寸: {width}x{height})")
            return image_bytes
        except Exception as e:
            logger.warning(f"图片处理失败，使用原始数据: {e}")
            return image_bytes

    async def analyze_food_image(
        self, image_path: str
    ) -> Union[FoodAnalysisResult, dict[str, Any], None]:
        """从文件路径读取图片并识别食物。"""
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            
        resized_bytes = self._resize_image(image_bytes)
        base64_image = base64.b64encode(resized_bytes).decode("utf-8")

        return await self._analyze(base64_image)

    async def analyze_food_image_bytes(
        self, image_bytes: bytes
    ) -> Union[FoodAnalysisResult, dict[str, Any], None]:
        """从字节数据识别食物（供后端 API 使用）。"""
        resized_bytes = self._resize_image(image_bytes)
        base64_image = base64.b64encode(resized_bytes).decode("utf-8")
        return await self._analyze(base64_image)

    async def _analyze(
        self, base64_image: str
    ) -> Union[FoodAnalysisResult, dict[str, Any], None]:
        """内部分析方法，接受 base64 编码的图片。"""
        prompt = self.prompt_manager.render("food_analyzer.j2")
        logger.debug(f"开始调用 LLM ({self.model}) 进行食物识别...")

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
            logger.debug(f"LLM 原始回复内容: {content}")
            
            if not content or not content.strip():
                logger.warning("LLM 返回空内容，食物识别失败")
                return None

            return parse_llm_json(content, model=FoodAnalysisResult)

        except Exception as e:
            logger.error("食物识别 API 调用失败: %s", e)
            return None
