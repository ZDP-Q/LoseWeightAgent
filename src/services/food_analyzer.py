from openai import AsyncOpenAI
import base64
from typing import Union, Dict, Any
from ..prompt_manager import PromptManager
from ..utils import parse_llm_json
from ..schemas import FoodAnalysisResult


class FoodAnalyzer:
    def __init__(
        self,
        client: AsyncOpenAI,
        prompt_manager: PromptManager,
        model: str = "qwen-vl-plus",  # 视觉模型通常需要 VL 系列
        stream: bool = False,
    ):
        self.client = client
        self.prompt_manager = prompt_manager
        self.model = model
        self.stream = stream

    async def analyze_food_image(self, image_path: str) -> Union[FoodAnalysisResult, Dict[str, Any], None]:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        prompt = self.prompt_manager.render("food_analyzer.j2")

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
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
            stream=self.stream,
        )

        if self.stream:
            return {"info": "Streaming mode enabled, response not parsed as JSON"}

        return parse_llm_json(response.choices[0].message.content, model=FoodAnalysisResult)
