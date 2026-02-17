from openai import AsyncOpenAI
from typing import List, Union, Dict, Any
from ..prompt_manager import PromptManager
from ..utils import parse_llm_json
from ..schemas import DailyMealPlan


class MealPlanner:
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

    async def plan_meals(
        self, ingredients: List[str], tdee: float, goal: str = "lose_weight"
    ) -> Union[DailyMealPlan, Dict[str, Any], None]:
        target_calories = tdee - 500 if goal == "lose_weight" else tdee

        prompt = self.prompt_manager.render(
            "meal_planner.j2",
            ingredients=ingredients,
            target_calories=target_calories,
            goal=goal,
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个只输出 JSON 的减脂餐规划专家。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            stream=self.stream,
        )

        if self.stream:
            return {"info": "Streaming mode enabled, response not parsed as JSON"}

        return parse_llm_json(response.choices[0].message.content, model=DailyMealPlan)
