import asyncio
from datetime import datetime
from .llm.factory import LLMFactory
from .database.db_manager import DBManager
from .database.models import User, Ingredient, FoodRecognition
from .services.tdee import TDEECalculator
from .services.meal_planner import MealPlanner
from .services.food_analyzer import FoodAnalyzer
from .prompt_manager import PromptManager
from .schemas import (
    UserSchema,
    DailyMealPlan,
    FoodAnalysisResult,
    FoodRecognitionResponse,
)
from typing import List, Optional, Dict, Union
from openai import AsyncOpenAI


class LoseWeightAgent:
    def __init__(
        self,
        provider: str = "qwen",
        model: str = "qwen3.5-plus",
        stream: bool = False,
    ):
        self.client: AsyncOpenAI = LLMFactory.create_async_client(provider)
        self.model = model
        self.stream = stream
        self.prompt_manager = PromptManager()
        self.db = DBManager()
        self.db.init_db()

        self.meal_planner = MealPlanner(
            self.client, self.prompt_manager, model=self.model, stream=self.stream
        )
        self.food_analyzer = FoodAnalyzer(
            self.client,
            self.prompt_manager,
            model="qwen3.5-plus",  # 统一使用 qwen3.5-plus
            stream=self.stream,
        )

    def register_user(
        self,
        username: str,
        weight: float,
        height: float,
        age: int,
        gender: str,
        activity_level: str,
    ) -> UserSchema:
        session = self.db.get_session()
        tdee = TDEECalculator.calculate_tdee(
            weight, height, age, gender, activity_level
        )

        user = session.query(User).filter_by(username=username).first()
        if user:
            user.weight = weight
            user.height = height
            user.age = age
            user.gender = gender
            user.activity_level = activity_level
            user.tdee = tdee
        else:
            user = User(
                username=username,
                weight=weight,
                height=height,
                age=age,
                gender=gender,
                activity_level=activity_level,
                tdee=tdee,
            )
            session.add(user)

        session.commit()
        session.close()
        return UserSchema(
            username=username,
            weight=weight,
            height=height,
            age=age,
            gender=gender,
            activity_level=activity_level,
            tdee=tdee,
        )

    def get_user_tdee(self, username: str) -> Optional[float]:
        session = self.db.get_session()
        user = session.query(User).filter_by(username=username).first()
        tdee = user.tdee if user else None
        session.close()
        return tdee

    def add_ingredient(self, username: str, ingredient_name: str) -> bool:
        session = self.db.get_session()
        user = session.query(User).filter_by(username=username).first()
        if user:
            ingredient = Ingredient(name=ingredient_name, user_id=user.id)
            session.add(ingredient)
            session.commit()
            session.close()
            return True
        session.close()
        return False

    def get_ingredients(self, username: str) -> List[str]:
        session = self.db.get_session()
        user = session.query(User).filter_by(username=username).first()
        ingredients = []
        if user:
            ingredients = [
                i.name
                for i in session.query(Ingredient).filter_by(user_id=user.id).all()
            ]
        session.close()
        return ingredients

    async def plan_daily_meals(self, username: str) -> Optional[DailyMealPlan]:
        tdee = self.get_user_tdee(username)
        if not tdee:
            return None

        ingredients = self.get_ingredients(username)
        return await self.meal_planner.plan_meals(ingredients, tdee)

    async def analyze_food(
        self, image_path: str, username: Optional[str] = None
    ) -> Union[FoodRecognitionResponse, Dict[str, str]]:
        # 并发执行 3 次识别
        tasks = [self.food_analyzer.analyze_food_image(image_path) for _ in range(3)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        results: List[FoodAnalysisResult] = []
        for r in responses:
            if isinstance(r, FoodAnalysisResult):
                results.append(r)
            elif isinstance(r, Exception):
                print(f"并发识别中的一项失败: {r}")

        if not results:
            return {"error": "所有并发识别尝试均失败"}

        # 计算平均卡路里并获取食物名称
        total_calories = sum(r.calories for r in results)
        avg_calories = int(total_calories / len(results))
        final_food_name = results[0].food_name if results else "未知食物"

        final_response = FoodRecognitionResponse(
            final_food_name=final_food_name,
            final_estimated_calories=avg_calories,
            raw_data=results,
            timestamp=datetime.now().isoformat(),
        )

        try:
            session = self.db.get_session()
            user = (
                session.query(User).filter_by(username=username).first()
                if username
                else None
            )

            recognition_log = FoodRecognition(
                user_id=user.id if user else None,
                image_path=image_path,
                food_name=final_response.final_food_name,
                calories=final_response.final_estimated_calories,
                verification_status="已取平均值",
                reason=f"基于 {len(results)} 次独立并发识别结果计算平均值",
            )
            session.add(recognition_log)
            session.commit()
            session.close()
        except Exception as e:
            print(f"数据库存储失败: {e}")

        return final_response

    async def get_guidance(self, username: str, question: str) -> str:
        session = self.db.get_session()
        user = session.query(User).filter_by(username=username).first()
        user_info = ""
        if user:
            user_info = f"用户信息：体重{user.weight}kg, 身高{user.height}cm, 年龄{user.age}, TDEE{user.tdee:.0f}kcal。"
        session.close()

        prompt = self.prompt_manager.render(
            "guidance.j2", user_info=user_info, question=question
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个充满动力且专业的减重教练。"},
                {"role": "user", "content": prompt},
            ],
            stream=self.stream,
        )

        if self.stream:
            return "Streaming mode enabled"

        return response.choices[0].message.content
