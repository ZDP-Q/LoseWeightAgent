"""Agent 动作执行器。

执行来自 LLM Tool Calling 的指令并执行对应的数据库操作。
"""

import logging
from datetime import datetime, timezone
from typing import Any

from sqlmodel import select
from src.models import WeightRecord, FoodLog
from src.repositories.user_repository import UserRepository
from src.repositories.weight_repository import WeightRepository
from src.repositories.food_log_repository import FoodLogRepository

logger = logging.getLogger("loseweight.agent.action")


class ActionExecutor:
    """执行 Agent 输出的工具调用指令。"""

    def __init__(self, session_factory):
        """初始化。

        Args:
            session_factory: 返回 SQLModel Session 的可调用对象
        """
        self._session_factory = session_factory
        self._food_search = None
        self._meal_planner = None

    async def execute(self, action_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """异步执行指定动作，返回结果。"""
        handler = getattr(self, f"_action_{action_name}", None)
        if not handler:
            return {"success": False, "error": f"不支持的动作: {action_name}"}

        try:
            # 兼容同步和异步处理函数
            import inspect
            if inspect.iscoroutinefunction(handler):
                return await handler(params)
            else:
                return handler(params)
        except Exception as e:
            logger.error("动作执行失败 [%s]: %s", action_name, e)
            return {"success": False, "error": str(e)}

    def _action_record_weight(self, params: dict[str, Any]) -> dict[str, Any]:
        """记录体重。"""
        weight_kg = params.get("weight_kg")
        if weight_kg is None:
            return {"success": False, "error": "缺少 weight_kg 参数"}

        notes = params.get("notes", "")

        with self._session_factory() as session:
            repo = WeightRepository(session)
            record = WeightRecord(
                weight_kg=float(weight_kg),
                recorded_at=datetime.now(timezone.utc),
                notes=notes,
            )
            record = repo.create_record(record)

            return {
                "success": True,
                "data": {
                    "id": record.id,
                    "weight_kg": record.weight_kg,
                    "recorded_at": record.recorded_at.isoformat(),
                    "notes": record.notes,
                },
                "message": f"已成功记录体重 {record.weight_kg} kg",
            }

    def _action_query_weight_history(self, params: dict[str, Any]) -> dict[str, Any]:
        """查询体重历史。"""
        limit = params.get("limit", 10)

        with self._session_factory() as session:
            repo = WeightRepository(session)
            records = repo.get_records(limit=limit)

            data = [
                {
                    "id": r.id,
                    "weight_kg": r.weight_kg,
                    "recorded_at": r.recorded_at.isoformat(),
                    "notes": r.notes,
                }
                for r in records
            ]

            return {
                "success": True,
                "data": data,
                "message": f"为您查询到最近 {len(data)} 条体重记录",
            }

    def _action_calculate_tdee(self, params: dict[str, Any]) -> dict[str, Any]:
        """计算 TDEE。"""
        from .services.tdee import TDEECalculator

        required = ["weight_kg", "height_cm", "age", "gender", "activity_level"]
        for key in required:
            if key not in params:
                return {"success": False, "error": f"缺少参数: {key}"}

        tdee = TDEECalculator.calculate_tdee(
            weight=float(params["weight_kg"]),
            height=float(params["height_cm"]),
            age=int(params["age"]),
            gender=params["gender"],
            activity_level=params["activity_level"],
        )

        # BMR 计算 (Mifflin-St Jeor)
        weight = float(params["weight_kg"])
        height = float(params["height_cm"])
        age = int(params["age"])
        if str(params["gender"]).lower() == "male":
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161

        return {
            "success": True,
            "data": {
                "bmr": round(bmr, 1),
                "tdee": round(tdee, 1),
                "weight_kg": weight,
                "height_cm": height,
                "age": age,
                "gender": params["gender"],
                "activity_level": params["activity_level"],
            },
            "message": f"计算完成：您的 BMR 为 {bmr:.0f} kcal，TDEE 为 {tdee:.0f} kcal",
        }

    def _action_record_food(self, params: dict[str, Any]) -> dict[str, Any]:
        """记录饮食。"""
        food_name = params.get("food_name")
        calories = params.get("calories")
        
        if not food_name:
            return {"success": False, "error": "缺少 food_name 参数"}
        
        if calories is None:
            return {"success": False, "error": "缺少 calories 参数"}

        with self._session_factory() as session:
            user_repo = UserRepository(session)
            food_log_repo = FoodLogRepository(session)
            
            user = user_repo.get_first_user()
            user_id = user.id if user else None

            log = food_log_repo.create_log(
                user_id=user_id,
                food_name=food_name,
                calories=float(calories)
            )

            return {
                "success": True,
                "data": {
                    "id": log.id,
                    "food_name": log.food_name,
                    "calories": log.calories,
                    "recorded_at": log.timestamp.isoformat(),
                },
                "message": f"已成功记录：{log.food_name}，热量为 {log.calories} kcal",
            }

    def _action_query_food_log(self, params: dict[str, Any]) -> dict[str, Any]:
        """查询饮食记录。"""
        limit = params.get("limit", 10)

        with self._session_factory() as session:
            stmt = (
                select(FoodLog)
                .order_by(FoodLog.timestamp.desc())
                .limit(limit)
            )
            records = session.exec(stmt).all()

            data = [
                {
                    "id": r.id,
                    "food_name": r.food_name,
                    "calories": r.calories,
                    "recorded_at": r.timestamp.isoformat(),
                }
                for r in records
            ]

            return {
                "success": True,
                "data": data,
                "message": f"为您查询到最近 {len(data)} 条饮食记录",
            }

    async def _action_search_food_nutrition(self, params: dict[str, Any]) -> dict[str, Any]:
        """搜索食物营养数据（通过 Milvus）。"""
        query = params.get("query")
        if not query:
            return {"success": False, "error": "缺少 query 参数"}

        if not self._food_search:
            return {"success": False, "error": "食物检索服务不可用"}

        try:
            results = self._food_search.search_by_text(query, limit=params.get("limit", 5))
            data = [
                {
                    "description": r.description,
                    "category": r.food_category,
                    "calories_per_100g": r.calories_per_100g,
                    "protein_per_100g": r.protein_per_100g,
                    "fat_per_100g": r.fat_per_100g,
                    "carbs_per_100g": r.carbs_per_100g,
                }
                for r in results
            ]
            return {
                "success": True,
                "data": data,
                "message": f"搜索完成，为您找到 {len(data)} 种相关食物",
            }
        except Exception as e:
            return {"success": False, "error": f"搜索过程出错: {e}"}

    async def _action_generate_meal_plan(self, params: dict[str, Any]) -> dict[str, Any]:
        """生成详细食谱。"""
        if not self._meal_planner:
            return {"success": False, "error": "食谱规划服务不可用"}

        ingredients = params.get("ingredients", [])
        calorie_goal = params.get("calorie_goal", 1800)
        preferences = params.get("preferences", "清淡")

        try:
            result = await self._meal_planner.plan_meals(
                ingredients=ingredients,
                target_calories=float(calorie_goal),
                goal="lose_weight"
            )
            
            if not result:
                return {"success": False, "error": "食谱规划生成结果为空"}

            return {
                "success": True,
                "data": result.model_dump(),
                "message": f"已为您成功规划包含 {', '.join(ingredients)} 的 {preferences} 减脂食谱",
            }
        except Exception as e:
            logger.error(f"食谱规划动作执行失败: {e}")
            return {"success": False, "error": f"规划出错: {e}"}

    def _action_query_user_profile(self, params: dict[str, Any]) -> dict[str, Any]:
        """查询用户档案。"""
        with self._session_factory() as session:
            user_repo = UserRepository(session)
            user = user_repo.get_first_user()
            if not user:
                return {"success": False, "error": "未找到用户信息，请先完善个人资料"}

            return {
                "success": True,
                "data": {
                    "name": user.name,
                    "gender": user.gender,
                    "age": user.age,
                    "height_cm": user.height_cm,
                    "weight_kg": user.initial_weight_kg,
                    "target_weight_kg": user.target_weight_kg,
                    "tdee": user.tdee,
                    "suggested_calories": user.tdee - 500 if user.tdee else 1800,
                },
                "message": f"已获取用户 {user.name} 的健康档案",
            }

    def _action_query_user_ingredients(self, params: dict[str, Any]) -> dict[str, Any]:
        """查询用户库存食材。"""
        with self._session_factory() as session:
            user_repo = UserRepository(session)
            user = user_repo.get_first_user()
            if not user:
                return {"success": False, "error": "未找到用户信息"}

            items = user_repo.get_ingredients(user.id)
            ingredient_list = [i.name for i in items]
            
            return {
                "success": True,
                "data": {
                    "ingredients": ingredient_list
                },
                "message": f"用户当前拥有 {len(ingredient_list)} 种食材: {', '.join(ingredient_list)}" if ingredient_list else "用户当前没有记录库存食材",
            }

    def set_food_search(self, food_search) -> None:
        """注入食物检索服务。"""
        self._food_search = food_search

    def set_meal_planner(self, meal_planner) -> None:
        """注入食谱规划服务。"""
        self._meal_planner = meal_planner
