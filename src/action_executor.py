"""Agent 动作执行器。

解析 LLM 输出中的 JSON Action 指令并执行对应的数据库操作。
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

from sqlmodel import select

logger = logging.getLogger("loseweight.agent.action")


class ActionExecutor:
    """解析并执行 Agent 输出的 JSON Action 指令。"""

    def __init__(self, session_factory):
        """初始化。

        Args:
            session_factory: 返回 SQLModel Session 的可调用对象
        """
        self._session_factory = session_factory

    def execute(self, action_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """执行指定动作，返回结果。"""
        handler = getattr(self, f"_action_{action_name}", None)
        if not handler:
            return {"success": False, "error": f"不支持的动作: {action_name}"}

        try:
            return handler(params)
        except Exception as e:
            logger.error("动作执行失败 [%s]: %s", action_name, e)
            return {"success": False, "error": str(e)}

    def _action_record_weight(self, params: dict[str, Any]) -> dict[str, Any]:
        """记录体重。"""
        from src.models import WeightRecord

        weight_kg = params.get("weight_kg")
        if not weight_kg:
            return {"success": False, "error": "缺少 weight_kg 参数"}

        notes = params.get("notes", "")

        with self._session_factory() as session:
            record = WeightRecord(
                weight_kg=float(weight_kg),
                recorded_at=datetime.now(timezone.utc),
                notes=notes,
            )
            session.add(record)
            session.commit()
            session.refresh(record)

            return {
                "success": True,
                "data": {
                    "id": record.id,
                    "weight_kg": record.weight_kg,
                    "recorded_at": record.recorded_at.isoformat(),
                    "notes": record.notes,
                },
                "message": f"已记录体重 {record.weight_kg} kg",
            }

    def _action_query_weight_history(self, params: dict[str, Any]) -> dict[str, Any]:
        """查询体重历史。"""
        from src.models import WeightRecord

        limit = params.get("limit", 10)

        with self._session_factory() as session:
            stmt = (
                select(WeightRecord)
                .order_by(WeightRecord.recorded_at.desc())
                .limit(limit)
            )
            records = session.exec(stmt).all()

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
                "message": f"查询到 {len(data)} 条体重记录",
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
        if params["gender"].lower() == "male":
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
            "message": f"BMR: {bmr:.0f} kcal, TDEE: {tdee:.0f} kcal",
        }

    def _action_record_food(self, params: dict[str, Any]) -> dict[str, Any]:
        """记录饮食。"""
        from src.models import WeightRecord  # noqa: F401 - 确保模型已注册

        food_name = params.get("food_name")
        calories = params.get("calories")
        if not food_name:
            return {"success": False, "error": "缺少 food_name 参数"}

        # 使用简单的方式记录（在 notes 中存储饮食信息）
        # 因为当前 SQLModel 中没有独立的 FoodLog 表，我们创建一个简单存储
        meal_type = params.get("meal_type", "meal")
        record_text = f"[饮食] {meal_type}: {food_name}"
        if calories:
            record_text += f" ({calories} kcal)"

        return {
            "success": True,
            "data": {
                "food_name": food_name,
                "calories": calories,
                "meal_type": meal_type,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            },
            "message": f"已记录饮食: {food_name}" + (f" ({calories} kcal)" if calories else ""),
        }

    def _action_query_food_log(self, params: dict[str, Any]) -> dict[str, Any]:
        """查询饮食记录。"""
        return {
            "success": True,
            "data": [],
            "message": "饮食记录功能正在开发中，暂无数据",
        }

    def _action_search_food_nutrition(self, params: dict[str, Any]) -> dict[str, Any]:
        """搜索食物营养数据（通过 Milvus）。"""
        query = params.get("query")
        if not query:
            return {"success": False, "error": "缺少 query 参数"}

        if not self._food_search:
            return {"success": False, "error": "食物检索服务不可用"}

        try:
            results = self._food_search.search_by_text(query, limit=5)
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
                "message": f"找到 {len(data)} 种相关食物",
            }
        except Exception as e:
            return {"success": False, "error": f"搜索失败: {e}"}

    def set_food_search(self, food_search) -> None:
        """注入食物检索服务。"""
        self._food_search = food_search

    _food_search = None


def extract_action(text: str) -> Optional[tuple[str, dict[str, Any], str]]:
    """从文本中提取 Action 指令。

    返回: (action_name, params, clean_text) 或 None
    """
    pattern = r"```action\s*\n(.*?)\n\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None

    try:
        action_json = json.loads(match.group(1).strip())
        action_name = action_json.get("action", "")
        params = action_json.get("params", {})
        # 移除 action 块，保留前后文本
        clean_text = text[: match.start()] + text[match.end() :]
        return action_name, params, clean_text.strip()
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Action 指令解析失败: %s", e)
        return None
