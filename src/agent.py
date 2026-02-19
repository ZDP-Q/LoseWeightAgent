import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any, Optional, Union

from openai import AsyncOpenAI

from .action_executor import ActionExecutor
from .database.db_manager import DBManager
from src.models import FoodRecognition, Ingredient, User
from .llm.factory import LLMFactory
from .prompt_manager import PromptManager
from .schemas import (
    DailyMealPlan,
    FoodAnalysisResult,
    FoodRecognitionResponse,
    UserSchema,
)
from .services.embedding_service import EmbeddingService
from .services.food_analyzer import FoodAnalyzer
from .services.food_search import FoodSearchService
from .services.meal_planner import MealPlanner
from .services.milvus_manager import MilvusManager
from .services.tdee import TDEECalculator
from .tools import TOOLS

logger = logging.getLogger("loseweight.agent")


class LoseWeightAgent:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        database_url: str = "",
        provider: str = "qwen",
        model: str = "qwen3.5-plus",
        stream: bool = True,
        # Milvus + Embedding 配置
        milvus_host: str = "127.0.0.1",
        milvus_port: int = 19530,
        milvus_collection: str = "usda_foods",
        embedding_model: str = "qwen3-vl-embedding",
        embedding_dimension: int = 1024,
        # 后端数据库 session 工厂（SQLModel）
        session_factory=None,
    ):
        # 注入 LLM 配置
        LLMFactory.configure(api_key=api_key, base_url=base_url)

        self.client: AsyncOpenAI = LLMFactory.create_async_client(provider)
        self.model = model
        self.stream = stream
        self.prompt_manager = PromptManager()

        # 初始化数据库（仅在提供 database_url 时）
        if database_url:
            self.db = DBManager(database_url)
            self.db.init_db()
        else:
            self.db = None

        self.meal_planner = MealPlanner(
            self.client, self.prompt_manager, model=self.model, stream=self.stream
        )
        self.food_analyzer = FoodAnalyzer(
            self.client,
            self.prompt_manager,
            model=self.model,
            stream=self.stream,
        )

        # 初始化食物检索服务
        try:
            self._embedding_service = EmbeddingService(
                api_key=api_key,
                model=embedding_model,
                dimension=embedding_dimension,
            )
            self._milvus_manager = MilvusManager(
                host=milvus_host,
                port=milvus_port,
                collection_name=milvus_collection,
                vector_dim=embedding_dimension,
            )
            self.food_search = FoodSearchService(
                embedding_service=self._embedding_service,
                milvus_manager=self._milvus_manager,
            )
        except Exception:
            self.food_search = None

        # 初始化动作执行器
        self._action_executor: Optional[ActionExecutor] = None
        if session_factory:
            self._action_executor = ActionExecutor(session_factory)
            if self.food_search:
                self._action_executor.set_food_search(self.food_search)

    # ------------------------------------------------------------------
    # 流式聊天（支持 Tool Calling + RAG）
    # ------------------------------------------------------------------

    async def chat_stream(
        self, message: str, user_info: str = ""
    ) -> AsyncGenerator[dict[str, Any], None]:
        """流式聊天方法，支持工具调用和 RAG。"""
        system_prompt = self.prompt_manager.render(
            "chat_agent.j2", user_info=user_info
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ]

        logger.info(f"开始流式对话请求: model={self.model}")
        
        # 最大迭代次数，防止死循环
        max_iterations = 5
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            
            try:
                stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    stream=True,
                )
            except Exception as e:
                logger.error(f"LLM 请求失败: {e}")
                yield {"event": "error", "data": str(e)}
                return

            full_content = ""
            tool_calls = []
            
            async for chunk in stream:
                if not chunk.choices:
                    continue
                
                delta = chunk.choices[0].delta
                
                # 处理文本内容
                if delta.content:
                    full_content += delta.content
                    yield {"event": "text", "data": delta.content}
                
                # 处理工具调用
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        if len(tool_calls) <= tc_delta.index:
                            tool_calls.append({
                                "id": tc_delta.id,
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            })
                        
                        tc = tool_calls[tc_delta.index]
                        if tc_delta.id:
                            tc["id"] = tc_delta.id
                        if tc_delta.function.name:
                            tc["function"]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc["function"]["arguments"] += tc_delta.function.arguments

            # 如果没有工具调用，结束循环
            if not tool_calls:
                break

            # 将助手消息添加到历史记录
            assistant_msg = {"role": "assistant", "content": full_content or None, "tool_calls": tool_calls}
            messages.append(assistant_msg)

            # 执行工具调用
            any_tool_executed = False
            for tc in tool_calls:
                action_name = tc["function"]["name"]
                try:
                    params = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    params = {}
                
                logger.info(f"执行工具: {action_name}, 参数: {params}")
                
                if self._action_executor:
                    exec_result = self._action_executor.execute(action_name, params)
                    exec_result["action"] = action_name
                    
                    # 发送 action_result 事件给前端
                    yield {"event": "action_result", "data": exec_result}
                    
                    # 将工具执行结果添加回对话，供 RAG 使用
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": action_name,
                        "content": json.dumps(exec_result, ensure_ascii=False)
                    })
                    any_tool_executed = True
            
            if not any_tool_executed:
                break
                
            # 继续下一次迭代，让 LLM 根据工具结果生成回复（RAG）
            logger.info("继续迭代以处理工具结果")

        logger.info(f"流式对话结束，共迭代 {iteration} 次")
        yield {"event": "done", "data": ""}

    # ------------------------------------------------------------------
    # 非流式聊天（向下兼容）
    # ------------------------------------------------------------------

    async def get_guidance_direct(
        self, question: str, user_info: str = ""
    ) -> str:
        """直接获取指导建议（支持工具调用和 RAG）。"""
        system_prompt = self.prompt_manager.render(
            "chat_agent.j2", user_info=user_info
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )

            msg = response.choices[0].message
            if not msg.tool_calls:
                return msg.content or ""

            # 处理工具调用
            messages.append(msg)
            for tc in msg.tool_calls:
                action_name = tc.function.name
                try:
                    params = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    params = {}
                
                if self._action_executor:
                    result = self._action_executor.execute(action_name, params)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": action_name,
                        "content": json.dumps(result, ensure_ascii=False)
                    })
        
        return "对话迭代次数过多，请稍后再试。"

    # ------------------------------------------------------------------
    # 用户管理
    # ------------------------------------------------------------------

    def register_user(
        self,
        username: str,
        weight: float,
        height: float,
        age: int,
        gender: str,
        activity_level: str,
    ) -> UserSchema:
        if not self.db:
            raise RuntimeError("数据库未配置")
        session = self.db.get_session()
        tdee = TDEECalculator.calculate_tdee(
            weight, height, age, gender, activity_level
        )

        # 使用 name 字段替代 username
        user = session.query(User).filter_by(name=username).first()
        if user:
            user.initial_weight_kg = weight  # Mapping to backend model fields
            user.height_cm = height
            user.age = age
            user.gender = gender
            user.activity_level = activity_level
            user.tdee = tdee
        else:
            user = User(
                name=username,
                initial_weight_kg=weight,
                target_weight_kg=weight, # Default to current weight if not specified
                height_cm=height,
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
        if not self.db:
            return None
        session = self.db.get_session()
        user = session.query(User).filter_by(name=username).first()
        tdee = user.tdee if user else None
        session.close()
        return tdee

    def add_ingredient(self, username: str, ingredient_name: str) -> bool:
        if not self.db:
            return False
        session = self.db.get_session()
        user = session.query(User).filter_by(name=username).first()
        if user:
            ingredient = Ingredient(name=ingredient_name, user_id=user.id)
            session.add(ingredient)
            session.commit()
            session.close()
            return True
        session.close()
        return False

    def get_ingredients(self, username: str) -> list[str]:
        if not self.db:
            return []
        session = self.db.get_session()
        user = session.query(User).filter_by(name=username).first()
        ingredients = []
        if user:
            ingredients = [
                i.name
                for i in session.query(Ingredient).filter_by(user_id=user.id).all()
            ]
        session.close()
        return ingredients

    # ------------------------------------------------------------------
    # 餐食规划
    # ------------------------------------------------------------------

    async def plan_daily_meals(self, username: str) -> Optional[DailyMealPlan]:
        tdee = self.get_user_tdee(username)
        if not tdee:
            return None
        ingredients = self.get_ingredients(username)
        return await self.meal_planner.plan_meals(ingredients, tdee)

    async def plan_meals_direct(
        self,
        ingredients: list[str],
        target_calories: int = 1800,
        goal: str = "lose_weight",
    ) -> Optional[DailyMealPlan]:
        """直接规划餐食（无需用户名，由后端 API 调用）。"""
        return await self.meal_planner.plan_meals(
            ingredients, target_calories, goal=goal
        )

    # ------------------------------------------------------------------
    # 食物识别
    # ------------------------------------------------------------------

    async def analyze_food(
        self, image_path: str, username: Optional[str] = None
    ) -> Union[FoodRecognitionResponse, dict[str, str]]:
        tasks = [self.food_analyzer.analyze_food_image(image_path) for _ in range(3)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[FoodAnalysisResult] = []
        for r in responses:
            if isinstance(r, FoodAnalysisResult):
                results.append(r)
            elif isinstance(r, Exception):
                logger.warning("并发识别中的一项失败: %s", r)

        if not results:
            return {"error": "所有并发识别尝试均失败"}

        total_calories = sum(r.calories for r in results)
        avg_calories = int(total_calories / len(results))
        final_food_name = results[0].food_name if results else "未知食物"

        final_response = FoodRecognitionResponse(
            final_food_name=final_food_name,
            final_estimated_calories=avg_calories,
            raw_data=results,
            timestamp=datetime.now().isoformat(),
        )

        if self.db:
            try:
                session = self.db.get_session()
                user = (
                    session.query(User).filter_by(name=username).first()
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
                logger.error("数据库存储失败: %s", e)

        return final_response

    async def analyze_food_bytes(
        self, image_bytes: bytes
    ) -> Union[FoodRecognitionResponse, dict[str, str]]:
        """接收图片字节数据进行食物识别（三路并发）。"""
        tasks = [
            self.food_analyzer.analyze_food_image_bytes(image_bytes) for _ in range(3)
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[FoodAnalysisResult] = []
        for r in responses:
            if isinstance(r, FoodAnalysisResult):
                results.append(r)
            elif isinstance(r, Exception):
                logger.warning("并发识别中的一项失败: %s", r)

        if not results:
            return {"error": "所有并发识别尝试均失败"}

        total_calories = sum(r.calories for r in results)
        avg_calories = int(total_calories / len(results))
        final_food_name = results[0].food_name if results else "未知食物"

        return FoodRecognitionResponse(
            final_food_name=final_food_name,
            final_estimated_calories=avg_calories,
            raw_data=results,
            timestamp=datetime.now().isoformat(),
        )

    # ------------------------------------------------------------------
    # 知识问答（保留旧方法兼容）
    # ------------------------------------------------------------------

    async def get_guidance(self, username: str, question: str) -> str:
        user_info = ""
        if self.db:
            session = self.db.get_session()
            user = session.query(User).filter_by(name=username).first()
            if user:
                user_info = (
                    f"用户信息：体重{user.weight}kg, 身高{user.height}cm, "
                    f"年龄{user.age}, TDEE{user.tdee:.0f}kcal。"
                )
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
