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
    FoodRecognitionResponse,
    UserSchema,
    FoodAnalysisResult,
)
from .services.embedding_service import EmbeddingService
from .services.food_analyzer import FoodAnalyzer
from .services.food_search import FoodSearchService
from .services.meal_planner import MealPlanner
from .services.milvus_manager import MilvusManager
from .services.tdee import TDEECalculator
from .tools import TOOLS

logger = logging.getLogger("loseweight.agent")

# 历史记录压缩阈值（字符数）
MAX_HISTORY_CHARS = 3000

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
            self._action_executor.set_meal_planner(self.meal_planner) # 注入食谱规划器
            if self.food_search:
                self._action_executor.set_food_search(self.food_search)

    # ------------------------------------------------------------------
    # 历史记录处理
    # ------------------------------------------------------------------

    async def _summarize_messages(self, messages: list[dict]) -> str:
        """对对话记录进行摘要压缩。"""
        if not messages:
            return ""
        
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages if m.get("content")])
        
        prompt = f"请简要总结以下对话内容，保留关键的健康信息、饮食偏好和已达成的目标：\n\n{history_text}\n\n摘要："
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"摘要生成失败: {e}")
            return "（摘要生成失败）"

    async def _process_history(self, history: list[dict]) -> list[dict]:
        """处理并压缩历史记录。"""
        if not history:
            return []
        
        total_chars = sum(len(m.get("content", "")) for m in history)
        
        if total_chars <= MAX_HISTORY_CHARS:
            return history
            
        logger.info(f"历史记录过长 ({total_chars} 字符)，正在执行压缩...")
        
        # 保留最近的 4 条消息（大约 2 轮对话）作为上下文，其余的进行摘要
        keep_count = 4
        if len(history) <= keep_count:
            return history
            
        to_summarize = history[:-keep_count]
        to_keep = history[-keep_count:]
        
        summary = await self._summarize_messages(to_summarize)
        
        compressed_history = [
            {"role": "system", "content": f"先前对话摘要：{summary}"}
        ]
        compressed_history.extend(to_keep)
        
        return compressed_history

    # ------------------------------------------------------------------
    # 流式聊天（支持 Tool Calling + RAG）
    # ------------------------------------------------------------------

    async def chat_stream(
        self, message: str, user_info: str = "", history: Optional[list[dict]] = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        """流式聊天方法，支持工具调用和 RAG。"""
        system_prompt = self.prompt_manager.render(
            "chat_agent.j2", user_info=user_info
        )

        # 处理并压缩历史记录
        processed_history = await self._process_history(history or [])

        messages = [
            {"role": "system", "content": system_prompt},
        ]
        messages.extend(processed_history)
        messages.append({"role": "user", "content": message})

        logger.info(f"开始流式对话请求: model={self.model}, 历史长度={len(processed_history)}")
        
        # 最大迭代次数，防止死循环
        max_iterations = 5
        iteration = 0
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        while iteration < max_iterations:
            iteration += 1
            
            try:
                stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    stream=True,
                    stream_options={"include_usage": True} # 开启 Token 统计
                )
            except Exception as e:
                logger.error(f"LLM 请求失败: {e}")
                yield {"event": "error", "data": str(e)}
                return

            full_content = ""
            tool_calls = []
            
            async for chunk in stream:
                # 处理 Token 消耗信息 (通常在最后一个 chunk)
                if hasattr(chunk, "usage") and chunk.usage:
                    u = chunk.usage
                    total_usage["prompt_tokens"] += u.prompt_tokens
                    total_usage["completion_tokens"] += u.completion_tokens
                    total_usage["total_tokens"] += u.total_tokens
                    yield {"event": "usage", "data": {
                        "prompt_tokens": u.prompt_tokens,
                        "completion_tokens": u.completion_tokens,
                        "total_tokens": u.total_tokens,
                        "accumulated": total_usage
                    }}

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
                    exec_result = await self._action_executor.execute(action_name, params)
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
                    result = await self._action_executor.execute(action_name, params)
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
        # 减重目标默认摄入 = TDEE - 500
        target_calories = tdee - 500
        return await self.meal_planner.plan_meals(ingredients, target_calories)

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
        """从文件路径识别食物（三路并发模式）。"""
        logger.info(f"开始从文件识别图片: {image_path} (三路并发)")
        
        tasks = [self.food_analyzer.analyze_food_image(image_path) for _ in range(3)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[FoodAnalysisResult] = []
        for i, r in enumerate(responses):
            if isinstance(r, FoodAnalysisResult):
                results.append(r)
                logger.debug(f"文件识别并发路 {i+1} 成功")
            else:
                logger.warning(f"文件识别并发路 {i+1} 失败: {r}")

        if not results:
            return {"error": "所有并发识别尝试均失败"}

        total_calories = sum(r.calories for r in results)
        avg_calories = int(total_calories / len(results))
        final_food_name = results[0].food_name

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
        """接收图片字节数据进行食物识别（三路并发模式）。"""
        logger.info("开始处理图片识别请求 (三路并发模式)...")
        
        # 显式创建协程对象
        coros = [self.food_analyzer.analyze_food_image_bytes(image_bytes) for _ in range(3)]
        
        try:
            # 确保所有创建的协程都被传入 gather，从而被管理和 await
            responses = await asyncio.gather(*coros, return_exceptions=True)
        except Exception as e:
            logger.error(f"并发调度发生异常: {e}")
            return {"error": f"识别请求调度失败: {str(e)}"}

        results: list[FoodAnalysisResult] = []
        for i, r in enumerate(responses):
            if isinstance(r, FoodAnalysisResult):
                results.append(r)
                logger.debug(f"并发路 {i+1} 识别成功: {r.food_name}, {r.calories} kcal")
            elif isinstance(r, Exception):
                logger.warning(f"并发路 {i+1} 识别出错: {r}")
            else:
                logger.warning(f"并发路 {i+1} 识别返回未知结果: {r}")

        if not results:
            logger.error("所有三路并发识别尝试均已失败")
            return {"error": "食物识别失败，请检查网络或图片内容"}

        total_calories = sum(r.calories for r in results)
        avg_calories = int(total_calories / len(results))
        final_food_name = results[0].food_name

        final_response = FoodRecognitionResponse(
            final_food_name=final_food_name,
            final_estimated_calories=avg_calories,
            raw_data=results,
            timestamp=datetime.now().isoformat(),
        )
        
        logger.info(f"三路并发聚合完成: {final_response.final_food_name}, 平均 {final_response.final_estimated_calories} kcal")
        return final_response

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
                    f"用户信息：体重{user.initial_weight_kg}kg, 身高{user.height_cm}cm, "
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
