import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any, Optional, Union

from openai import AsyncOpenAI

from .action_executor import ActionExecutor
from src.models import User
from src.repositories.user_repository import UserRepository
from src.repositories.food_recognition_repository import FoodRecognitionRepository
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

MEMORY_RECENT_MESSAGES = 12
MEMORY_RECENT_CHAR_BUDGET = 7000
MEMORY_MAX_FACT_ITEMS = 30
MEMORY_MAX_FACT_CHARS = 1400
MEMORY_MAX_TIMELINE_ITEMS = 20
MEMORY_MAX_TIMELINE_CHARS = 1800


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
        self._session_factory = session_factory

        # 废弃直接的 DBManager，统一使用 session_factory + repositories
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
            self._action_executor.set_meal_planner(self.meal_planner)  # 注入食谱规划器
            if self.food_search:
                self._action_executor.set_food_search(self.food_search)

    # ------------------------------------------------------------------
    # 历史记录处理
    # ------------------------------------------------------------------

    def _truncate_text(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return f"{text[: limit - 1]}…"

    def _normalize_history(self, history: list[dict]) -> list[dict]:
        normalized: list[dict] = []
        for item in history:
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "") or "").strip()
            if role not in {"user", "assistant", "tool"}:
                continue
            if not content:
                continue
            normalized.append({"role": role, "content": content})
        return normalized

    def _extract_memory_facts(self, history: list[dict]) -> str:
        categories: dict[str, list[str]] = {
            "目标与阶段": [],
            "饮食偏好": [],
            "饮食限制": [],
            "作息与运动": [],
            "关键健康信息": [],
        }

        def add_fact(category: str, sentence: str):
            sentence = self._truncate_text(sentence.strip(), 80)
            if len(sentence) < 3:
                return
            if sentence in categories[category]:
                return
            categories[category].append(sentence)

        for msg in history:
            if msg["role"] != "user":
                continue
            content = msg["content"].replace("\r\n", "\n")
            sentences = re.split(r"[。！？\n]", content)
            for raw in sentences:
                sentence = raw.strip()
                if not sentence:
                    continue

                if any(
                    k in sentence
                    for k in ["目标", "减到", "减重", "体重", "kg", "公斤"]
                ):
                    add_fact("目标与阶段", sentence)
                if any(k in sentence for k in ["喜欢", "爱吃", "偏好", "口味", "常吃"]):
                    add_fact("饮食偏好", sentence)
                if any(
                    k in sentence
                    for k in ["过敏", "忌口", "不吃", "不能吃", "戒", "乳糖不耐"]
                ):
                    add_fact("饮食限制", sentence)
                if any(
                    k in sentence
                    for k in ["每天", "每周", "运动", "步数", "作息", "睡眠"]
                ):
                    add_fact("作息与运动", sentence)
                if any(
                    k in sentence
                    for k in ["血糖", "血压", "脂肪肝", "甲状腺", "痛风", "医生", "药"]
                ):
                    add_fact("关键健康信息", sentence)

        lines: list[str] = []
        item_count = 0
        for title, facts in categories.items():
            if not facts:
                continue
            lines.append(f"- {title}:")
            for fact in facts:
                lines.append(f"  - {fact}")
                item_count += 1
                if item_count >= MEMORY_MAX_FACT_ITEMS:
                    break
            if item_count >= MEMORY_MAX_FACT_ITEMS:
                break

        packed = "\n".join(lines)
        return self._truncate_text(packed, MEMORY_MAX_FACT_CHARS)

    def _build_timeline_digest(self, older_history: list[dict]) -> str:
        lines: list[str] = []
        for msg in older_history:
            role = "用户" if msg["role"] == "user" else "助手"
            text = self._truncate_text(msg["content"].replace("\n", " ").strip(), 60)
            if not text:
                continue
            lines.append(f"- {role}: {text}")
            if len(lines) >= MEMORY_MAX_TIMELINE_ITEMS:
                break
        packed = "\n".join(lines)
        return self._truncate_text(packed, MEMORY_MAX_TIMELINE_CHARS)

    def _compress_memory(self, history: list[dict]) -> tuple[str, list[dict]]:
        normalized = self._normalize_history(history)
        if not normalized:
            return "", []

        recent = normalized[-MEMORY_RECENT_MESSAGES:]
        while (
            sum(len(m["content"]) for m in recent) > MEMORY_RECENT_CHAR_BUDGET
            and len(recent) > 2
        ):
            recent.pop(0)

        older = normalized[: -len(recent)] if len(normalized) > len(recent) else []
        facts = self._extract_memory_facts(normalized)
        timeline = self._build_timeline_digest(older)

        sections: list[str] = []
        if facts:
            sections.append("### 长期记忆（事实）\n" + facts)
        if timeline:
            sections.append("### 历史轨迹（压缩）\n" + timeline)

        memory_context = "\n\n".join(sections)
        return memory_context, recent

    # ------------------------------------------------------------------
    # 流式聊天（支持 Tool Calling + RAG）
    # ------------------------------------------------------------------

    async def chat_stream(
        self,
        message: str,
        user_info: str = "",
        history: Optional[list[dict]] = None,
        user_id: Optional[int] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """流式聊天方法，支持工具调用和 RAG。"""
        memory_context, processed_history = self._compress_memory(history or [])
        system_prompt = self.prompt_manager.render(
            "chat_agent.j2", user_info=user_info, memory_context=memory_context
        )

        messages = [
            {"role": "system", "content": system_prompt},
        ]
        messages.extend(processed_history)
        messages.append({"role": "user", "content": message})

        logger.info(
            "开始流式对话请求: model=%s, User=%s, 历史条数=%s",
            self.model,
            user_id,
            len(processed_history),
        )

        # 最大迭代次数，防止死循环
        max_iterations = 5
        iteration = 0
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        while iteration < max_iterations:
            iteration += 1

            try:
                # 阿里百炼部分模型支持开启深度思考
                # 使用 extra_body 传递 enable_thinking
                stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    stream=True,
                    stream_options={"include_usage": True},
                    extra_body={"enable_thinking": True} if "qwen" in self.model.lower() else None,
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
                    yield {
                        "event": "usage",
                        "data": {
                            "prompt_tokens": u.prompt_tokens,
                            "completion_tokens": u.completion_tokens,
                            "total_tokens": u.total_tokens,
                            "accumulated": total_usage,
                        },
                    }

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # 1. 处理思考内容 (Reasoning Content)
                # 阿里百炼/DeepSeek 等模型在开启思考模式后会返回此字段
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    yield {"event": "thought", "data": delta.reasoning_content}

                # 2. 处理最终回复文本
                if delta.content:
                    content_to_send = delta.content
                    
                    # 增强型流式格式保障：
                    # 检查当前 delta 是否包含标题起始符（#），且 full_content 末尾没有足够的换行
                    # 我们检查 delta 的开头（忽略开头的空格）是否为 #
                    stripped_delta = content_to_send.lstrip()
                    if stripped_delta.startswith("#") and full_content:
                        # 如果前面没有任何换行，补两个换行；如果只有一个换行，补一个换行
                        if not full_content.endswith("\n"):
                            content_to_send = "\n\n" + content_to_send
                        elif not full_content.endswith("\n\n"):
                            content_to_send = "\n" + content_to_send
                    
                    full_content += content_to_send
                    yield {"event": "text", "data": content_to_send}

                # 处理工具调用
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        if len(tool_calls) <= tc_delta.index:
                            tool_calls.append(
                                {
                                    "id": tc_delta.id,
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            )

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
            assistant_msg = {
                "role": "assistant",
                "content": full_content or None,
                "tool_calls": tool_calls,
            }
            messages.append(assistant_msg)

            # 执行工具调用
            any_tool_executed = False
            for tc in tool_calls:
                action_name = tc["function"]["name"]
                try:
                    params = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    params = {}

                logger.info(
                    f"执行工具: {action_name} (User: {user_id}), 参数: {params}"
                )

                if self._action_executor:
                    exec_result = await self._action_executor.execute(
                        action_name, params, user_id=user_id
                    )
                    exec_result["action"] = action_name

                    # 发送 action_result 事件给前端
                    yield {"event": "action_result", "data": exec_result}

                    # 将工具执行结果添加回对话，供 RAG 使用
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": action_name,
                            "content": json.dumps(exec_result, ensure_ascii=False),
                        }
                    )
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
        self,
        question: str,
        user_info: str = "",
        history: Optional[list[dict]] = None,
        user_id: Optional[int] = None,
    ) -> str:
        """直接获取指导建议（支持工具调用和 RAG）。"""
        memory_context, processed_history = self._compress_memory(history or [])
        system_prompt = self.prompt_manager.render(
            "chat_agent.j2", user_info=user_info, memory_context=memory_context
        )

        messages = [
            {"role": "system", "content": system_prompt},
        ]
        messages.extend(processed_history)
        messages.append({"role": "user", "content": question})

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
                    result = await self._action_executor.execute(
                        action_name, params, user_id=user_id
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": action_name,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )

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
        if not self._session_factory:
            raise RuntimeError("Session factory 未配置")

        tdee = TDEECalculator.calculate_tdee(
            weight, height, age, gender, activity_level
        )

        with self._session_factory() as session:
            repo = UserRepository(session)
            user = repo.get_user_by_name(username)
            if user:
                repo.update_user(
                    user,
                    {
                        "initial_weight_kg": weight,
                        "height_cm": height,
                        "age": age,
                        "gender": gender,
                        "activity_level": activity_level,
                        "tdee": tdee,
                    },
                )
            else:
                user = User(
                    name=username,
                    initial_weight_kg=weight,
                    target_weight_kg=weight,
                    height_cm=height,
                    age=age,
                    gender=gender,
                    activity_level=activity_level,
                    tdee=tdee,
                )
                repo.create_user(user)

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
        if not self._session_factory:
            return None
        with self._session_factory() as session:
            repo = UserRepository(session)
            user = repo.get_user_by_name(username)
            return user.tdee if user else None

    def add_ingredient(self, username: str, ingredient_name: str) -> bool:
        if not self._session_factory:
            return False
        with self._session_factory() as session:
            repo = UserRepository(session)
            user = repo.get_user_by_name(username)
            if user:
                repo.add_ingredient(user.id, ingredient_name)
                return True
        return False

    def get_ingredients(self, username: str) -> list[str]:
        if not self._session_factory:
            return []
        with self._session_factory() as session:
            repo = UserRepository(session)
            user = repo.get_user_by_name(username)
            if user:
                ingredients = repo.get_ingredients(user.id)
                return [i.name for i in ingredients]
        return []

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
                logger.debug(f"文件识别并发路 {i + 1} 成功")
            else:
                logger.warning(f"文件识别并发路 {i + 1} 失败: {r}")

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

        if self._session_factory:
            try:
                with self._session_factory() as session:
                    user_repo = UserRepository(session)
                    recognition_repo = FoodRecognitionRepository(session)

                    user = user_repo.get_user_by_name(username) if username else None

                    recognition_repo.create_recognition_log(
                        user_id=user.id if user else None,
                        image_path=image_path,
                        food_name=final_response.final_food_name,
                        calories=final_response.final_estimated_calories,
                        verification_status="已取平均值",
                        reason=f"基于 {len(results)} 次独立并发识别结果计算平均值",
                    )
            except Exception as e:
                logger.error("数据库存储失败: %s", e)

        return final_response

    async def analyze_food_bytes(
        self, image_bytes: bytes
    ) -> Union[FoodRecognitionResponse, dict[str, str]]:
        """接收图片字节数据进行食物识别（三路并发模式）。"""
        logger.info("开始处理图片识别请求 (三路并发模式)...")

        # 显式创建协程对象
        coros = [
            self.food_analyzer.analyze_food_image_bytes(image_bytes) for _ in range(3)
        ]

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
                logger.debug(
                    f"并发路 {i + 1} 识别成功: {r.food_name}, {r.calories} kcal"
                )
            elif isinstance(r, Exception):
                logger.warning(f"并发路 {i + 1} 识别出错: {r}")
            else:
                logger.warning(f"并发路 {i + 1} 识别返回未知结果: {r}")

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

        logger.info(
            f"三路并发聚合完成: {final_response.final_food_name}, 平均 {final_response.final_estimated_calories} kcal"
        )
        return final_response

    # ------------------------------------------------------------------
    # 知识问答（保留旧方法兼容）
    # ------------------------------------------------------------------

    async def get_guidance(self, username: str, question: str) -> str:
        user_info = ""
        if self._session_factory:
            with self._session_factory() as session:
                repo = UserRepository(session)
                user = repo.get_user_by_name(username)
                if user:
                    user_info = (
                        f"用户信息：体重{user.initial_weight_kg}kg, 身高{user.height_cm}cm, "
                        f"年龄{user.age}, TDEE{user.tdee:.0f}kcal。"
                    )

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
