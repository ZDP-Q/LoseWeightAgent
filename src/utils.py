import json
import logging
import re
from typing import Any, Optional, TypeVar, Union

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger("loseweight.agent.utils")


def parse_llm_json(
    text: str, model: Optional[type[T]] = None
) -> Union[dict[str, Any], T, None]:
    """清洗并解析 LLM 返回的 JSON 字符串。

    如果提供了 Pydantic 模型，将尝试进行校验并返回模型实例。
    """
    if not text or not text.strip():
        logger.warning("LLM 返回空内容，无法解析 JSON")
        return None

    try:
        # 提取 JSON 内容（处理包含 Markdown 代码块的情况）
        json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = text.strip()

        # 尝试清理可能存在的前后杂乱字符
        start_idx = json_str.find("{")
        end_idx = json_str.rfind("}")
        if start_idx != -1 and end_idx != -1:
            json_str = json_str[start_idx : end_idx + 1]

        data = json.loads(json_str)

        if model:
            try:
                return model.model_validate(data)
            except ValidationError as e:
                logger.warning("Pydantic 校验失败: %s", e)
                return data  # 校验失败但解析成功，返回原始字典供回退
        return data
    except Exception as e:
        logger.error("JSON 解析失败: %s | 原始内容: %s", e, text[:200])
        return None
