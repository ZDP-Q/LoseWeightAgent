"""DashScope qwen3-vl-embedding 多模态向量化服务。

支持文本和图片的向量化，通过 DashScope HTTP API 调用。
"""

import base64
import logging
from typing import Optional

import dashscope
from http import HTTPStatus

logger = logging.getLogger("loseweight.embedding")


class EmbeddingService:
    """封装 DashScope qwen3-vl-embedding 多模态嵌入模型。"""

    def __init__(
        self,
        api_key: str,
        model: str = "qwen3-vl-embedding",
        dimension: int = 1024,
    ):
        self.model = model
        self.dimension = dimension
        dashscope.api_key = api_key

    def _call_api(
        self, contents: list[dict], dimension: Optional[int] = None
    ) -> list[list[float]]:
        """调用 DashScope MultiModalEmbedding API。"""
        dim = dimension or self.dimension
        resp = dashscope.MultiModalEmbedding.call(
            model=self.model,
            input=contents,
            dimension=dim,
            output_type="dense",
        )

        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(
                f"DashScope embedding 调用失败: "
                f"code={getattr(resp, 'code', '?')}, "
                f"message={getattr(resp, 'message', '?')}"
            )

        # 提取向量
        embeddings = resp.output.get("embeddings", [])
        return [item["embedding"] for item in embeddings]

    def embed_text(self, text: str) -> list[float]:
        """将文本转换为向量。"""
        results = self._call_api([{"text": text}])
        if not results:
            raise RuntimeError("向量化返回空结果")
        return results[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量将文本转换为向量（每次最多 20 条）。"""
        all_embeddings: list[list[float]] = []
        batch_size = 6  # DashScope 限制：总内容元素 ≤ 20，保守设 6

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            contents = [{"text": t} for t in batch]
            embeddings = self._call_api(contents)
            all_embeddings.extend(embeddings)
            logger.debug("向量化批次 %d/%d 完成", i // batch_size + 1,
                         (len(texts) + batch_size - 1) // batch_size)

        return all_embeddings

    def embed_image(self, image_data: bytes, image_format: str = "jpeg") -> list[float]:
        """将图片字节数据转换为向量。"""
        b64 = base64.b64encode(image_data).decode("utf-8")
        data_uri = f"data:image/{image_format};base64,{b64}"
        results = self._call_api([{"image": data_uri}])
        if not results:
            raise RuntimeError("图片向量化返回空结果")
        return results[0]
