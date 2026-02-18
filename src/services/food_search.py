"""食物卡路里向量检索服务。

支持通过文本或图片输入，在 Milvus 中检索 USDA 食物营养数据。
"""

import logging

from ..schemas import FoodNutritionSearchResult
from .embedding_service import EmbeddingService
from .milvus_manager import MilvusManager

logger = logging.getLogger("loseweight.food_search")


class FoodSearchService:
    """食物卡路里检索服务，基于 Milvus + DashScope 多模态向量。"""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        milvus_manager: MilvusManager,
    ):
        self.embedding = embedding_service
        self.milvus = milvus_manager

    def search_by_text(
        self, query: str, limit: int = 10
    ) -> list[FoodNutritionSearchResult]:
        """通过文本搜索食物。"""
        logger.info("文本检索: query=%s, limit=%d", query, limit)
        query_vector = self.embedding.embed_text(query)
        return self._search(query_vector, limit)

    def search_by_image(
        self,
        image_data: bytes,
        limit: int = 10,
        image_format: str = "jpeg",
    ) -> list[FoodNutritionSearchResult]:
        """通过图片搜索食物（利用多模态向量）。"""
        logger.info("图片检索: image_size=%d bytes, limit=%d", len(image_data), limit)
        query_vector = self.embedding.embed_image(image_data, image_format)
        return self._search(query_vector, limit)

    def _search(
        self, query_vector: list[float], limit: int
    ) -> list[FoodNutritionSearchResult]:
        """通用向量搜索逻辑。"""
        hits = self.milvus.search(query_vector, limit=limit)

        results = []
        for hit in hits:
            results.append(
                FoodNutritionSearchResult(
                    fdc_id=hit.get("fdc_id", 0),
                    description=hit.get("description", ""),
                    food_category=hit.get("food_category", ""),
                    calories_per_100g=hit.get("calories_per_100g"),
                    protein_per_100g=hit.get("protein_per_100g"),
                    fat_per_100g=hit.get("fat_per_100g"),
                    carbs_per_100g=hit.get("carbs_per_100g"),
                    similarity=hit.get("distance", 0.0),
                )
            )

        return results
