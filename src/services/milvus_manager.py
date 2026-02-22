"""Milvus 向量数据库管理模块。

管理 usda_foods 集合的创建、索引、数据插入和搜索。
"""

import logging
import time
from typing import Optional

from pymilvus import (
    MilvusClient,
    DataType,
    CollectionSchema,
    FieldSchema,
)

logger = logging.getLogger("loseweight.milvus")

# 默认集合 Schema
COLLECTION_NAME = "usda_foods"
VECTOR_DIM = 1024


class MilvusManager:
    """Milvus 向量数据库管理器。"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 19530,
        collection_name: str = COLLECTION_NAME,
        vector_dim: int = VECTOR_DIM,
        max_retries: int = 3,
        retry_interval: int = 5,
    ):
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        uri = f"http://{host}:{port}"

        # 尝试连接并增加重试逻辑，解决 "Proxy not ready" 问题
        last_exception = None
        for i in range(max_retries):
            try:
                self.client = MilvusClient(uri=uri)
                # 尝试一个简单的操作来验证连接是否真的可用
                self.client.list_collections()
                logger.info("已成功连接 Milvus: %s", uri)
                return
            except Exception as e:
                last_exception = e
                # 提取精简的错误信息
                error_msg = str(e)
                if "Milvus Proxy is not ready yet" in error_msg:
                    short_error = "Milvus Proxy 正在启动中..."
                else:
                    # 尝试只保留第一行或比较短的描述
                    short_error = error_msg.split('\n')[0][:100]

                if i < max_retries - 1:
                    logger.warning(
                        "Milvus 连接 (第 %d/%d 次重试): %s. %d 秒后再次尝试...",
                        i + 1,
                        max_retries,
                        short_error,
                        retry_interval,
                    )
                    time.sleep(retry_interval)
                else:
                    logger.error("Milvus 连接失败，已达到最大重试次数: %s (错误: %s)", uri, short_error)
                    raise last_exception

    def create_collection(self, drop_if_exists: bool = False) -> None:
        """创建 usda_foods 集合。"""
        if drop_if_exists and self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            logger.info("已删除旧集合: %s", self.collection_name)

        if self.client.has_collection(self.collection_name):
            logger.info("集合已存在: %s", self.collection_name)
            return

        # 定义 Schema
        fields = [
            FieldSchema(
                name="fdc_id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=False,
            ),
            FieldSchema(
                name="description",
                dtype=DataType.VARCHAR,
                max_length=512,
            ),
            FieldSchema(
                name="food_category",
                dtype=DataType.VARCHAR,
                max_length=256,
            ),
            FieldSchema(
                name="calories_per_100g",
                dtype=DataType.FLOAT,
            ),
            FieldSchema(
                name="protein_per_100g",
                dtype=DataType.FLOAT,
            ),
            FieldSchema(
                name="fat_per_100g",
                dtype=DataType.FLOAT,
            ),
            FieldSchema(
                name="carbs_per_100g",
                dtype=DataType.FLOAT,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.vector_dim,
            ),
        ]
        schema = CollectionSchema(fields=fields, description="USDA Foundation Foods")

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
        )
        logger.info("已创建集合: %s", self.collection_name)

        # 创建向量索引
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params,
        )
        logger.info("已创建向量索引 (IVF_FLAT, COSINE)")

    def insert_batch(self, data: list[dict]) -> int:
        """批量插入数据。

        每条数据应包含:
        fdc_id, description, food_category,
        calories_per_100g, protein_per_100g, fat_per_100g, carbs_per_100g,
        embedding
        """
        if not data:
            return 0

        result = self.client.insert(
            collection_name=self.collection_name,
            data=data,
        )
        count = result.get("insert_count", len(data))
        logger.debug("插入 %d 条数据", count)
        return count

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        output_fields: Optional[list[str]] = None,
    ) -> list[dict]:
        """向量相似度搜索。"""
        if output_fields is None:
            output_fields = [
                "fdc_id",
                "description",
                "food_category",
                "calories_per_100g",
                "protein_per_100g",
                "fat_per_100g",
                "carbs_per_100g",
            ]

        # 确保集合已加载
        self.client.load_collection(self.collection_name)

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=limit,
            output_fields=output_fields,
            search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
        )

        # 转换为 list[dict]
        hits = []
        if results and len(results) > 0:
            for hit in results[0]:
                item = hit.get("entity", {})
                item["distance"] = hit.get("distance", 0.0)
                hits.append(item)

        return hits

    def get_collection_stats(self) -> dict:
        """获取集合统计信息。"""
        stats = self.client.get_collection_stats(self.collection_name)
        return stats
