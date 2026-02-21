from openai import OpenAI, AsyncOpenAI


class LLMFactory:
    """LLM 客户端工厂，支持多模型、多厂商客户端创建。"""

    @staticmethod
    def create_client(api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1") -> OpenAI:
        """创建一个同步 OpenAI 客户端。"""
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=120.0,
        )

    @staticmethod
    def create_async_client(api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1") -> AsyncOpenAI:
        """创建一个异步 OpenAI 客户端。"""
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=120.0,
        )

    # 保留旧方法兼容（如果其他地方在用）
    @classmethod
    def configure(cls, api_key: str, base_url: str = "") -> None:
        pass

    @staticmethod
    def create_legacy_async_client(provider: str = "qwen") -> AsyncOpenAI:
        return AsyncOpenAI()
