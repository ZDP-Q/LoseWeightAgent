from openai import OpenAI, AsyncOpenAI


class LLMFactory:
    """LLM 客户端工厂，通过 configure() 注入配置，不再依赖环境变量。"""

    _api_key: str = ""
    _base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    @classmethod
    def configure(cls, api_key: str, base_url: str = "") -> None:
        """注入 LLM 配置（由后端 config.yaml 提供）。"""
        cls._api_key = api_key
        if base_url:
            cls._base_url = base_url

    @staticmethod
    def create_client(provider: str = "qwen") -> OpenAI:
        if provider == "qwen":
            return OpenAI(
                api_key=LLMFactory._api_key,
                base_url=LLMFactory._base_url,
                timeout=120.0,
            )
        raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def create_async_client(provider: str = "qwen") -> AsyncOpenAI:
        if provider == "qwen":
            return AsyncOpenAI(
                api_key=LLMFactory._api_key,
                base_url=LLMFactory._base_url,
                timeout=120.0,
            )
        raise ValueError(f"Unsupported provider: {provider}")
