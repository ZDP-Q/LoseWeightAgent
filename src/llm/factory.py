import os
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()


class LLMFactory:
    @staticmethod
    def create_client(provider: str = "qwen") -> OpenAI:
        if provider == "qwen":
            return OpenAI(
                api_key=os.getenv("QWEN_API_KEY"),
                base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            )
        raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def create_async_client(provider: str = "qwen") -> AsyncOpenAI:
        if provider == "qwen":
            return AsyncOpenAI(
                api_key=os.getenv("QWEN_API_KEY"),
                base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            )
        raise ValueError(f"Unsupported provider: {provider}")
