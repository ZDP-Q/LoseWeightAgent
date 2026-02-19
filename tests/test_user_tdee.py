import asyncio
import sys
from pathlib import Path

# 添加后端根目录以便导入
backend_root = Path(__file__).parent.parent.parent
sys.path.append(str(backend_root))

from LoseWeightAgent.src.agent import LoseWeightAgent  # noqa: E402
from src.core.config import get_settings  # noqa: E402

async def test_tdee():
    settings = get_settings()
    # TDEE 计算不需要数据库 session 也能运行内部逻辑，但 register_user 需要数据库连接
    # 如果 database_url 为空，agent 内部会尝试初始化 DBManager
    # 更好的做法是传入 settings.database.url
    agent = LoseWeightAgent(
        api_key=settings.llm.api_key,
        database_url=settings.database.url
    )
    print("--- 测试用户注册与 TDEE 计算 ---")
    user_data = agent.register_user(
        username="test_user_tdee",
        weight=80.0,
        height=180.0,
        age=25,
        gender="male",
        activity_level="active"
    )
    print(f"注册结果: {user_data}")
    
    tdee = agent.get_user_tdee("test_user_tdee")
    print(f"查询 TDEE: {tdee} kcal")

if __name__ == "__main__":
    asyncio.run(test_tdee())
