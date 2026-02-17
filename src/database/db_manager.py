from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base


class DBManager:
    """数据库管理器，通过构造参数接收数据库 URL，不再依赖环境变量。"""

    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def init_db(self):
        Base.metadata.create_all(bind=self.engine)

    def get_session(self):
        return self.SessionLocal()
