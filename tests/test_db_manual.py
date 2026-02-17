from src.database.db_manager import DBManager
from src.database.models import User


def test_db():
    try:
        db = DBManager()
        db.init_db()
        session = db.get_session()
        print("数据库连接成功！")

        # 测试写入
        test_user = User(
            username="db_test",
            weight=70,
            height=170,
            age=30,
            gender="male",
            activity_level="moderate",
            tdee=2000,
        )
        session.merge(
            test_user
        )  # merge avoids unique constraint error on repeated runs
        session.commit()
        print("用户写入测试成功！")

        # 测试读取
        user = session.query(User).filter_by(username="db_test").first()
        print(f"读取到的用户: {user.username}, TDEE: {user.tdee}")

        session.close()
    except Exception as e:
        print(f"数据库测试失败: {e}")


if __name__ == "__main__":
    test_db()
