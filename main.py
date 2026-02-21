import asyncio
from src.agent import LoseWeightAgent
from src.schemas import DailyMealPlan, FoodRecognitionResponse


async def main():
    # 初始化异步 Agent
    agent = LoseWeightAgent()

    # 1. 用户注册与 TDEE
    print("\n[1. 用户注册]")
    user_info = agent.register_user(
        username="小明",
        weight=85.0,
        height=180.0,
        age=25,
        gender="male",
        activity_level="sedentary",
    )
    print(f"用户注册成功: {user_info.username}, TDEE: {user_info.tdee:.2f}")

    # 2. 食材管理
    print("\n[2. 食材管理]")
    agent.add_ingredient("小明", "鸡胸肉")
    agent.add_ingredient("小明", "红薯")
    agent.add_ingredient("小明", "菠菜")
    ingredients = agent.get_ingredients("小明")
    print(f"小明的冰箱: {ingredients}")

    # 3. 异步餐食规划
    print("\n[3. 餐食规划]")
    meal_plan = await agent.plan_daily_meals("小明")
    if isinstance(meal_plan, DailyMealPlan):
        print(f"总目标卡路里: {meal_plan.daily_summary.target_calories} kcal")
        print(f"饮食贴士: {', '.join(meal_plan.tips)}")
        for meal_name, detail in meal_plan.meals.items():
            print(f"- {meal_name}: {detail.name} ({detail.calories} kcal)")
    else:
        print("餐食规划生成失败")

    # 4. 异步并发图片识别
    print("\n[4. 食物识别 (3路并发平均值)]")
    food_result = await agent.analyze_food("t.jpg", username="小明")
    if isinstance(food_result, FoodRecognitionResponse):
        print(f"最终识别结果: {food_result.final_food_name}")
        print(f"最终平均卡路里: {food_result.final_estimated_calories} kcal")
        print(
            f"单次识别详情 (首例): {food_result.raw_data[0].food_name} ({food_result.raw_data[0].calories} kcal)"
        )
    else:
        print(f"识别失败: {food_result}")

    # 5. 异步减重指导
    print("\n[5. 减重指导]")
    advice = await agent.get_guidance("小明", "如果我晚上加班太饿了，可以吃什么宵夜？")
    print(f"教练建议:\n{advice}")


if __name__ == "__main__":
    asyncio.run(main())
