
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "record_weight",
            "description": "记录用户的体重",
            "parameters": {
                "type": "object",
                "properties": {
                    "weight_kg": {
                        "type": "number",
                        "description": "体重数值，单位为 kg",
                    },
                    "notes": {
                        "type": "string",
                        "description": "备注信息，例如：早起空腹",
                    },
                },
                "required": ["weight_kg"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_weight_history",
            "description": "查询用户的体重历史记录",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "返回记录的最大数量，默认为 10",
                        "default": 10,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_tdee",
            "description": "计算用户的 TDEE (总每日能量消耗) 和 BMR (基础代谢率)",
            "parameters": {
                "type": "object",
                "properties": {
                    "weight_kg": {"type": "number", "description": "体重 (kg)"},
                    "height_cm": {"type": "number", "description": "身高 (cm)"},
                    "age": {"type": "integer", "description": "年龄"},
                    "gender": {
                        "type": "string",
                        "enum": ["male", "female"],
                        "description": "性别",
                    },
                    "activity_level": {
                        "type": "string",
                        "enum": [
                            "sedentary",
                            "light",
                            "moderate",
                            "active",
                            "very_active",
                        ],
                        "description": "活动等级",
                    },
                },
                "required": [
                    "weight_kg",
                    "height_cm",
                    "age",
                    "gender",
                    "activity_level",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_food",
            "description": "记录用户摄入的食物和热量",
            "parameters": {
                "type": "object",
                "properties": {
                    "food_name": {"type": "string", "description": "食物名称"},
                    "calories": {"type": "integer", "description": "摄入热量 (kcal)"},
                    "meal_type": {
                        "type": "string",
                        "enum": ["breakfast", "lunch", "dinner", "snack"],
                        "description": "餐次类型",
                    },
                },
                "required": ["food_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_food_log",
            "description": "查询用户的饮食摄入记录",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "返回记录的最大数量，默认为 10",
                        "default": 10,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_food_nutrition",
            "description": "在食物营养数据库中搜索相关食物及其热量/营养信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词，例如：鸡蛋、鸡胸肉"},
                    "limit": {
                        "type": "integer",
                        "description": "返回结果的最大数量，默认为 5",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
]
