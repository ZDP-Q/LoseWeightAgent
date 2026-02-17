class TDEECalculator:
    ACTIVITY_MULTIPLIERS = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9,
    }

    @staticmethod
    def calculate_bmr(weight: float, height: float, age: int, gender: str) -> float:
        if gender.lower() == "male":
            return 10 * weight + 6.25 * height - 5 * age + 5
        else:
            return 10 * weight + 6.25 * height - 5 * age - 161

    @classmethod
    def calculate_tdee(
        cls, weight: float, height: float, age: int, gender: str, activity_level: str
    ) -> float:
        bmr = cls.calculate_bmr(weight, height, age, gender)
        multiplier = cls.ACTIVITY_MULTIPLIERS.get(activity_level.lower(), 1.2)
        return bmr * multiplier
