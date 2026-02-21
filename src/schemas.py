from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Union

class UserBase(BaseModel):
    username: str
    weight: float
    height: float
    age: int
    gender: str
    activity_level: str

class UserSchema(UserBase):
    tdee: float

class IngredientSchema(BaseModel):
    name: str

class MealDetail(BaseModel):
    name: str = Field(..., description="菜品名称")
    calories: int = Field(..., description="卡路里")
    ingredients_used: List[str] = Field(default_factory=list, description="使用的食材")
    instructions: str = Field("", description="简短做法")

class DailySummary(BaseModel):
    target_calories: int
    total_protein: Optional[Union[str, int]] = None
    total_carbs: Optional[Union[str, int]] = None
    total_fat: Optional[Union[str, int]] = None

class DailyMealPlan(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    daily_summary: DailySummary
    meals: Dict[str, MealDetail]
    tips: List[str] = Field(default_factory=list)

class FoodAnalysisResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    food_name: str = Field(..., alias="food_name")
    calories: int = Field(..., alias="estimated_calories")
    confidence: float = Field(default=0.0)
    components: List[str] = Field(default_factory=list)
    nutrients: Dict[str, Union[str, float, int]] = Field(default_factory=dict, description="营养成分，如 {'蛋白质': '10g'}")

class FoodRecognitionResponse(BaseModel):
    final_food_name: str
    final_estimated_calories: int
    raw_data: List[FoodAnalysisResult]
    timestamp: str


class FoodNutritionSearchResult(BaseModel):
    """USDA 食物营养数据检索结果。"""
    fdc_id: int = Field(..., description="USDA FDC ID")
    description: str = Field(..., description="食物描述")
    food_category: str = Field(default="", description="食物分类")
    calories_per_100g: Optional[float] = Field(None, description="每100g热量(kcal)")
    protein_per_100g: Optional[float] = Field(None, description="每100g蛋白质(g)")
    fat_per_100g: Optional[float] = Field(None, description="每100g脂肪(g)")
    carbs_per_100g: Optional[float] = Field(None, description="每100g碳水化合物(g)")
    similarity: float = Field(default=0.0, description="相似度得分")

