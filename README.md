# LoseWeightAgent (LWAgent) 🏃‍♂️🥗

LoseWeightAgent 是一款基于 **qwen3.5-plus** 大语言模型的健康管理助手。

## 核心功能

- **TDEE 计算**：基于用户身高、体重、年龄及活动水平，科学计算每日总消耗。
- **智能餐食规划**：结合用户现有食材与热量缺口，基于 **qwen3.5-plus** 自动生成符合 Pydantic 规范化的每日饮食计划。
- **3路异步并发食物识别**：采用 **qwen3.5-plus** 强大的多模态能力，通过 3 次并发识别并取热量平均值，极大地提升识别准确度。
- **AI 减重教练**：提供专业的减重指导建议与问题答疑。

## 技术栈

- **模型**：通义千问 **qwen3.5-plus** (统一应用于文本与视觉任务)
- **包管理**：[uv](https://github.com/astral-sh/uv)
- **核心库**：Pydantic V2, SQLAlchemy, Jinja2, OpenAI SDK

## 快速开始

### 1. 配置环境变量

```env
QWEN_API_KEY=your_api_key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 2. 运行

```bash
$env:PYTHONPATH="."
uv run main.py
```
