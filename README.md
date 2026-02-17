# LoseWeightAgent (LWAgent) 🏃‍♂️🥗

LoseWeightAgent 是一款基于大语言模型的健康管理助手，旨在通过智能化手段帮助用户达成减重目标。

## 核心功能

- **TDEE 计算**：基于用户身高、体重、年龄及活动水平，科学计算每日总消耗。
- **智能餐食规划**：结合用户现有食材与热量缺口，自动生成符合 Pydantic 规范化的每日饮食计划。
- **3路异步并发食物识别**：采用 `qwen-vl-plus` 视觉模型，通过 3 次并发识别并取热量平均值，提升识别准确度与响应速度。
- **AI 减重教练**：提供专业的减重指导建议与问题答疑。

## 技术栈

- **语言**：Python 3.12+
- **包管理**：[uv](https://github.com/astral-sh/uv)
- **大模型**：通义千问 (Qwen3.5-Plus, Qwen-VL-Plus)
- **数据库**：PostgreSQL / SQLite (SQLAlchemy)
- **规范化**：Pydantic V2 (支持 Alias 映射)

## 快速开始

### 1. 环境准备

确保已安装 `uv`，然后克隆项目并安装依赖：

```bash
uv sync
```

### 2. 配置环境变量

在根目录下创建 `.env` 文件，并填写以下内容：

```env
# 阿里云百炼 API Key
QWEN_API_KEY=your_api_key_here
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 数据库配置 (若不配置则默认使用 SQLite)
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres
```

### 3. 运行项目

执行主程序进行全链路演示：

```bash
$env:PYTHONPATH="."
uv run main.py
```

## 项目结构

```text
D:\desktop\LWAgent\
├── src/
│   ├── database/       # 数据库模型与管理
│   ├── llm/            # LLM 客户端工厂
│   ├── prompts/        # Jinja2 提示词模板
│   ├── services/       # 业务逻辑（TDEE、识别、规划）
│   ├── agent.py        # 核心 Agent 类
│   ├── schemas.py      # Pydantic 数据模型
│   └── utils.py        # 工具函数（JSON 解析等）
├── tests/              # 自动化测试脚本
└── main.py             # 集成演示入口
```

## 提交规范

本项目遵循基本的 Git 提交规范，提交信息使用中文，分类清晰。
