# Alpha因子分析报告生成器 (Report Agent)

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)

一个专业的Alpha因子分析报告生成工具，能够将因子分析数据转换为结构化的专业报告，支持Markdown和PDF格式输出。

## 功能特性

- 📊 自动生成完整的Alpha因子分析报告
- 🧠 支持多种LLM提供商：OpenAI(GPT)、阿里百炼、DeepSeek等
- 📝 支持Markdown和PDF格式输出
- 🔍 详细的因子分析，包括原理、实现、优化和评估
- 📈 投资组合优化结果分析和可视化
- ⚙️ 灵活的配置选项，支持多种模型和参数调优

## 系统架构

![系统架构](https://via.placeholder.com/800x400?text=Alpha+Report+Agent+Architecture)

### 主要组件

1. **LLM适配器** - 支持多种LLM提供商的统一接口（OpenAI、阿里百炼、DeepSeek）
2. **报告生成器** - 核心组件，处理数据并生成报告内容
3. **主程序入口** - 提供命令行接口，处理参数和配置

## 安装说明

### 1. 克隆仓库

```bash
git clone <repository_url>
cd report_agent
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 安装PDF生成依赖（可选）

要生成PDF格式的报告，需要安装以下PDF引擎之一：

**Windows:**
```bash
# 使用chocolatey安装wkhtmltopdf
choco install wkhtmltopdf

# 或者安装MikTeX（LaTeX）
choco install miktex
```

**macOS:**
```bash
# 使用Homebrew安装wkhtmltopdf
brew install wkhtmltopdf

# 或者安装MacTeX（LaTeX）
brew install mactex
```

**Linux:**
```bash
# 安装wkhtmltopdf
sudo apt-get install wkhtmltopdf

# 或者安装TeX Live（LaTeX）
sudo apt-get install texlive-full
```

## 配置说明

### 环境变量

根据使用的LLM提供商设置相应的API密钥（推荐方式）：

**OpenAI:**
- Windows: `set OPENAI_API_KEY=your_api_key_here`
- macOS/Linux: `export OPENAI_API_KEY=your_api_key_here`

**阿里百炼:**
- Windows: `set DASHSCOPE_API_KEY=your_api_key_here`
- macOS/Linux: `export DASHSCOPE_API_KEY=your_api_key_here`

**DeepSeek:**
- Windows: `set DEEPSEEK_API_KEY=your_api_key_here`
- macOS/Linux: `export DEEPSEEK_API_KEY=your_api_key_here`

### 配置文件

复制配置模板并根据需要修改：

```bash
cp config_template.json config.json
```

编辑`config.json`文件，支持多种LLM提供商：

**OpenAI配置示例:**
```json
{
  "llm": {
    "provider": "openai",
    "model_name": "gpt-4",
    "api_key": "your_api_key_here",
    "base_url": "",  # 使用OpenAI默认URL
    "temperature": 0.1,
    "max_tokens": 4000
  }
}
```

**阿里百炼配置示例:**
```json
{
  "llm": {
    "provider": "dashscope",
    "model_name": "qwen-plus",
    "api_key": "your_api_key_here",
    "temperature": 0.1,
    "max_tokens": 4000
  }
}
```

**DeepSeek配置示例:**
```json
{
  "llm": {
    "provider": "deepseek",
    "model_name": "deepseek-chat",
    "api_key": "your_api_key_here",
    "temperature": 0.1,
    "max_tokens": 4000
  }
}
```

## 使用方法

### 基本用法

生成Markdown格式报告：

```bash
python main.py --input input_data.json --output report.md
```

生成PDF格式报告：

```bash
python main.py --input input_data.json --output report.pdf --format pdf
```

### 高级用法

使用自定义配置文件：

```bash
python main.py --input input_data.json --output report.md --config config.json
```

使用OpenAI模型：

```bash
python main.py --input input_data.json --output report.md \
    --provider openai \
    --model gpt-4-turbo \
    --temperature 0.2 \
    --max-tokens 6000
```

使用阿里百炼模型：

```bash
python main.py --input input_data.json --output report.md \
    --provider dashscope \
    --model qwen-plus \
    --api-key your_dashscope_key
```

使用DeepSeek模型：

```bash
python main.py --input input_data.json --output report.md \
    --provider deepseek \
    --model deepseek-chat \
    --api-key your_deepseek_key
```

启用调试日志：

```bash
python main.py --input input_data.json --output report.md --debug
```

## 输入数据格式

输入数据必须是JSON格式，包含以下主要部分：

### 1. 元数据 (metadata)

```json
{
  "metadata": {
    "generated_at": "2025-09-30T23:43:13.863061",
    "data_source": "your_data_source",
    "total_markets": 3,
    "total_factors": 30,
    "optimization_methods": ["max_sharpe", "min_volatility", ...]
  }
}
```

### 2. 因子分析数据 (factor_analysis)

每个市场的因子统计和质量评估：

```json
{
  "factor_analysis": {
    "a_share": {
      "factor_statistics": {
        "FactorName1": {
          "count": 4836,
          "mean": 0.0,
          "std": 2.5095789670829327,
          "min": -6.401723013967534,
          "max": 8.171049792345565,
          "range": 14.5727728063131,
          "cv": 0.0
        },
        // 更多因子...
      },
      "factor_quality": {
        "FactorName1": {
          "stability_score": 0.0,
          "range_score": 0.6862112058501234,
          "distribution_score": 1.0,
          "overall_quality": 0.5620704019500411,
          "is_outlier_prone": true,
          "is_extreme_range": true
        },
        // 更多因子...
      },
      "selected_factors": ["FactorName1", "FactorName2", ...],
      "problematic_factors": ["FactorName1", ...]
    },
    // 更多市场...
  }
}
```

### 3. 优化结果 (optimization_results)

投资组合优化的详细结果：

```json
{
  "optimization_results": {
    "max_sharpe": {
      "return": 0.156,
      "sharpe_ratio": 1.89,
      "max_drawdown": -0.125,
      "ic": 0.052,
      "factor_weights": {
        "FactorName1": 0.25,
        "FactorName2": 0.30,
        // 更多因子权重...
      }
    },
    // 更多优化方法...
  }
}
```

## 报告内容

生成的报告包含以下主要部分：

1. **报告摘要** - 整体分析概述和关键发现
2. **Alpha因子详细分析** - 按市场分类的因子分析
   - 因子原理和理论基础
   - 因子统计和质量评估
   - 因子选择和优化过程
3. **组合优化结果分析** - 不同优化方法的结果对比
   - 收益率、夏普比率、最大回撤等指标
   - IC分析和因子权重分配
4. **结论与投资建议** - 基于分析的具体建议
5. **风险提示与免责声明**

## 开发说明

### 项目结构

```
report_agent/
├── __init__.py          # 包初始化文件
├── llm_adapter.py       # LLM适配器模块
├── report_generator.py  # 报告生成器核心模块
├── main.py              # 主程序入口
├── config_template.json # 配置文件模板
├── requirements.txt     # 依赖包列表
└── README.md           # 项目文档
```

### 扩展功能

1. **添加新的报告部分**：在`report_generator.py`中扩展`_generate_markdown_report`方法
2. **支持新的LLM提供商**：在`llm_adapter.py`中继承`BaseLLMAdapter`并实现相应方法
3. **自定义报告格式**：修改`_generate_markdown_report`方法中的报告结构

## 故障排除

### PDF生成失败

如果PDF生成失败，请尝试：

1. 确保已安装PDF引擎（wkhtmltopdf或LaTeX）
2. 检查环境变量设置是否正确
3. 尝试使用Markdown格式输出作为替代

### LLM连接错误

1. 检查对应提供商的API密钥是否正确设置
2. 验证网络连接是否正常
3. 确保使用了正确的provider参数和模型名称
4. 如果使用代理，确保代理设置正确

## 许可证

[MIT License](LICENSE)

## 联系方式

如有问题或建议，请联系：quant_dev@example.com