# Alpha Master - 量化因子分析与投资组合优化系统

## 项目概述

Alpha Master是一个完整的量化因子分析与投资组合优化系统，专注于生成、评估、优化和报告金融市场中的Alpha因子。该系统支持A股、加密货币和美股三大市场，提供从因子生成到投资组合优化的全流程解决方案。

## 系统架构

系统由以下核心模块组成：

1. **因子生成器** (`factor_generator/`): 负责从原始市场数据生成多种类型的Alpha因子
2. **数据存储** (`data/`): 存储各市场的原始数据
3. **因子优化** (`portfolios/`): 对生成的因子进行筛选、优化和投资组合构建
4. **报告生成** (`report_agent/`): 自动生成因子分析和投资组合优化报告

## 目录结构

```
alpha-master/
├── a_share_factor_analysis_plots/  # A股因子分析图表
├── alpha_pool/                     # 生成的因子池数据
├── data/                           # 原始市场数据
│   ├── a_share/                    # A股数据
│   ├── crypto/                     # 加密货币数据
│   └── us/                         # 美股数据
├── evaluate_factor_pool_quality.py # 因子池质量评估脚本
├── factor_generator/               # 因子生成模块
├── factor_pool_quality_analysis_summary.md # 因子池质量分析摘要
├── factor_pool_quality_evaluation_report.md # 因子池质量评估报告
├── generate_report_input.py        # 生成报告输入数据脚本
├── market_downloader.py            # 市场数据下载工具
├── portfolios/                     # 投资组合优化模块
├── report_agent/                   # 报告生成模块
└── verify_ultra_optimization.py    # 超优化验证脚本
```

## 核心功能模块

### 1. 因子生成器 (`factor_generator/`)

因子生成器负责从原始市场数据生成多种类型的Alpha因子。系统为每个市场提供了专门的因子生成器：

- **A股因子生成器** (`a_share_alpha_factor_generator_ultra_optimized.py`)
- **加密货币因子生成器** (`crypto_alpha_factor_generator_ultra_optimized.py`)
- **美股因子生成器** (`us_alpha_factor_generator_ultra_optimized.py`)

每个因子生成器包含以下核心功能：
- 数据加载和预处理
- 多种Alpha因子的计算（动量因子、波动率因子、成交量相关性因子等）
- 因子标准化和清洗（超严格Winsorization、稳健标准化等）
- 因子质量评估

### 2. 数据存储 (`data/`)

数据目录包含三大市场的原始交易数据：

- **A股数据** (`data/a_share/`): 包含多只A股的历史行情数据
- **加密货币数据** (`data/crypto/`): 包含多种加密货币的历史行情数据
- **美股数据** (`data/us/`): 包含美股的历史行情数据

数据格式通常包括日期、开盘价、最高价、最低价、收盘价、成交量等信息。

### 3. 投资组合优化 (`portfolios/`)

投资组合优化模块负责对生成的因子进行筛选、优化和投资组合构建：

- **因子选择器** (`factor_selector.py`): 根据因子质量指标选择最优因子子集
- **因子优化器** (`factor_optimizer.py`): 对选择的因子进行权重优化
- **投资组合优化器** (`portfolio_optimizer.py`): 使用PyPortfolioOpt进行投资组合优化
- **主优化流程** (`main_optimization_pipeline.py`): 协调整个优化流程

该模块支持多种优化方法：
- 均值-方差优化
- 层次风险平价优化
- 布莱克-利特曼模型
- 夏普比率最大化

### 4. 报告生成器 (`report_agent/`)

报告生成器模块负责自动生成因子分析和投资组合优化报告：

- **报告生成器** (`report_generator.py`): 核心报告生成逻辑
- **LLM适配器** (`llm_adapter.py`): 连接不同的大语言模型提供商
- **输入处理** (`input_json.py`): 处理输入数据格式
- **主程序** (`main.py`): 命令行接口和工作流控制

支持的报告格式包括Markdown和PDF，支持多种LLM提供商如OpenAI、阿里百炼和深度求索。

## 工作流程

1. **数据准备**：将各市场的原始数据放入对应的`data/`子目录
2. **因子生成**：运行各市场的因子生成器，生成Alpha因子
3. **因子评估**：使用`evaluate_factor_pool_quality.py`评估因子质量
4. **因子优化**：运行`portfolios/main_optimization_pipeline.py`进行因子选择和优化
5. **报告生成**：使用`report_agent/main.py`生成最终分析报告

## 依赖项

### 核心依赖
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- PyPortfolioOpt

### 报告生成依赖
- 各LLM API (OpenAI, DashScope, DeepSeek)

## 使用指南

### 1. 因子生成

以A股因子生成为例：

```bash
python factor_generator/a_share_alpha_factor_generator_ultra_optimized.py
```

### 2. 投资组合优化

运行主优化流程：

```bash
python portfolios/main_optimization_pipeline.py
```

### 3. 报告生成

使用报告生成器创建分析报告：

```bash
cd report_agent
python main.py --input input_data.json --output report.md --format markdown
```

详细使用说明请参考各模块中的文档和示例。

## 输出结果

系统生成的主要输出包括：

1. **因子数据**：存储在`alpha_pool/`目录下
2. **优化因子**：存储在`portfolios/optimized_factors/`目录下
3. **投资组合权重**：存储在`portfolios/optimization_reports/`目录下
4. **分析报告**：由`report_agent`生成的最终报告
5. **可视化图表**：优化过程中的各种分析图表

## 系统特点

1. **多市场支持**：同时支持A股、加密货币和美股
2. **超优化实现**：各模块都经过性能和质量优化
3. **全面的因子库**：包含多种类型的Alpha因子
4. **灵活的优化策略**：支持多种投资组合优化方法
5. **智能报告生成**：集成LLM生成高质量分析报告

## 注意事项

1. 确保所有数据文件格式正确且完整
2. 运行前请安装所有必需的依赖项
3. 使用LLM生成报告时需要配置相应的API密钥
4. 大规模数据处理可能需要较长时间和足够的计算资源
