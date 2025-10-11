# 投资组合优化模块

## 模块概述

投资组合优化模块是Alpha Master系统的关键组件，负责对生成的Alpha因子进行筛选、优化和投资组合构建。该模块实现了多种先进的优化算法，旨在最大化投资组合的风险调整收益。

## 文件结构

```
portfolios/
├── __pycache__/                      # Python编译文件
├── comprehensive_optimization_report.md  # 综合优化报告
├── factor_optimizer.py               # 因子优化器
├── factor_selector.py                # 因子选择器
├── main_optimization_pipeline.py     # 主优化流程
├── optimization_plots/               # 优化过程图表
│   ├── risk_return_scatter.png       # 风险收益散点图
│   └── sharpe_ratio_comparison.png   # 夏普比率比较图
├── optimization_reports/             # 优化结果报告
│   ├── a_share_portfolio_weights.csv # A股投资组合权重
│   ├── crypto_portfolio_weights.csv  # 加密货币投资组合权重
│   ├── portfolio_optimization_summary.csv # 优化摘要
│   └── us_portfolio_weights.csv      # 美股投资组合权重
├── optimized_factors/                # 优化后的因子
│   ├── a_share_optimized_factors.csv # A股优化因子
│   ├── a_share_optimized_factors_stats.csv # A股优化因子统计
│   ├── crypto_optimized_factors.csv  # 加密货币优化因子
│   ├── crypto_optimized_factors_stats.csv # 加密货币优化因子统计
│   ├── us_optimized_factors.csv      # 美股优化因子
│   └── us_optimized_factors_stats.csv # 美股优化因子统计
├── portfolio_optimizer.py            # 投资组合优化器
├── selected_factors/                 # 选择的因子
│   ├── a_share_selected_factors.csv  # A股选择因子
│   ├── a_share_selected_factors_stats.csv # A股选择因子统计
│   ├── crypto_selected_factors.csv   # 加密货币选择因子
│   ├── crypto_selected_factors_stats.csv # 加密货币选择因子统计
│   ├── us_selected_factors.csv       # 美股选择因子
│   └── us_selected_factors_stats.csv # 美股选择因子统计
├── summary_statistics.csv            # 汇总统计数据
└── README.md                         # 本文档
```

## 核心组件

### 1. 因子选择器 (`factor_selector.py`)

因子选择器负责从生成的因子池中选择最优的因子子集，主要功能包括：

- **基于相关性的筛选**：移除高度相关的因子，减少冗余
- **基于信息比率的排序**：选择信息比率最高的因子
- **基于风险调整收益的评估**：综合考虑因子的收益和风险特性
- **多目标优化选择**：平衡多个评估指标

### 2. 因子优化器 (`factor_optimizer.py`)

因子优化器对选择的因子进行权重优化，实现方式包括：

- **线性回归优化**：基于预测能力优化因子权重
- **机器学习方法**：使用高级算法优化因子组合
- **因子衰减调整**：考虑因子时效性，动态调整权重
- **市场状态自适应**：根据市场环境调整因子权重

### 3. 投资组合优化器 (`portfolio_optimizer.py`)

投资组合优化器是该模块的核心，负责最终的资产配置，主要功能包括：

- **均值-方差优化**：最大化预期收益，同时控制风险
- **层次风险平价优化**：基于资产层次结构的风险分配
- **布莱克-利特曼模型**：结合市场均衡和投资者观点
- **夏普比率最大化**：优化风险调整收益
- **风险约束优化**：在各种风险限制下寻找最优组合

### 4. 主优化流程 (`main_optimization_pipeline.py`)

主优化流程协调整个优化过程，包括：

- 加载选择的因子数据
- 调用因子优化器进行因子权重优化
- 使用投资组合优化器构建最优投资组合
- 生成优化报告和可视化图表
- 保存优化结果

## 使用方法

### 运行完整优化流程

最简单的使用方式是运行主优化流程脚本：

```bash
python main_optimization_pipeline.py
```

这将自动处理所有市场的因子选择、优化和投资组合构建。

### 单独使用投资组合优化器

```python
from portfolio_optimizer import PortfolioOptimizer

# 创建优化器实例
optimizer = PortfolioOptimizer(selected_factors_path="selected_factors")

# 对特定市场进行优化
for market in ["a_share", "crypto", "us"]:
    try:
        # 加载因子数据
        factor_df, stats_df = optimizer.load_selected_factors(market)
        
        # 准备因子收益率
        returns_df = optimizer.prepare_factor_returns(factor_df, market)
        
        # 计算期望收益率
        exp_returns = optimizer.calculate_expected_returns(returns_df)
        
        # 计算风险模型
        risk_model = optimizer.calculate_risk_model(returns_df)
        
        # 执行优化
        weights = optimizer.optimize_portfolio(exp_returns, risk_model)
        
        # 保存结果
        optimizer.save_portfolio_weights(weights, market)
        
    except Exception as e:
        print(f"市场 {market} 优化失败: {e}")
```

## 优化方法详解

### 均值-方差优化

基于Markowitz投资组合理论，在给定风险水平下最大化预期收益，或在给定预期收益下最小化风险。

### 层次风险平价 (HRP) 优化

基于资产间的相关性构建层次结构，然后在每个层次分配风险，避免传统方法对估计误差的敏感性。

### 布莱克-利特曼 (Black-Litterman) 模型

结合市场均衡收益和投资者观点，生成更合理的预期收益估计，减少对历史数据的依赖。

### 夏普比率最大化

直接优化投资组合的夏普比率（超额收益与标准差的比值），这是最常用的风险调整收益指标。

## 输出结果

优化模块生成的主要输出包括：

1. **优化后的因子数据**：存储在`optimized_factors/`目录
2. **投资组合权重**：存储在`optimization_reports/`目录
3. **可视化图表**：存储在`optimization_plots/`目录
4. **综合优化报告**：`comprehensive_optimization_report.md`

## 依赖项

- pandas
- numpy
- PyPortfolioOpt
- matplotlib
- seaborn
- scipy

## 注意事项

1. 优化结果高度依赖于输入的因子数据质量
2. 应定期重新优化，以适应市场变化
3. 历史表现不代表未来表现，投资决策需谨慎
4. 大规模优化可能需要较长计算时间

## 性能调优

- 对于大规模数据集，可以调整优化算法的精度参数
- 考虑使用并行计算加速因子评估过程
- 对于实时应用，可预计算部分结果并缓存