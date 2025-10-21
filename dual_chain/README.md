# 双链协同因子优化系统

## 简介

双链协同因子优化系统是一个先进的量化研究框架，实现了"生成→评估→优化→再评估→更新因子池"的自动化闭环流程。通过有效因子池(effective_pool)和废弃因子池(discarded_pool)的双向反馈机制，结合大语言模型(LLM)的智能优化能力，系统能够持续生成高质量的交易因子。

## 系统架构

![双链协同架构](https://i.imgur.com/placeholder.png)

系统包含以下核心组件：

1. **因子池管理器 (FactorPool)** - 管理有效因子池和废弃因子池
2. **因子评估器 (FactorEvaluator)** - 计算因子的各项评估指标
3. **LLM因子适配器 (LLMFactorAdapter)** - 集成大语言模型生成和优化因子
4. **双链协同管理器 (DualChainManager)** - 协调整个流程的运行

## 功能特性

- **双链协同**：同时维护有效因子池和废弃因子池，提供正向和负向参考
- **LLM智能优化**：利用大语言模型生成和优化因子表达式
- **自动化闭环**：实现因子从生成到评估、优化、再评估、更新池的完整闭环
- **多维度评估**：计算IC、IC-IR、Sharpe比率、年化收益率等关键指标
- **结果可视化**：生成详细的因子评估报告和统计信息

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置文件

复制配置模板并修改参数：

```bash
cp config_template.json config.json
```

配置文件说明：

```json
{
  "data_path": "data/a_share",         # 数据路径
  "pool_dir": "dual_chain/pools",     # 因子池目录
  "llm_model": "gpt-4",               # LLM模型名称
  "iterations": 5,                     # 迭代次数
  "factors_per_iteration": 5           # 每次迭代生成的因子数量
}
```

### 运行系统

使用命令行参数运行：

```bash
python run_dual_chain.py --iterations 5 --factors_per_iteration 5 --llm_model gpt-4
```

或者使用配置文件运行：

```bash
python run_dual_chain.py --config config.json
```

### 运行测试

验证组件功能是否正常：

```bash
python test_dual_chain.py
```

## 核心组件详解

### 1. 因子池 (FactorPool)

管理有效因子和废弃因子的存储与查询。

- **有效因子池**：存储通过评估的高质量因子
- **废弃因子池**：存储未通过评估的低质量因子
- **关键功能**：
  - 添加/查询/删除因子
  - 获取参考因子（用于指导新因子生成）
  - 更新因子元数据
  - 统计因子池状态

### 2. 因子评估器 (FactorEvaluator)

计算因子的各项评估指标，判断因子质量。

- **评估指标**：
  - IC (信息系数)
  - IC-IR (信息比率)
  - 因子收益率
  - Sharpe比率
  - 年化收益率
- **质量判断**：
  - 基于IC和Sharpe比率的阈值判断
  - 默认阈值：IC > 0.01, Sharpe > 0.3

### 3. LLM因子适配器 (LLMFactorAdapter)

集成大语言模型，用于生成和优化因子表达式。

- **支持的LLM模型**：
  - GPT系列
  - 阿里百炼
  - DeepSeek
  - 其他兼容OpenAI API的模型
- **主要功能**：
  - 生成因子表达式
  - 优化因子表达式
  - 生成改进建议
  - 验证因子表达式语法

### 4. 双链协同管理器 (DualChainManager)

协调整个因子优化流程的运行。

- **工作流程**：
  1. 从因子池获取参考因子
  2. 使用LLM生成新因子
  3. 评估因子质量
  4. 优化低质量因子
  5. 再次评估优化后的因子
  6. 更新因子池
- **迭代优化**：支持多轮迭代，持续改进因子质量

## 数据格式要求

### 输入数据格式

系统期望的数据格式为CSV文件，包含以下字段：

```csv
date,symbol,open,high,low,close,volume
2020-01-02,000001.SZ,15.00,15.20,14.90,15.10,10000000
...
```

### 输出文件

系统会生成以下输出文件：

1. **有效因子列表**：`pools/output/effective_factors_report_YYYYMMDD.csv`
2. **迭代结果**：`pools/output/iteration_YYYYMMDD_HHMMSS.json`
3. **最终报告**：`pools/output/final_report_YYYYMMDD_HHMMSS.json`
4. **运行日志**：`logs/dual_chain_YYYYMMDD_HHMMSS.log`

## 因子表达式示例

以下是一些有效的因子表达式示例：

```python
# 价格动量
close.pct_change(20) - close.pct_change(60)

# 成交量变化率
volume.pct_change(10)

# RSI指标
pd.Series(close, index=close.index).rolling(14).apply(lambda x: 100 - 100 / (1 + (x[x > x.shift(1)].mean() / abs(x[x <= x.shift(1)].mean()))))

# MACD指标
exp1 = close.ewm(span=12, adjust=False).mean()
exp2 = close.ewm(span=26, adjust=False).mean()
macd = exp1 - exp2
signal = macd.ewm(span=9, adjust=False).mean()
macd - signal

# 价格波动率
close.rolling(20).std() / close.rolling(20).mean()
```

## 系统调优建议

1. **调整评估阈值**：根据市场情况调整IC和Sharpe比率的阈值
2. **优化LLM提示词**：改进提示词模板以生成更高质量的因子
3. **增加迭代次数**：对于复杂的优化任务，增加迭代次数可以获得更好的结果
4. **平衡因子数量**：每次迭代生成的因子数量不宜过多，建议5-10个

## 故障排除

### 常见问题

1. **因子表达式执行失败**
   - 检查表达式语法是否正确
   - 确保使用了支持的函数和操作符

2. **LLM调用失败**
   - 检查API密钥是否配置正确
   - 验证网络连接是否正常
   - 尝试切换不同的LLM模型

3. **因子评估结果异常**
   - 检查输入数据的质量
   - 验证收益率数据的计算是否正确

## 扩展指南

### 添加新的LLM模型

1. 在`llm_factor_adapter.py`中添加新的模型适配器
2. 实现`generate`和`chat`方法
3. 在`model_adapters`字典中注册新模型

### 自定义评估指标

1. 在`factor_evaluator.py`中添加新的评估方法
2. 在`evaluate_factor`方法中集成新指标
3. 更新`determine_factor_quality`方法以使用新指标

## 许可证

© 2024 量化研究团队

## 联系方式

如有任何问题或建议，请联系：quant_research@example.com