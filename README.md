# Alpha因子生成与评估系统

这是一个完整的量化交易Alpha因子生成、筛选优化和回测评估系统，主要包含三个核心模块：因子生成器、筛选优化器和回测评估器。

## 系统架构

### 1. 因子生成器 (a_factor_generate)
多种算法并行生成Alpha因子，包括：
- **a_agent**: 基于智能代理的因子生成
- **a_gen**: 基础生成器
- **a_genetic**: 遗传算法生成器
- **a_gfn**: 基于生成流网络的因子生成
- **a_miner**: 基于AlphaMiner的因子挖掘

### 2. 筛选优化器 (dual_chain)
采用双链式结构对生成的因子进行筛选和优化：
- 因子评估：计算IC、IC-IR、分组收益率等指标
- 因子池管理：维护有效因子池和待评估因子池
- 迭代优化：持续筛选出最优质的Alpha因子

### 3. 回测评估器 (compare)
对优化后的因子进行全面的回测和评估：
- 多因子模型构建
- 回测策略执行
- 性能指标计算
- 可视化报告生成

## 目录结构

```
a_factor_generate/       # 因子生成器
  ├── a_agent/           # 智能代理因子生成
  ├── a_gen/             # 基础因子生成
  ├── a_genetic/         # 遗传算法因子生成
  ├── a_gfn/             # 生成流网络因子生成
  └── a_miner/           # AlphaMiner因子挖掘
dual_chain/              # 筛选优化器
  ├── dual_chain_manager.py  # 双链管理器
  ├── factor_evaluator.py    # 因子评估器
  └── factor_pool.py         # 因子池管理
compare/                 # 回测评估器
  ├── alpha_evaluation.py    # 回测评估核心
  └── alpha_evaluation_report.json  # 评估报告
data/                    # 数据目录
  ├── a_share/           # A股数据
  ├── crypto/            # 加密货币数据
  └── us/                # 美股数据
alpha_pool/              # 因子池存储
```

## 核心功能

### 因子生成器
- 支持多种因子类型：动量因子、波动率因子、交易量因子、统计因子等
- 实现多种生成算法：智能代理、遗传算法、生成流网络等
- 自动保存生成的因子表达式和计算结果

### 筛选优化器
- **因子评估器 (FactorEvaluator)**: 计算信息系数(IC)、IC信息比率、分组收益率等核心指标
- **因子池 (FactorPool)**: 管理有效因子和待评估因子
- **双链管理器 (DualChainManager)**: 协调因子生成、评估和优化的完整流程

### 回测评估器
- **AlphaEvaluationSystem**: 基于LightGBM的多因子模型构建
- 支持各种性能指标：收益率、夏普比率、最大回撤等
- 生成可视化报告和评估结果

## 快速开始

### 1. 生成因子
```bash
cd a_factor_generate/a_miner
python run_alphaminer.py
```

### 2. 筛选优化因子
```bash
cd ../../dual_chain
python run_dual_chain.py
```

### 3. 回测评估因子
```bash
cd ../compare
python alpha_evaluation.py
```

## 关键指标说明

### 因子质量指标
- **IC (Information Coefficient)**: 因子值与未来收益率的相关系数
- **IC-IR (Information Ratio)**: IC的均值除以IC的标准差
- **分组收益**: 将股票按因子值分组后的平均收益率

### 回测性能指标
- **收益率**: 策略的累计和日均收益率
- **夏普比率**: 超额收益除以波动率
- **最大回撤**: 策略历史最大亏损幅度
- **胜率**: 盈利交易占总交易的比例

## 支持的数据类型

- A股市场 (CSI500, CSI1000)
- 美股市场
- 加密货币市场

## 技术栈

- Python 3.8+
- pandas, numpy: 数据处理
- scikit-learn, LightGBM: 机器学习模型
- matplotlib, seaborn: 数据可视化
- 各种量化分析库

## 注意事项

1. 本系统仅用于研究和回测，不构成投资建议
2. 因子生成和回测需要大量计算资源
3. 建议使用虚拟环境安装依赖

## 开发指南

### 添加自定义因子
1. 在对应的生成器目录下创建新的因子文件
2. 实现calculate_factor函数
3. 在生成器的运行脚本中注册新因子

### 修改评估指标
1. 编辑dual_chain/factor_evaluator.py
2. 添加或修改评估方法
3. 更新评估逻辑

## 项目维护

- 定期更新数据以保持因子的有效性
- 根据市场变化调整评估参数
- 持续优化因子生成算法

## 许可证

本项目仅供研究使用，未经授权不得用于商业用途。