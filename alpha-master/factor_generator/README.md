# Alpha因子生成器

## 模块概述

因子生成器模块是Alpha Master系统的核心组件，负责从原始市场数据中提取和生成各种Alpha因子。该模块实现了高度优化的因子计算算法，确保生成的因子具有良好的预测能力和稳定性。

## 文件结构

```
factor_generator/
├── a_share_alpha_factor_generator_ultra_optimized.py    # A股Alpha因子生成器
├── crypto_alpha_factor_generator_ultra_optimized.py     # 加密货币Alpha因子生成器
├── us_alpha_factor_generator_ultra_optimized.py         # 美股Alpha因子生成器
└── README.md                                           # 本文档
```

## 核心功能

### 1. 数据加载与预处理

每个因子生成器都包含专门的数据加载器，负责：
- 从CSV文件加载市场数据
- 处理缺失值和异常值
- 计算收益率等基本指标
- 构建数据矩阵用于批量处理

### 2. 因子标准化

`UltraFactorNormalizer`类实现了多种高级标准化技术：
- **超严格Winsorization**：限制极端值，防止异常值影响
- **超稳健标准化**：基于中位数绝对偏差(MAD)的稳健标准化
- **超Z-score标准化**：基于均值和标准差的标准化
- **超排名标准化**：将因子值转换为排名百分比

### 3. 因子生成算法

每个市场的因子生成器都实现了多种先进的因子生成算法：

#### GFlowNet风格因子
- **动量因子**：捕捉价格趋势
- **波动率因子**：衡量价格波动程度
- **成交量相关性因子**：分析成交量与价格的关系
- **对数价格因子**：处理价格的非线性特性
- **复合技术因子**：结合多种技术指标

#### AlphaAgent风格因子
- 基于机器学习的预测因子
- 市场微观结构因子
- 情绪分析因子

## 使用方法

### A股因子生成

```python
from a_share_alpha_factor_generator_ultra_optimized import AShareDataLoader, AShareAlphaGFNGenerator

# 加载数据
data_loader = AShareDataLoader(data_path="../data/a_share")

# 创建因子生成器
generator = AShareAlphaGFNGenerator(data_loader)

# 生成因子
factors = generator.generate_factors()

# 保存因子
for factor_name, factor_data in factors.items():
    factor_data.to_csv(f"../alpha_pool/a_share_{factor_name}.csv")
```

### 加密货币因子生成

```python
from crypto_alpha_factor_generator_ultra_optimized import CryptoDataLoader, CryptoAlphaGFNGenerator

# 加载数据
data_loader = CryptoDataLoader(data_path="../data/crypto")

# 创建因子生成器
generator = CryptoAlphaGFNGenerator(data_loader)

# 生成因子
factors = generator.generate_factors()
```

### 美股因子生成

```python
from us_alpha_factor_generator_ultra_optimized import USShareDataLoader, USShareAlphaGFNGenerator

# 加载数据
data_loader = USShareDataLoader(data_path="../data/us")

# 创建因子生成器
generator = USShareAlphaGFNGenerator(data_loader)

# 生成因子
factors = generator.generate_factors()
```

## 因子质量保证

该模块实现了多种质量保证机制：

1. **数值稳定性处理**：严格处理无穷大、NaN和异常值
2. **超优化计算**：使用向量化运算提高计算效率
3. **多阶段验证**：对生成的因子进行统计特性验证
4. **参数自适应**：根据数据特性自动调整计算参数

## 依赖项

- pandas
- numpy
- scikit-learn
- statsmodels

## 注意事项

1. 确保数据目录结构正确，CSV文件格式符合要求
2. 大规模数据处理可能需要较长时间，建议使用性能较好的计算机
3. 因子生成参数可以根据具体需求进行调整
4. 生成的因子应进行回测验证后再用于实际投资决策

## 输出结果

因子生成器输出的因子数据格式为CSV文件，包含以下信息：
- 日期索引
- 资产标识符（股票代码或加密货币代码）
- 因子值

生成的因子将被保存到项目根目录的`alpha_pool/`文件夹中，供后续的因子评估和优化使用。