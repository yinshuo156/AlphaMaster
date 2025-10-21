#!/usr/bin/env python3
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import json
import datetime

# 添加Genetic-Alpha-main路径
sys.path.insert(0, str(Path(r'c:\Users\Administrator\Desktop\alpha-master\Genetic-Alpha-main')))

# 导入Genetic-Alpha的核心模块
from genetic import SymbolicRegressor
from fitness import make_fitness
from functions import _function_map
print("成功导入Genetic-Alpha模块")

def load_csi500_data(data_dir):
    """加载CSI500数据"""
    print(f"正在从 {data_dir} 加载CSI500数据...")
    
    # 读取成分股列表
    constituents_path = os.path.join(data_dir, 'constituents.csv')
    if os.path.exists(constituents_path):
        constituents = pd.read_csv(constituents_path)
        print(f"已加载成分股列表，共 {len(constituents)} 只股票")
    else:
        print("未找到成分股列表，使用目录中的CSV文件")
    
    # 获取所有股票文件
    stock_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and f != 'constituents.csv' and f != 'daily_data.csv']
    print(f"找到 {len(stock_files)} 个股票数据文件")
    
    # 读取第一只股票的数据来获取日期范围
    sample_stock = pd.read_csv(os.path.join(data_dir, stock_files[0]))
    date_range = sample_stock['date'].tolist()
    n_days = len(date_range)
    n_stocks = min(20, len(stock_files))  # 限制股票数量以加快处理
    
    # 创建特征矩阵 (股票 x 日期 x 特征)
    # 特征: open, high, low, close, volume
    data_array = np.zeros((n_stocks, n_days, 5))
    stock_codes = []
    
    for i, stock_file in enumerate(stock_files[:n_stocks]):
        try:
            stock_data = pd.read_csv(os.path.join(data_dir, stock_file))
            stock_code = stock_file.split('.')[0]
            stock_codes.append(stock_code)
            
            # 填充特征数据
            data_array[i, :, 0] = stock_data['open'].values  # open
            data_array[i, :, 1] = stock_data['high'].values  # high
            data_array[i, :, 2] = stock_data['low'].values   # low
            data_array[i, :, 3] = stock_data['close'].values # close
            data_array[i, :, 4] = stock_data['volume'].values # volume
            
        except Exception as e:
            print(f"处理股票文件 {stock_file} 时出错: {e}")
    
    # 转置为 (日期 x 股票 x 特征)
    data_array = np.transpose(data_array, (1, 0, 2))
    
    # 计算未来收益率作为目标变量
    close_prices = data_array[:, :, 3]  # 收盘价
    # 计算未来5天收益率
    target_returns = np.zeros_like(close_prices[:-5])
    for i in range(len(close_prices) - 5):
        target_returns[i] = (close_prices[i+5] - close_prices[i]) / (close_prices[i] + 1e-10)
    
    print(f"数据加载完成，特征形状: {data_array.shape}, 目标形状: {target_returns.shape}")
    return {
        'X': data_array[:-5],  # 输入数据
        'y': target_returns,   # 目标变量
        'date_range': date_range[:-5],
        'stock_codes': stock_codes
    }

def custom_fitness(y, y_pred, sample_weight=None):
    """自定义的适应度函数，使用IC相关系数"""
    try:
        # 计算每一行（每一天）的相关系数
        ic_values = []
        for i in range(y.shape[0]):
            if np.std(y[i]) > 0 and np.std(y_pred[i]) > 0:
                ic = np.corrcoef(y[i], y_pred[i])[0, 1]
                ic_values.append(ic)
        
        # 计算平均IC
        mean_ic = np.mean(ic_values) if ic_values else 0
        return abs(mean_ic)  # 使用IC绝对值作为适应度
    except:
        return 0

def gen_better_formula(fit_result, threshold=0.0):
    """从fit结果中提取超过阈值的公式"""
    if hasattr(fit_result, '_programs') and fit_result._programs:
        # 从最后一代中提取
        last_generation = fit_result._programs[-1]
        better_expressions = []
        
        for program in last_generation:
            if hasattr(program, 'fitness_') and program.fitness_ > threshold:
                expr_str = str(program)
                better_expressions.append({
                    'expression': expr_str,
                    'fitness': program.fitness_
                })
        
        # 按适应度排序
        better_expressions.sort(key=lambda x: x['fitness'], reverse=True)
        return better_expressions
    return []

def main():
    """主函数"""
    # 数据和输出路径
    data_dir = r'c:\Users\Administrator\Desktop\alpha-master\data\a_share\csi500data'
    output_dir = r'c:\Users\Administrator\Desktop\alpha-master\a_factor_generate\a_genetic'
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    data = load_csi500_data(data_dir)
    X = data['X']
    y = data['y']
    
    # 数据形状已符合要求，无需重塑
    # Genetic-Alpha的SymbolicRegressor期望3维输入: (n_days, n_stocks, n_features)
    n_days, n_stocks, n_features = X.shape
    print(f"数据形状确认: X={X.shape}, y={y.shape}")
    
    # 创建适应度函数
    fitness_function = make_fitness(
        function=custom_fitness,
        greater_is_better=True
    )
    
    # 定义函数集和特征名
    function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'abs']
    feature_names = ['open', 'high', 'low', 'close', 'volume']
    
    print("开始运行遗传编程生成因子...")
    
    # 创建符号回归器
    est_gp = SymbolicRegressor(
        population_size=50,
        generations=3,
        tournament_size=5,
        init_depth=(1, 3),
        function_set=function_set,
        metric=fitness_function,
        feature_names=feature_names,
        verbose=1,
        const_range=(1, 10),
        init_method='half and half'
    )
    
    # 运行遗传编程
    print("运行SymbolicRegressor.fit()...")
    est_gp.fit(X, y)  # 使用原始的3维数据
    print("遗传编程运行完成")
    
    # 提取生成的因子
    factors = gen_better_formula(est_gp, threshold=0.0)
    
    # 确保至少有一些因子
    if not factors:
        print("警告: 未生成符合条件的因子，尝试获取所有生成的因子")
        # 尝试直接从SymbolicRegressor获取所有生成的因子
        if hasattr(est_gp, '_programs') and est_gp._programs:
            last_generation = est_gp._programs[-1]
            factors = []
            for program in last_generation[:10]:  # 最多取10个
                if hasattr(program, 'fitness_'):
                    expr_str = str(program)
                    factors.append({
                        'expression': expr_str,
                        'fitness': program.fitness_
                    })
    
    if not factors:
        raise ValueError("无法生成有效的因子，请检查数据或参数设置")
    
    # 保存生成的因子
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    factors_file = os.path.join(output_dir, f'genetic_factors_{timestamp}.json')
    
    with open(factors_file, 'w', encoding='utf-8') as f:
        json.dump(factors, f, indent=2, ensure_ascii=False)
    
    # 生成README文件
    readme_file = os.path.join(output_dir, 'README.md')
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(f"# 遗传编程生成的Alpha因子\n\n")
        f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"数据源: CSI500数据\n\n")
        f.write(f"生成因子数量: {len(factors)}\n\n")
        f.write("## 因子列表\n\n")
        
        for i, factor in enumerate(factors[:10]):  # 最多显示10个
            f.write(f"### 因子 {i+1}\n")
            f.write(f"- **表达式**: {factor['expression']}\n")
            f.write(f"- **适应度**: {factor['fitness']}\n\n")
    
    # 生成因子代码文件
    for i, factor in enumerate(factors):
        factor_code = "# Genetic-Alpha generated factor " + str(i+1) + "\n"
        factor_code += "import numpy as np\n"
        factor_code += "import pandas as pd\n\n"
        factor_code += "def calculate_factor(stock_data):\n"
        factor_code += "    # Factor generated by Genetic-Alpha\n"
        factor_code += "    # Original expression: " + factor['expression'] + "\n\n"
        
        # 添加基本特征计算
        factor_code += "    # Calculate basic derived features\n"
        factor_code += "    stock_data['returns'] = stock_data['close'] / stock_data['close'].shift(1) - 1\n"
        factor_code += "    stock_data['range'] = stock_data['high'] - stock_data['low']\n"
        factor_code += "    stock_data['hlc3'] = (stock_data['high'] + stock_data['low'] + stock_data['close']) / 3\n\n"
        
        # 基于生成的表达式实现因子计算
        factor_code += "    # Implement factor calculation based on genetic programming result\n"
        
        # 简单处理表达式，使其适合pandas语法
        expr = factor['expression']
        # 替换函数名
        expr = expr.replace('add', '+').replace('sub', '-').replace('mul', '*').replace('div', '/')
        # 替换特征名以匹配stock_data列名
        for feat in ['open', 'high', 'low', 'close', 'volume']:
            expr = expr.replace(feat, f"stock_data['{feat}']")
            
        factor_code += f"    try:\n"
        factor_code += f"        factor_value = {expr}\n"
        # 处理可能的除零问题
        factor_code += "        # Handle potential NaN and inf values\n"
        factor_code += "        factor_value = factor_value.replace([np.inf, -np.inf], np.nan)\n"
        factor_code += "        return factor_value\n"
        factor_code += "    except Exception as e:\n"
        factor_code += "        # Fallback calculation if expression execution fails\n"
        factor_code += "        print(f'Factor calculation error: {e}')\n"
        factor_code += "        return stock_data['returns']  # Default to returns if error occurs\n"
        code_file = os.path.join(output_dir, f'factor_{i+1}.py')
        with open(code_file, 'w', encoding='utf-8') as f:
            f.write(factor_code)
    
    print(f"生成完成！")
    print(f"因子信息保存在: {factors_file}")
    print(f"README保存在: {readme_file}")
    print(f"因子代码文件保存在: {output_dir}\factor_*.py")

if __name__ == "__main__":
    main()