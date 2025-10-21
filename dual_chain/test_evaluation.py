#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本：检查因子评估问题
用于诊断为什么所有因子的IC和Sharpe值都为0
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_evaluation')

def test_data_loading():
    """测试数据加载和基本统计"""
    logger.info("🔍 开始测试数据加载")
    
    # 尝试直接加载数据并检查
    data_path = os.path.join(os.path.dirname(__file__), 'data/a_share')
    if not os.path.exists(data_path):
        # 使用默认路径
        data_path = 'data/a_share'
    
    logger.info(f"📂 测试数据路径: {data_path}")
    
    # 尝试读取一些CSV文件
    import glob
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    logger.info(f"📄 找到 {len(csv_files)} 个CSV文件")
    
    if csv_files:
        # 读取第一个文件进行测试
        test_file = csv_files[0]
        logger.info(f"🔍 测试文件: {test_file}")
        
        try:
            df = pd.read_csv(test_file)
            logger.info(f"✅ 成功读取文件，形状: {df.shape}")
            logger.info(f"📋 数据列: {list(df.columns)}")
            logger.info(f"📊 数据摘要:\n{df.describe()}")
            
            # 检查日期列
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                logger.info(f"📅 日期范围: {df['date'].min()} 到 {df['date'].max()}")
            
            # 检查价格列
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    logger.info(f"📈 {col} 列统计: 均值={df[col].mean():.2f}, 标准差={df[col].std():.2f}")
                    # 检查是否有变化
                    if df[col].std() == 0:
                        logger.warning(f"⚠️  {col} 列没有变化，所有值相同")
            
            # 计算收益率
            if 'close' in df.columns:
                df['returns'] = df['close'].pct_change()
                logger.info(f"📊 收益率统计: 均值={df['returns'].mean():.6f}, 标准差={df['returns'].std():.6f}")
                logger.info(f"📊 非零收益率数量: {(df['returns'].fillna(0) != 0).sum()}")
                
                # 如果收益率全为0，记录警告
                if (df['returns'].fillna(0) == 0).all():
                    logger.warning("❌ 所有收益率都为0，这会导致IC和Sharpe值为0")
            
        except Exception as e:
            logger.error(f"❌ 读取测试文件失败: {e}")
    else:
        logger.warning(f"⚠️  在路径 {data_path} 下未找到CSV文件")

def test_factor_calculation():
    """测试因子计算逻辑"""
    logger.info("🔍 开始测试因子计算逻辑")
    
    # 创建简单的测试数据
    dates = pd.date_range(start='2024-01-01', periods=20)
    stocks = ['A', 'B', 'C', 'D', 'E']
    
    # 创建价格数据
    np.random.seed(42)
    price_data = pd.DataFrame(
        np.random.rand(len(dates), len(stocks)),
        index=dates,
        columns=stocks
    )
    
    logger.info(f"📊 创建测试价格数据，形状: {price_data.shape}")
    logger.info(f"📊 价格数据示例:\n{price_data.head()}")
    
    # 创建收益率数据
    returns_data = price_data.pct_change().fillna(0)
    logger.info(f"📊 收益率数据示例:\n{returns_data.head()}")
    
    # 创建一个简单因子 (价格动量)
    factor_data = price_data.pct_change(5)
    logger.info(f"📊 因子数据示例:\n{factor_data.head()}")
    
    # 测试IC计算
    daily_ics = []
    for date in dates[6:]:  # 跳过前5天的NA值
        if date in factor_data.index and date in returns_data.index:
            # 获取当日的因子值和次日收益率
            factor_values = factor_data.loc[date].dropna()
            ret_values = returns_data.shift(-1).loc[date].dropna()
            
            # 对齐股票
            common_stocks = factor_values.index.intersection(ret_values.index)
            if len(common_stocks) >= 2:
                factor_values = factor_values.loc[common_stocks]
                ret_values = ret_values.loc[common_stocks]
                
                # 计算Spearman相关系数
                if factor_values.std() > 0 and ret_values.std() > 0:
                    ic = factor_values.rank().corr(ret_values.rank(), method='spearman')
                    daily_ics.append(ic)
                    logger.info(f"📅 日期 {date}: IC = {ic:.4f}")
    
    if daily_ics:
        avg_ic = np.mean(daily_ics)
        logger.info(f"📊 平均IC: {avg_ic:.4f}")
    else:
        logger.warning("⚠️  无法计算IC，数据不足或有问题")

def test_config_file():
    """测试配置文件加载"""
    logger.info("🔍 开始测试配置文件")
    
    config_path = 'standardized_eval_config.json'
    if os.path.exists(config_path):
        try:
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"✅ 成功加载配置文件，内容: {config}")
            
            # 检查mock-mode设置
            if 'mock-mode' in config and config['mock-mode'] is True:
                logger.warning("⚠️  发现关键问题: mock-mode 设置为 True，这会导致使用模拟数据或默认值")
                logger.warning("⚠️  这很可能是所有因子IC和Sharpe值都为0的原因")
            
        except Exception as e:
            logger.error(f"❌ 加载配置文件失败: {e}")
    else:
        logger.warning(f"⚠️  配置文件 {config_path} 不存在")
        # 尝试创建配置文件，关闭mock模式
        try:
            default_config = {
                'data-path': 'c:/Users/Administrator/Desktop/alpha-master/data/a_share/csi500data',
                'factors-file': 'c:/Users/Administrator/Desktop/alpha-master/a_factor_generate/all_factors_expressions.json',
                'complementary-count': 3,
                'llm-provider': 'dashscope',
                'llm-model': 'qwen-turbo',
                'llm-api-key': 'sk-429ed6e6cfa347ddb5005e178edd3f90',
                'run-standardized-eval': True,
                'mock-mode': False  # 关闭mock模式
            }
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"✅ 创建了新的配置文件，关闭了mock模式: {config_path}")
        except Exception as e:
            logger.error(f"❌ 创建配置文件失败: {e}")

def test_factors_file():
    """测试因子文件加载"""
    logger.info("🔍 开始测试因子文件")
    
    factors_path = 'c:/Users/Administrator/Desktop/alpha-master/a_factor_generate/all_factors_expressions.json'
    if os.path.exists(factors_path):
        try:
            import json
            with open(factors_path, 'r', encoding='utf-8') as f:
                factors = json.load(f)
            logger.info(f"✅ 成功加载因子文件，共 {len(factors)} 个因子")
            # 显示前2个因子作为示例
            if isinstance(factors, list) and factors:
                for i, factor in enumerate(factors[:2]):
                    logger.info(f"📊 因子 {i+1}: {factor}")
            else:
                logger.warning(f"⚠️  因子文件格式异常，不是预期的列表格式: {type(factors)}")
        except Exception as e:
            logger.error(f"❌ 加载因子文件失败: {e}")
    else:
        logger.warning(f"⚠️  因子文件 {factors_path} 不存在")

def test_mock_mode_impact():
    """测试mock模式的影响"""
    logger.info("🔍 开始测试mock模式影响")
    
    # 分析mock模式对评估结果的影响
    logger.info("📊 mock模式通常会导致:")
    logger.info("   1. 使用模拟数据而不是真实数据")
    logger.info("   2. 返回默认评估值（如IC=0.0000，Sharpe=0.0000）")
    logger.info("   3. 跳过实际的因子计算和回测")
    logger.info("📋 建议操作:")
    logger.info("   1. 关闭mock-mode，设置为False")
    logger.info("   2. 确保数据路径正确且包含有效数据")
    logger.info("   3. 重新运行评估流程")

def main():
    """主函数"""
    logger.info("🚀 开始因子评估问题诊断")
    
    # 测试配置文件（优先检查，因为mock-mode是关键问题）
    test_config_file()
    
    # 测试mock模式影响分析
    test_mock_mode_impact()
    
    # 测试数据加载
    test_data_loading()
    
    # 测试因子计算逻辑
    test_factor_calculation()
    
    # 测试因子文件
    test_factors_file()
    
    logger.info("🎉 诊断完成")
    logger.info("📋 总结: IC和Sharpe值为0.0000的最可能原因是mock-mode=True")
    logger.info("📋 解决方案: 修改配置文件，将mock-mode设置为False后重新运行")

if __name__ == "__main__":
    main()