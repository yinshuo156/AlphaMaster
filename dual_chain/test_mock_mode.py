#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单测试mock模式功能
"""

import pandas as pd
import numpy as np
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('simple_test_mock')

class SimpleMockTest:
    """
    简化的mock模式测试类
    """
    def __init__(self, mock_mode=True):
        self.mock_mode = mock_mode
        logger.info(f"🎯 初始化测试类，mock_mode={mock_mode}")
    
    def generate_mock_returns_data(self):
        """
        模拟_get_returns_data方法的mock功能
        """
        if self.mock_mode:
            logger.info("🎯 Mock模式: 生成测试收益率数据")
            dates = pd.date_range(end=pd.Timestamp.now(), periods=60)
            symbols = [f"mock_stock_{i}" for i in range(10)]
            
            # 生成随机收益率数据
            np.random.seed(42)
            random_returns = np.random.normal(0, 0.02, size=(60, 10))
            
            returns_data = pd.DataFrame(random_returns, index=dates, columns=symbols)
            logger.info(f"✅ 生成的收益率数据形状: {returns_data.shape}")
            logger.info(f"📊 收益率数据统计: 均值={returns_data.mean().mean():.6f}, 标准差={returns_data.std().std():.6f}")
            logger.info(f"🔢 非零值比例: {(returns_data != 0).sum().sum() / returns_data.size:.4f}")
            return returns_data
        else:
            return pd.DataFrame()
    
    def generate_mock_factor_data(self):
        """
        模拟因子数据生成
        """
        if self.mock_mode:
            logger.info("🎯 Mock模式: 生成测试因子评估值")
            # 生成有意义的随机IC值（范围在-0.05到0.05之间）
            mock_ic = np.random.uniform(-0.05, 0.05)
            # 生成对应的Sharpe值（与IC方向一致）
            sharpe_sign = 1 if mock_ic > 0 else -1
            mock_sharpe = sharpe_sign * np.random.uniform(0.1, 0.8)
            mock_ic_ir = abs(mock_ic) * np.random.uniform(5, 20)
            
            evaluation_metrics = {
                "ic": mock_ic,
                "ic_ir": mock_ic_ir,
                "sharpe": mock_sharpe
            }
            logger.info(f"📊 生成的测试评估值: IC={mock_ic:.4f}, IC_IR={mock_ic_ir:.4f}, Sharpe={mock_sharpe:.4f}")
            return evaluation_metrics
        else:
            return {"ic": 0, "ic_ir": 0, "sharpe": 0}

def main():
    """
    主测试函数
    """
    logger.info("🔍 开始简化测试mock模式修复...")
    
    # 创建测试实例
    test = SimpleMockTest(mock_mode=True)
    
    # 测试收益率数据生成
    logger.info("📈 测试收益率数据生成...")
    returns_data = test.generate_mock_returns_data()
    logger.info(f"✅ 收益率数据样本:\n{returns_data.head(2)}")
    
    # 测试因子评估值生成
    logger.info("📊 测试因子评估值生成...")
    metrics = test.generate_mock_factor_data()
    logger.info(f"✅ 因子评估结果: {metrics}")
    
    # 验证值不为0
    if metrics["ic"] != 0 and metrics["sharpe"] != 0:
        logger.info("✅ ✓ 成功: mock模式下生成的IC和Sharpe值不为0")
    else:
        logger.error("❌ ✗ 失败: mock模式下生成的IC或Sharpe值仍为0")
    
    logger.info("✅ 简化测试完成")

if __name__ == "__main__":
    main()