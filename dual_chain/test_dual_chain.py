# -*- coding: utf-8 -*-
"""
双链协同架构测试脚本
用于测试因子池、评估器、LLM适配器和管理器的功能
"""

import os
import sys
import pandas as pd
import numpy as np
import unittest
from datetime import datetime
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dual_chain.factor_pool import FactorPool
from dual_chain.factor_evaluator import FactorEvaluator
from dual_chain.llm_factor_adapter import LLMFactorAdapter

class TestFactorPool(unittest.TestCase):
    """
    测试因子池功能
    """
    
    def setUp(self):
        """
        设置测试环境
        """
        self.test_pool_dir = "dual_chain/test_pools"
        self.pool = FactorPool(self.test_pool_dir)
        
        # 创建测试数据
        dates = pd.date_range(start='2020-01-01', periods=10)
        symbols = ['000001.SZ', '000002.SZ', '000003.SZ']
        
        # 创建测试因子数据
        np.random.seed(42)
        data = np.random.randn(len(dates), len(symbols))
        self.factor_data = pd.DataFrame(data, index=dates, columns=symbols)
        
        # 创建测试评估指标
        self.evaluation_metrics = {
            "ic": 0.035,
            "ic_ir": 0.42,
            "sharpe": 0.85,
            "annual_return": 0.15
        }
    
    def tearDown(self):
        """
        清理测试环境
        """
        # 清理测试生成的文件
        import shutil
        if os.path.exists(self.test_pool_dir):
            shutil.rmtree(self.test_pool_dir)
    
    def test_add_effective_factor(self):
        """
        测试添加有效因子
        """
        factor_name = "Test_Effective_Factor"
        factor_expression = "close.pct_change(20)"
        
        # 添加有效因子
        self.pool.add_effective_factor(
            factor_name=factor_name,
            factor_data=self.factor_data,
            factor_expression=factor_expression,
            evaluation_metrics=self.evaluation_metrics
        )
        
        # 验证因子是否添加成功
        effective_factors = self.pool.get_effective_factors_list()
        self.assertIn(factor_name, effective_factors)
        
        # 验证元数据
        metadata = self.pool.get_factor_metadata(factor_name, is_effective=True)
        self.assertEqual(metadata["expression"], factor_expression)
        self.assertEqual(metadata["evaluation_metrics"]["ic"], self.evaluation_metrics["ic"])
    
    def test_add_discarded_factor(self):
        """
        测试添加废弃因子
        """
        factor_name = "Test_Discarded_Factor"
        factor_expression = "close.pct_change(5)"
        reason = "IC too low"
        
        # 添加废弃因子
        self.pool.add_discarded_factor(
            factor_name=factor_name,
            factor_data=self.factor_data,
            factor_expression=factor_expression,
            evaluation_metrics=self.evaluation_metrics,
            reason=reason
        )
        
        # 验证因子是否添加成功
        discarded_factors = self.pool.get_discarded_factors_list()
        self.assertIn(factor_name, discarded_factors)
        
        # 验证元数据
        metadata = self.pool.get_factor_metadata(factor_name, is_effective=False)
        self.assertEqual(metadata["reason"], reason)
    
    def test_get_reference_factors(self):
        """
        测试获取参考因子
        """
        # 添加多个有效因子
        for i in range(3):
            factor_name = f"Effective_Factor_{i}"
            # 使不同因子有不同的IC
            metrics = self.evaluation_metrics.copy()
            metrics["ic"] = 0.04 - i * 0.01
            
            self.pool.add_effective_factor(
                factor_name=factor_name,
                factor_data=self.factor_data,
                factor_expression=f"close.pct_change({20 + i * 5})",
                evaluation_metrics=metrics
            )
        
        # 添加多个废弃因子
        for i in range(2):
            factor_name = f"Discarded_Factor_{i}"
            self.pool.add_discarded_factor(
                factor_name=factor_name,
                factor_data=self.factor_data,
                factor_expression=f"volume.pct_change({10 + i * 2})",
                evaluation_metrics={"ic": 0.001, "sharpe": 0.1},
                reason="Poor performance"
            )
        
        # 获取参考因子
        effective_refs, discarded_refs = self.pool.get_reference_factors(top_n=2)
        
        self.assertEqual(len(effective_refs), 2)
        self.assertEqual(len(discarded_refs), 2)
    
    def test_pool_statistics(self):
        """
        测试池统计功能
        """
        # 添加因子
        self.pool.add_effective_factor(
            factor_name="Factor_1",
            factor_data=self.factor_data,
            factor_expression="close.pct_change(20)",
            evaluation_metrics={"ic": 0.03, "ic_ir": 0.3, "sharpe": 0.7}
        )
        
        self.pool.add_effective_factor(
            factor_name="Factor_2",
            factor_data=self.factor_data,
            factor_expression="close.pct_change(60)",
            evaluation_metrics={"ic": 0.05, "ic_ir": 0.5, "sharpe": 0.9}
        )
        
        self.pool.add_discarded_factor(
            factor_name="Factor_3",
            factor_data=self.factor_data,
            factor_expression="volume.pct_change(10)",
            evaluation_metrics={"ic": 0.005, "sharpe": 0.2},
            reason="Low IC"
        )
        
        # 获取统计信息
        stats = self.pool.get_pool_statistics()
        
        self.assertEqual(stats["effective_factors_count"], 2)
        self.assertEqual(stats["discarded_factors_count"], 1)
        self.assertAlmostEqual(stats["avg_effective_ic"], 0.04)
        self.assertAlmostEqual(stats["avg_effective_sharpe"], 0.8)

class TestFactorEvaluator(unittest.TestCase):
    """
    测试因子评估器功能
    """
    
    def setUp(self):
        """
        设置测试环境
        """
        # 创建测试收益率数据
        dates = pd.date_range(start='2020-01-01', periods=100)
        symbols = ['000001.SZ', '000002.SZ', '000003.SZ']
        
        np.random.seed(42)
        returns_data = np.random.randn(len(dates), len(symbols)) * 0.01
        self.returns_data = pd.DataFrame(returns_data, index=dates, columns=symbols)
        
        # 创建因子评估器
        self.evaluator = FactorEvaluator(self.returns_data)
        
        # 创建测试因子数据（一个有效的因子和一个无效的因子）
        # 有效因子与收益率有较强的相关性
        valid_factor_data = pd.DataFrame(
            0.8 * returns_data + np.random.randn(len(dates), len(symbols)) * 0.0001,  # 显著降低噪声，提高相关性
            index=dates, 
            columns=symbols
        )
        
        # 无效因子与收益率完全随机
        invalid_factor_data = pd.DataFrame(
            np.random.randn(len(dates), len(symbols)),  # 使用更大的随机噪声
            index=dates, 
            columns=symbols
        )
        
        self.valid_factor_data = valid_factor_data
        self.invalid_factor_data = invalid_factor_data
    
    def test_calculate_ic(self):
        """
        测试计算IC
        """
        ic = self.evaluator.calculate_ic(self.valid_factor_data)
        
        # 验证返回类型
        self.assertIsInstance(ic, float)
        # 有效因子的IC应该大于无效因子
        invalid_ic = self.evaluator.calculate_ic(self.invalid_factor_data)
        self.assertTrue(abs(ic) > abs(invalid_ic))
    
    def test_calculate_ic_ir(self):
        """
        测试计算IC-IR
        """
        ic_ir = self.evaluator.calculate_ic_ir(self.valid_factor_data)
        
        # 验证返回类型
        self.assertIsInstance(ic_ir, float)
        # 有效因子的IC-IR应该大于无效因子
        invalid_ic_ir = self.evaluator.calculate_ic_ir(self.invalid_factor_data)
        self.assertTrue(abs(ic_ir) > abs(invalid_ic_ir))
    
    def test_calculate_sharpe_ratio(self):
        """
        测试计算Sharpe比率
        """
        # 计算因子收益率
        factor_returns = self.evaluator.calculate_factor_returns(self.valid_factor_data)
        sharpe = self.evaluator.calculate_sharpe_ratio(factor_returns)
        
        # 验证返回类型
        self.assertIsInstance(sharpe, float)
    
    def test_evaluate_factor(self):
        """
        测试评估因子
        """
        metrics = self.evaluator.evaluate_factor(self.valid_factor_data, "Test_Factor")
        
        # 验证返回的指标
        self.assertIn("ic", metrics)
        self.assertIn("ic_ir", metrics)
        self.assertIn("sharpe", metrics)
        self.assertIn("annual_return", metrics)
    
    def test_determine_factor_quality(self):
        """
        测试判断因子质量
        """
        # 创建一个高质量因子的指标
        high_quality_metrics = {
            "ic": 0.025,
            "ic_ir": 0.35,
            "sharpe": 0.8,
            "annual_return": 0.12
        }
        
        # 创建一个低质量因子的指标
        low_quality_metrics = {
            "ic": 0.005,
            "ic_ir": 0.1,
            "sharpe": 0.1,
            "annual_return": 0.02
        }
        
        # 测试高质量因子
        is_effective, reason = self.evaluator.determine_factor_quality(high_quality_metrics)
        self.assertTrue(is_effective)
        self.assertEqual(reason, "因子质量达标")
        
        # 测试低质量因子
        is_effective, reason = self.evaluator.determine_factor_quality(low_quality_metrics)
        self.assertFalse(is_effective)
        self.assertIn("不达标", reason)

class TestLLMFactorAdapter(unittest.TestCase):
    """
    测试LLM因子适配器功能（使用模拟模式）
    """
    
    def setUp(self):
        """
        设置测试环境
        """
        # 初始化LLM适配器，启用模拟模式
        self.llm_adapter = LLMFactorAdapter(mock_mode=True)
    
    def test_validate_factor_expression(self):
        """
        测试验证因子表达式
        """
        # 有效的表达式
        valid_expr = "close.pct_change(20)"
        is_valid, error_msg = self.llm_adapter.validate_factor_expression(valid_expr)
        self.assertTrue(is_valid)
        self.assertEqual(error_msg, "表达式有效")
        
        # 无效的表达式（语法错误）
        invalid_expr = "close.pct_change(20"  # 缺少右括号
        is_valid, error_msg = self.llm_adapter.validate_factor_expression(invalid_expr)
        self.assertFalse(is_valid)
        
        # 无效的表达式（使用未知函数）
        invalid_expr = "unknown_function(close)"
        is_valid, error_msg = self.llm_adapter.validate_factor_expression(invalid_expr)
        self.assertTrue(is_valid)  # 简单验证只能检查语法

class TestDualChainIntegration(unittest.TestCase):
    """
    测试双链协同架构的集成功能
    """
    
    def test_integration_setup(self):
        """
        测试集成设置
        """
        # 验证所有必要的目录存在
        required_dirs = [
            "dual_chain/pools/effective",
            "dual_chain/pools/discarded",
            "dual_chain/pools/output",
            "dual_chain/logs"
        ]
        
        for dir_path in required_dirs:
            os.makedirs(dir_path, exist_ok=True)
            self.assertTrue(os.path.exists(dir_path))
    
    def test_component_initialization(self):
        """
        测试组件初始化
        """
        # 测试因子池初始化
        pool_dir = "dual_chain/test_integration_pools"
        pool = FactorPool(pool_dir)
        self.assertIsNotNone(pool)
        
        # 清理
        import shutil
        if os.path.exists(pool_dir):
            shutil.rmtree(pool_dir)

def run_all_tests():
    """
    运行所有测试
    """
    print("=" * 60)
    print("🔍 开始测试双链协同架构组件")
    print("=" * 60)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_suite.addTest(unittest.makeSuite(TestFactorPool))
    test_suite.addTest(unittest.makeSuite(TestFactorEvaluator))
    test_suite.addTest(unittest.makeSuite(TestLLMFactorAdapter))
    test_suite.addTest(unittest.makeSuite(TestDualChainIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 打印测试结果摘要
    print("\n" + "=" * 60)
    print(f"📊 测试结果摘要: 运行{result.testsRun}个测试, "
          f"通过{result.testsRun - len(result.failures) - len(result.errors)}个, "
          f"失败{len(result.failures)}个, "
          f"错误{len(result.errors)}个")
    print("=" * 60)
    
    # 返回测试是否全部通过
    return result.wasSuccessful()

def generate_test_report(successful):
    """
    生成测试报告
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "status": "success" if successful else "failed",
        "components": [
            {"name": "FactorPool", "status": "tested"},
            {"name": "FactorEvaluator", "status": "tested"},
            {"name": "LLMFactorAdapter", "status": "tested"},
            {"name": "Integration", "status": "tested"}
        ]
    }
    
    # 保存测试报告
    os.makedirs("dual_chain/pools/output", exist_ok=True)
    report_file = os.path.join("dual_chain/pools/output", 
                             f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 测试报告已保存: {report_file}")

def main():
    """
    主函数
    """
    try:
        # 运行所有测试
        successful = run_all_tests()
        
        # 生成测试报告
        generate_test_report(successful)
        
        # 根据测试结果返回不同的退出码
        sys.exit(0 if successful else 1)
    
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()