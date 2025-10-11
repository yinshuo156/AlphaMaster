#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主优化管道
整合因子优化、因子选择和投资组合优化的完整流程
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from factor_optimizer import FactorOptimizer
from factor_selector import FactorSelector
from portfolio_optimizer import PortfolioOptimizer

class MainOptimizationPipeline:
    """主优化管道"""
    
    def __init__(self, alpha_pool_path: str = "alpha_pool"):
        self.alpha_pool_path = alpha_pool_path
        self.pipeline_config = {}
        self.results = {}
        
    def set_optimization_config(self, config: dict):
        """设置因子优化配置"""
        self.pipeline_config['optimization'] = config
    
    def set_selection_config(self, config: dict):
        """设置因子选择配置"""
        self.pipeline_config['selection'] = config
    
    def set_portfolio_config(self, config: dict):
        """设置投资组合优化配置"""
        self.pipeline_config['portfolio'] = config
    
    def run_complete_pipeline(self) -> dict:
        """运行完整的优化管道"""
        print("=" * 100)
        print("开始完整的因子优化和投资组合优化管道")
        print("=" * 100)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 步骤1: 因子优化
        print(f"\n{'='*80}")
        print("步骤1: 因子优化")
        print(f"{'='*80}")
        
        optimizer = FactorOptimizer(self.alpha_pool_path)
        optimization_config = self.pipeline_config.get('optimization', {
            'optimize_outliers': True,
            'apply_selection': True,
            'selection_method': 'variance',
            'n_factors': 15,
            'apply_decorrelation': True,
            'decorrelation_method': 'pca',
            'pca_components': 10,
            'explained_variance_threshold': 0.95
        })
        
        optimization_results = optimizer.optimize_all_markets(optimization_config)
        optimizer.save_optimized_factors("portfolios/optimized_factors")
        
        self.results['optimization'] = optimization_results
        
        # 步骤2: 因子选择
        print(f"\n{'='*80}")
        print("步骤2: 因子选择")
        print(f"{'='*80}")
        
        selector = FactorSelector("portfolios/optimized_factors")
        selection_config = self.pipeline_config.get('selection', {
            'statistical_method': 'f_regression',
            'statistical_k': 12,
            'ml_method': 'random_forest',
            'ml_k': 12,
            'final_factors': 8,
            'ic_weight': 0.3,
            'risk_weight': 0.2,
            'diversity_weight': 0.2,
            'statistical_weight': 0.15,
            'ml_weight': 0.15
        })
        
        selection_results = selector.select_all_markets(selection_config)
        selector.save_selected_factors("portfolios/selected_factors")
        
        self.results['selection'] = selection_results
        
        # 步骤3: 投资组合优化
        print(f"\n{'='*80}")
        print("步骤3: 投资组合优化")
        print(f"{'='*80}")
        
        portfolio_optimizer = PortfolioOptimizer("portfolios/selected_factors")
        portfolio_config = self.pipeline_config.get('portfolio', {
            'expected_return_method': 'mean',
            'risk_model_method': 'ledoit_wolf',
            'add_l2_reg': True,
            'l2_gamma': 1,
            'target_volatility': 0.15,
            'target_return': 0.1
        })
        
        portfolio_results = portfolio_optimizer.optimize_all_markets(portfolio_config)
        portfolio_optimizer.generate_optimization_report(portfolio_results, "portfolios/optimization_reports")
        portfolio_optimizer.create_visualizations(portfolio_results, "portfolios/optimization_plots")
        
        self.results['portfolio'] = portfolio_results
        
        print(f"\n{'='*80}")
        print("管道执行完成")
        print(f"{'='*80}")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return self.results
    
    def generate_comprehensive_report(self, output_file: str = "portfolios/comprehensive_optimization_report.md"):
        """生成综合优化报告"""
        print(f"\n{'='*80}")
        print("生成综合优化报告")
        print(f"{'='*80}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 综合因子优化和投资组合优化报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 1. 执行摘要
            f.write("## 执行摘要\n\n")
            f.write("本报告展示了基于质量分析报告的因子优化、因子选择和投资组合优化的完整流程。\n\n")
            
            # 2. 因子优化结果
            f.write("## 因子优化结果\n\n")
            optimization_results = self.results.get('optimization', {})
            
            for market, result in optimization_results.items():
                f.write(f"### {market.upper()}市场\n\n")
                f.write(f"- **原始因子数量**: {result['original_factors']}\n")
                f.write(f"- **优化后因子数量**: {result['optimized_factors']}\n")
                f.write(f"- **原始高相关性对**: {result['original_high_corr_pairs']}\n")
                f.write(f"- **优化后高相关性对**: {result['optimized_high_corr_pairs']}\n")
                f.write(f"- **相关性改善**: {result['original_high_corr_pairs'] - result['optimized_high_corr_pairs']}\n\n")
            
            # 3. 因子选择结果
            f.write("## 因子选择结果\n\n")
            selection_results = self.results.get('selection', {})
            
            for market, result in selection_results.items():
                f.write(f"### {market.upper()}市场\n\n")
                f.write(f"- **总因子数量**: {result['total_factors']}\n")
                f.write(f"- **选择因子数量**: {len(result['selected_factors'])}\n")
                f.write(f"- **选择的因子**: {', '.join(result['selected_factors'])}\n\n")
            
            # 4. 投资组合优化结果
            f.write("## 投资组合优化结果\n\n")
            portfolio_results = self.results.get('portfolio', {})
            
            for market, result in portfolio_results.items():
                f.write(f"### {market.upper()}市场\n\n")
                f.write(f"- **因子数量**: {result['factor_count']}\n")
                
                optimization_results = result['optimization_results']
                for method, method_result in optimization_results.items():
                    if method_result is not None:
                        f.write(f"- **{method}**:\n")
                        f.write(f"  - 期望收益率: {method_result['expected_return']:.4f}\n")
                        f.write(f"  - 波动率: {method_result['volatility']:.4f}\n")
                        f.write(f"  - 夏普比率: {method_result['sharpe_ratio']:.4f}\n")
                f.write("\n")
            
            # 5. 最佳组合推荐
            f.write("## 最佳组合推荐\n\n")
            
            best_portfolios = {}
            for market, result in portfolio_results.items():
                optimization_results = result['optimization_results']
                
                best_sharpe = 0
                best_method = None
                
                for method, method_result in optimization_results.items():
                    if method_result is not None and method_result['sharpe_ratio'] > best_sharpe:
                        best_sharpe = method_result['sharpe_ratio']
                        best_method = method
                
                if best_method:
                    best_portfolios[market] = {
                        'method': best_method,
                        'sharpe_ratio': best_sharpe,
                        'weights': optimization_results[best_method]['weights']
                    }
            
            for market, best_portfolio in best_portfolios.items():
                f.write(f"### {market.upper()}市场最佳组合\n\n")
                f.write(f"- **优化方法**: {best_portfolio['method']}\n")
                f.write(f"- **夏普比率**: {best_portfolio['sharpe_ratio']:.4f}\n")
                f.write("- **因子权重**:\n")
                
                weights = best_portfolio['weights']
                for factor, weight in weights.items():
                    if weight > 0.01:  # 只显示权重大于1%的因子
                        f.write(f"  - {factor}: {weight:.2%}\n")
                f.write("\n")
            
            # 6. 改进建议
            f.write("## 改进建议\n\n")
            f.write("### 短期改进\n")
            f.write("1. **因子相关性优化**: 继续优化因子间的相关性，提高投资组合的多样性\n")
            f.write("2. **风险模型改进**: 尝试不同的风险模型方法，如半协方差、指数协方差等\n")
            f.write("3. **约束条件**: 添加行业、市值等约束条件，提高组合的实用性\n\n")
            
            f.write("### 长期改进\n")
            f.write("1. **因子有效性验证**: 实施更严格的因子有效性测试，包括样本外测试\n")
            f.write("2. **动态优化**: 实现动态因子选择和组合优化，适应市场变化\n")
            f.write("3. **交易成本**: 考虑交易成本对组合表现的影响\n")
            f.write("4. **流动性约束**: 添加流动性约束，确保组合的可交易性\n\n")
            
            # 7. 结论
            f.write("## 结论\n\n")
            f.write("通过完整的因子优化和投资组合优化流程，我们成功：\n\n")
            f.write("1. **优化了因子质量**: 减少了因子间的相关性，提高了因子的多样性\n")
            f.write("2. **筛选了有效因子**: 基于多种指标选择了最具预测能力的因子\n")
            f.write("3. **构建了最优组合**: 使用多种优化方法构建了风险调整后收益最优的投资组合\n\n")
            f.write("这些结果为量化投资策略的开发和实施提供了坚实的基础。\n")
        
        print(f"综合报告已保存到: {output_file}")
    
    def create_summary_statistics(self, output_file: str = "portfolios/summary_statistics.csv"):
        """创建汇总统计"""
        print(f"\n{'='*80}")
        print("创建汇总统计")
        print(f"{'='*80}")
        
        summary_data = []
        
        # 收集所有结果
        optimization_results = self.results.get('optimization', {})
        selection_results = self.results.get('selection', {})
        portfolio_results = self.results.get('portfolio', {})
        
        for market in ['a_share', 'crypto', 'us']:
            # 优化结果
            opt_result = optimization_results.get(market, {})
            
            # 选择结果
            sel_result = selection_results.get(market, {})
            
            # 投资组合结果
            port_result = portfolio_results.get(market, {})
            
            # 找到最佳投资组合
            best_sharpe = 0
            best_method = None
            best_return = 0
            best_volatility = 0
            
            if port_result:
                optimization_results_port = port_result.get('optimization_results', {})
                for method, method_result in optimization_results_port.items():
                    if method_result is not None and method_result['sharpe_ratio'] > best_sharpe:
                        best_sharpe = method_result['sharpe_ratio']
                        best_method = method
                        best_return = method_result['expected_return']
                        best_volatility = method_result['volatility']
            
            summary_data.append({
                'market': market,
                'original_factors': opt_result.get('original_factors', 0),
                'optimized_factors': opt_result.get('optimized_factors', 0),
                'selected_factors': len(sel_result.get('selected_factors', [])),
                'original_high_corr_pairs': opt_result.get('original_high_corr_pairs', 0),
                'optimized_high_corr_pairs': opt_result.get('optimized_high_corr_pairs', 0),
                'best_method': best_method,
                'best_sharpe_ratio': best_sharpe,
                'best_expected_return': best_return,
                'best_volatility': best_volatility
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"汇总统计已保存到: {output_file}")
        
        # 显示汇总统计
        print(f"\n{'='*60}")
        print("汇总统计")
        print(f"{'='*60}")
        print(summary_df.to_string(index=False))

def main():
    """主函数"""
    # 创建主管道
    pipeline = MainOptimizationPipeline()
    
    # 设置配置
    optimization_config = {
        'optimize_outliers': True,
        'apply_selection': True,
        'selection_method': 'variance',
        'n_factors': 15,
        'apply_decorrelation': True,
        'decorrelation_method': 'pca',
        'pca_components': 10,
        'explained_variance_threshold': 0.95
    }
    
    selection_config = {
        'statistical_method': 'f_regression',
        'statistical_k': 12,
        'ml_method': 'random_forest',
        'ml_k': 12,
        'final_factors': 8,
        'ic_weight': 0.3,
        'risk_weight': 0.2,
        'diversity_weight': 0.2,
        'statistical_weight': 0.15,
        'ml_weight': 0.15
    }
    
    portfolio_config = {
        'expected_return_method': 'mean',
        'risk_model_method': 'ledoit_wolf',
        'add_l2_reg': True,
        'l2_gamma': 1,
        'target_volatility': 0.15,
        'target_return': 0.1
    }
    
    pipeline.set_optimization_config(optimization_config)
    pipeline.set_selection_config(selection_config)
    pipeline.set_portfolio_config(portfolio_config)
    
    # 运行完整管道
    results = pipeline.run_complete_pipeline()
    
    # 生成报告
    pipeline.generate_comprehensive_report()
    pipeline.create_summary_statistics()
    
    print(f"\n{'='*100}")
    print("完整优化管道执行完成！")
    print(f"{'='*100}")
    print("生成的文件:")
    print("- portfolios/optimized_factors/: 优化后的因子数据")
    print("- portfolios/selected_factors/: 选择的因子数据")
    print("- portfolios/optimization_reports/: 投资组合优化报告")
    print("- portfolios/optimization_plots/: 优化可视化图表")
    print("- portfolios/comprehensive_optimization_report.md: 综合优化报告")
    print("- portfolios/summary_statistics.csv: 汇总统计")

if __name__ == "__main__":
    main()


