#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
投资组合优化器
使用PyPortfolioOpt对筛选后的因子进行组合优化
"""

import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 添加PyPortfolioOpt路径
sys.path.append('PyPortfolioOpt-master')

try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    from pypfopt.hierarchical_portfolio import HRPOpt
    from pypfopt.cla import CLA
    from pypfopt.black_litterman import BlackLittermanModel
    from pypfopt.plotting import plot_efficient_frontier
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"导入PyPortfolioOpt失败: {e}")
    print("请确保PyPortfolioOpt已正确安装")
    sys.exit(1)

class PortfolioOptimizer:
    """投资组合优化器"""
    
    def __init__(self, selected_factors_path: str = "portfolios/selected_factors"):
        self.selected_factors_path = selected_factors_path
        self.markets = ["a_share", "crypto", "us"]
        self.optimized_portfolios = {}
        
    def load_selected_factors(self, market: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载选择的因子数据"""
        factor_file = os.path.join(self.selected_factors_path, f"{market}_selected_factors.csv")
        stats_file = os.path.join(self.selected_factors_path, f"{market}_selected_factors_stats.csv")
        
        if not os.path.exists(factor_file):
            raise FileNotFoundError(f"选择因子文件不存在: {factor_file}")
        if not os.path.exists(stats_file):
            raise FileNotFoundError(f"选择统计文件不存在: {stats_file}")
        
        factor_df = pd.read_csv(factor_file)
        stats_df = pd.read_csv(stats_file)
        
        return factor_df, stats_df
    
    def prepare_factor_returns(self, factor_df: pd.DataFrame, market: str) -> pd.DataFrame:
        """准备因子收益率数据"""
        # 转换为宽格式
        asset_col = 'stock' if market == 'a_share' or market == 'us' else 'crypto'
        
        pivot_df = factor_df.pivot_table(
            index=['date', asset_col],
            columns='factor_name',
            values='factor_value'
        ).reset_index()
        
        # 按日期分组计算因子收益率
        factor_returns = []
        
        for date, group in pivot_df.groupby('date'):
            if len(group) < 3:  # 至少需要3个数据点
                continue
            
            date_returns = {'date': date}
            
            # 获取因子列
            factor_columns = [col for col in group.columns if col not in ['date', asset_col]]
            
            for factor_name in factor_columns:
                factor_values = group[factor_name].values
                
                if len(factor_values) > 1:
                    # 计算因子收益率（使用因子值的标准差作为收益率代理）
                    factor_return = np.std(factor_values)
                    date_returns[factor_name] = factor_return
                else:
                    date_returns[factor_name] = 0
            
            factor_returns.append(date_returns)
        
        returns_df = pd.DataFrame(factor_returns)
        returns_df.set_index('date', inplace=True)
        
        return returns_df
    
    def calculate_expected_returns(self, returns_df: pd.DataFrame, method: str = 'mean') -> pd.Series:
        """计算期望收益率"""
        if method == 'mean':
            return expected_returns.mean_historical_return(returns_df)
        elif method == 'ema':
            return expected_returns.ema_historical_return(returns_df, span=252)
        elif method == 'capm':
            # 简化CAPM，使用市场组合收益率
            market_return = returns_df.mean().mean()
            return expected_returns.capm_return(returns_df, market_return=market_return)
        else:
            return returns_df.mean()
    
    def calculate_risk_model(self, returns_df: pd.DataFrame, method: str = 'sample') -> pd.DataFrame:
        """计算风险模型"""
        if method == 'sample':
            return risk_models.sample_cov(returns_df)
        elif method == 'semicov':
            return risk_models.semicovariance(returns_df)
        elif method == 'exp_cov':
            return risk_models.exp_cov(returns_df)
        elif method == 'ledoit_wolf':
            return risk_models.CovarianceShrinkage(returns_df).ledoit_wolf()
        elif method == 'oracle_approx':
            return risk_models.CovarianceShrinkage(returns_df).oracle_approximating()
        else:
            return risk_models.sample_cov(returns_df)
    
    def optimize_portfolio(self, returns_df: pd.DataFrame, market: str, 
                          optimization_config: Dict) -> Dict:
        """优化单个市场的投资组合"""
        print(f"\n{'='*60}")
        print(f"优化 {market.upper()} 市场投资组合")
        print(f"{'='*60}")
        
        # 计算期望收益率
        mu = self.calculate_expected_returns(
            returns_df, 
            method=optimization_config.get('expected_return_method', 'mean')
        )
        print(f"期望收益率计算完成，因子数量: {len(mu)}")
        
        # 计算风险模型
        S = self.calculate_risk_model(
            returns_df, 
            method=optimization_config.get('risk_model_method', 'ledoit_wolf')
        )
        print(f"风险模型计算完成，协方差矩阵形状: {S.shape}")
        
        # 创建有效前沿
        ef = EfficientFrontier(mu, S)
        
        # 添加L2正则化
        if optimization_config.get('add_l2_reg', True):
            from pypfopt.objective_functions import L2_reg
            ef.add_objective(L2_reg, gamma=optimization_config.get('l2_gamma', 1))
        
        optimization_results = {}
        
        # 1. 最大夏普比率组合
        try:
            print("计算最大夏普比率组合...")
            ef_max_sharpe = EfficientFrontier(mu, S)
            if optimization_config.get('add_l2_reg', True):
                from pypfopt.objective_functions import L2_reg
                ef_max_sharpe.add_objective(L2_reg, gamma=optimization_config.get('l2_gamma', 1))
            
            weights_max_sharpe = ef_max_sharpe.max_sharpe()
            cleaned_weights_max_sharpe = ef_max_sharpe.clean_weights()
            
            performance_max_sharpe = ef_max_sharpe.portfolio_performance(verbose=False)
            
            optimization_results['max_sharpe'] = {
                'weights': cleaned_weights_max_sharpe,
                'expected_return': performance_max_sharpe[0],
                'volatility': performance_max_sharpe[1],
                'sharpe_ratio': performance_max_sharpe[2]
            }
            print(f"最大夏普比率: {performance_max_sharpe[2]:.4f}")
        except Exception as e:
            print(f"最大夏普比率优化失败: {e}")
            optimization_results['max_sharpe'] = None
        
        # 2. 最小波动率组合
        try:
            print("计算最小波动率组合...")
            ef_min_vol = EfficientFrontier(mu, S)
            if optimization_config.get('add_l2_reg', True):
                from pypfopt.objective_functions import L2_reg
                ef_min_vol.add_objective(L2_reg, gamma=optimization_config.get('l2_gamma', 1))
            
            weights_min_vol = ef_min_vol.min_volatility()
            cleaned_weights_min_vol = ef_min_vol.clean_weights()
            
            performance_min_vol = ef_min_vol.portfolio_performance(verbose=False)
            
            optimization_results['min_volatility'] = {
                'weights': cleaned_weights_min_vol,
                'expected_return': performance_min_vol[0],
                'volatility': performance_min_vol[1],
                'sharpe_ratio': performance_min_vol[2]
            }
            print(f"最小波动率: {performance_min_vol[1]:.4f}")
        except Exception as e:
            print(f"最小波动率优化失败: {e}")
            optimization_results['min_volatility'] = None
        
        # 3. 高效风险组合
        try:
            print("计算高效风险组合...")
            target_volatility = optimization_config.get('target_volatility', 0.15)
            
            ef_efficient_risk = EfficientFrontier(mu, S)
            if optimization_config.get('add_l2_reg', True):
                from pypfopt.objective_functions import L2_reg
                ef_efficient_risk.add_objective(L2_reg, gamma=optimization_config.get('l2_gamma', 1))
            
            weights_efficient_risk = ef_efficient_risk.efficient_risk(target_volatility=target_volatility)
            cleaned_weights_efficient_risk = ef_efficient_risk.clean_weights()
            
            performance_efficient_risk = ef_efficient_risk.portfolio_performance(verbose=False)
            
            optimization_results['efficient_risk'] = {
                'weights': cleaned_weights_efficient_risk,
                'expected_return': performance_efficient_risk[0],
                'volatility': performance_efficient_risk[1],
                'sharpe_ratio': performance_efficient_risk[2],
                'target_volatility': target_volatility
            }
            print(f"高效风险组合波动率: {performance_efficient_risk[1]:.4f}")
        except Exception as e:
            print(f"高效风险优化失败: {e}")
            optimization_results['efficient_risk'] = None
        
        # 4. 高效收益组合
        try:
            print("计算高效收益组合...")
            target_return = optimization_config.get('target_return', 0.1)
            
            ef_efficient_return = EfficientFrontier(mu, S)
            if optimization_config.get('add_l2_reg', True):
                from pypfopt.objective_functions import L2_reg
                ef_efficient_return.add_objective(L2_reg, gamma=optimization_config.get('l2_gamma', 1))
            
            weights_efficient_return = ef_efficient_return.efficient_return(target_return=target_return)
            cleaned_weights_efficient_return = ef_efficient_return.clean_weights()
            
            performance_efficient_return = ef_efficient_return.portfolio_performance(verbose=False)
            
            optimization_results['efficient_return'] = {
                'weights': cleaned_weights_efficient_return,
                'expected_return': performance_efficient_return[0],
                'volatility': performance_efficient_return[1],
                'sharpe_ratio': performance_efficient_return[2],
                'target_return': target_return
            }
            print(f"高效收益组合收益率: {performance_efficient_return[0]:.4f}")
        except Exception as e:
            print(f"高效收益优化失败: {e}")
            optimization_results['efficient_return'] = None
        
        # 5. 分层风险平价组合
        try:
            print("计算分层风险平价组合...")
            hrp = HRPOpt(returns_df)
            weights_hrp = hrp.optimize()
            
            # 计算HRP组合表现
            hrp_returns = (returns_df * pd.Series(weights_hrp)).sum(axis=1)
            hrp_expected_return = hrp_returns.mean()
            hrp_volatility = hrp_returns.std()
            hrp_sharpe = hrp_expected_return / hrp_volatility if hrp_volatility > 0 else 0
            
            optimization_results['hrp'] = {
                'weights': weights_hrp,
                'expected_return': hrp_expected_return,
                'volatility': hrp_volatility,
                'sharpe_ratio': hrp_sharpe
            }
            print(f"HRP夏普比率: {hrp_sharpe:.4f}")
        except Exception as e:
            print(f"HRP优化失败: {e}")
            optimization_results['hrp'] = None
        
        # 6. 临界线算法
        try:
            print("计算临界线算法组合...")
            cla = CLA(mu, S)
            cla.max_sharpe()
            
            weights_cla = cla.weights
            performance_cla = cla.portfolio_performance(verbose=False)
            
            # 将numpy数组转换为字典格式，使用原始因子名称
            factor_names = returns_df.columns.tolist()
            weights_cla_dict = {factor_names[i]: weights_cla[i] for i in range(len(weights_cla))}
            
            optimization_results['cla'] = {
                'weights': weights_cla_dict,
                'expected_return': performance_cla[0],
                'volatility': performance_cla[1],
                'sharpe_ratio': performance_cla[2]
            }
            print(f"CLA夏普比率: {performance_cla[2]:.4f}")
        except Exception as e:
            print(f"CLA优化失败: {e}")
            optimization_results['cla'] = None
        
        return {
            'market': market,
            'factor_count': len(mu),
            'optimization_results': optimization_results,
            'expected_returns': mu,
            'covariance_matrix': S
        }
    
    def optimize_all_markets(self, optimization_config: Dict) -> Dict:
        """优化所有市场的投资组合"""
        print("=" * 80)
        print("开始投资组合优化")
        print("=" * 80)
        
        all_results = {}
        
        for market in self.markets:
            try:
                # 加载选择的因子
                factor_df, stats_df = self.load_selected_factors(market)
                
                # 准备因子收益率
                returns_df = self.prepare_factor_returns(factor_df, market)
                print(f"{market}市场因子收益率数据形状: {returns_df.shape}")
                
                # 优化投资组合
                result = self.optimize_portfolio(returns_df, market, optimization_config)
                all_results[market] = result
                self.optimized_portfolios[market] = result
                
            except Exception as e:
                print(f"❌ 优化{market}市场投资组合时出错: {e}")
                continue
        
        return all_results
    
    def generate_optimization_report(self, results: Dict, output_path: str = "portfolios/optimization_reports"):
        """生成优化报告"""
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # 创建汇总报告
        summary_data = []
        
        for market, result in results.items():
            optimization_results = result['optimization_results']
            
            for method, method_result in optimization_results.items():
                if method_result is not None:
                    summary_data.append({
                        'market': market,
                        'method': method,
                        'expected_return': method_result['expected_return'],
                        'volatility': method_result['volatility'],
                        'sharpe_ratio': method_result['sharpe_ratio'],
                        'factor_count': result['factor_count']
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_path, "portfolio_optimization_summary.csv")
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        
        # 为每个市场生成详细报告
        for market, result in results.items():
            market_report = []
            optimization_results = result['optimization_results']
            
            for method, method_result in optimization_results.items():
                if method_result is not None:
                    weights = method_result['weights']
                    
                    if isinstance(weights, dict):
                        for factor, weight in weights.items():
                            if weight > 0.001:  # 只记录权重大于0.1%的因子
                                market_report.append({
                                    'method': method,
                                    'factor': factor,
                                    'weight': weight,
                                    'expected_return': method_result['expected_return'],
                                    'volatility': method_result['volatility'],
                                    'sharpe_ratio': method_result['sharpe_ratio']
                                })
                    else:
                        # 处理numpy数组格式的权重
                        factor_names = list(method_result.get('factor_names', [f'Factor_{i}' for i in range(len(weights))]))
                        for i, weight in enumerate(weights):
                            if weight > 0.001:  # 只记录权重大于0.1%的因子
                                market_report.append({
                                    'method': method,
                                    'factor': factor_names[i] if i < len(factor_names) else f'Factor_{i}',
                                    'weight': weight,
                                    'expected_return': method_result['expected_return'],
                                    'volatility': method_result['volatility'],
                                    'sharpe_ratio': method_result['sharpe_ratio']
                                })
            
            market_df = pd.DataFrame(market_report)
            market_file = os.path.join(output_path, f"{market}_portfolio_weights.csv")
            market_df.to_csv(market_file, index=False, encoding='utf-8-sig')
        
        print(f"优化报告已保存到: {output_path}")
    
    def create_visualizations(self, results: Dict, output_path: str = "portfolios/optimization_plots"):
        """创建可视化图表"""
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 各市场夏普比率对比
        fig, ax = plt.subplots(figsize=(12, 8))
        
        markets = []
        methods = []
        sharpe_ratios = []
        
        for market, result in results.items():
            optimization_results = result['optimization_results']
            
            for method, method_result in optimization_results.items():
                if method_result is not None:
                    markets.append(market)
                    methods.append(method)
                    sharpe_ratios.append(method_result['sharpe_ratio'])
        
        # 创建DataFrame用于绘图
        plot_df = pd.DataFrame({
            'market': markets,
            'method': methods,
            'sharpe_ratio': sharpe_ratios
        })
        
        # 绘制条形图
        sns.barplot(data=plot_df, x='market', y='sharpe_ratio', hue='method', ax=ax)
        ax.set_title('各市场不同优化方法的夏普比率对比', fontsize=16)
        ax.set_xlabel('市场', fontsize=12)
        ax.set_ylabel('夏普比率', fontsize=12)
        ax.legend(title='优化方法', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'sharpe_ratio_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 风险收益散点图
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = {'a_share': 'red', 'crypto': 'blue', 'us': 'green'}
        
        for market, result in results.items():
            optimization_results = result['optimization_results']
            
            volatilities = []
            returns = []
            
            for method, method_result in optimization_results.items():
                if method_result is not None:
                    volatilities.append(method_result['volatility'])
                    returns.append(method_result['expected_return'])
            
            if volatilities and returns:
                ax.scatter(volatilities, returns, c=colors.get(market, 'black'), 
                          label=market, s=100, alpha=0.7)
        
        ax.set_xlabel('波动率', fontsize=12)
        ax.set_ylabel('期望收益率', fontsize=12)
        ax.set_title('风险收益散点图', fontsize=16)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'risk_return_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可视化图表已保存到: {output_path}")

def main():
    """主函数"""
    # 优化配置
    optimization_config = {
        'expected_return_method': 'mean',        # 期望收益率计算方法
        'risk_model_method': 'ledoit_wolf',      # 风险模型方法
        'add_l2_reg': True,                      # 添加L2正则化
        'l2_gamma': 1,                           # L2正则化参数
        'target_volatility': 0.15,               # 目标波动率
        'target_return': 0.1                     # 目标收益率
    }
    
    # 创建优化器
    optimizer = PortfolioOptimizer()
    
    # 执行优化
    results = optimizer.optimize_all_markets(optimization_config)
    
    # 生成报告
    optimizer.generate_optimization_report(results)
    
    # 创建可视化
    optimizer.create_visualizations(results)
    
    # 显示优化结果
    print(f"\n{'='*80}")
    print("投资组合优化结果总结")
    print(f"{'='*80}")
    
    for market, result in results.items():
        print(f"\n{market.upper()}市场:")
        print(f"  因子数量: {result['factor_count']}")
        
        optimization_results = result['optimization_results']
        for method, method_result in optimization_results.items():
            if method_result is not None:
                print(f"  {method}:")
                print(f"    期望收益率: {method_result['expected_return']:.4f}")
                print(f"    波动率: {method_result['volatility']:.4f}")
                print(f"    夏普比率: {method_result['sharpe_ratio']:.4f}")

if __name__ == "__main__":
    main()
