#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子筛选器
基于多种指标对优化后的因子进行筛选
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
import warnings
warnings.filterwarnings('ignore')

class FactorSelector:
    """因子筛选器"""
    
    def __init__(self, optimized_factors_path: str = "portfolios/optimized_factors"):
        self.optimized_factors_path = optimized_factors_path
        self.markets = ["a_share", "crypto", "us"]
        self.selected_factors = {}
        
    def load_optimized_factors(self, market: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载优化后的因子数据"""
        factor_file = os.path.join(self.optimized_factors_path, f"{market}_optimized_factors.csv")
        stats_file = os.path.join(self.optimized_factors_path, f"{market}_optimized_factors_stats.csv")
        
        if not os.path.exists(factor_file):
            raise FileNotFoundError(f"优化因子文件不存在: {factor_file}")
        if not os.path.exists(stats_file):
            raise FileNotFoundError(f"优化统计文件不存在: {stats_file}")
        
        factor_df = pd.read_csv(factor_file)
        stats_df = pd.read_csv(stats_file)
        
        return factor_df, stats_df
    
    def convert_to_wide_format(self, factor_df: pd.DataFrame, market: str) -> pd.DataFrame:
        """将因子数据转换为宽格式"""
        asset_col = 'stock' if market == 'a_share' or market == 'us' else 'crypto'
        
        pivot_df = factor_df.pivot_table(
            index=['date', asset_col],
            columns='factor_name',
            values='factor_value'
        ).reset_index()
        
        return pivot_df
    
    def calculate_ic_analysis(self, factor_matrix: pd.DataFrame, market: str) -> Dict[str, float]:
        """计算信息系数(IC)分析"""
        # 获取因子列
        factor_columns = [col for col in factor_matrix.columns if col not in ['date', 'stock', 'crypto']]
        
        # 按日期分组计算IC
        ic_results = {}
        
        for factor_name in factor_columns:
            ic_values = []
            
            # 按日期分组
            for date, group in factor_matrix.groupby('date'):
                if len(group) < 5:  # 至少需要5个数据点
                    continue
                
                factor_values = group[factor_name].values
                # 这里使用因子值作为收益率代理（实际应用中应该使用真实收益率）
                returns = group[factor_name].values  # 简化处理
                
                # 计算IC（Spearman相关系数）
                if len(factor_values) > 1 and len(returns) > 1:
                    ic = np.corrcoef(factor_values, returns)[0, 1]
                    if not np.isnan(ic):
                        ic_values.append(ic)
            
            if ic_values:
                ic_results[factor_name] = {
                    'mean_ic': np.mean(ic_values),
                    'std_ic': np.std(ic_values),
                    'ic_ir': np.mean(ic_values) / (np.std(ic_values) + 1e-8),  # IC信息比率
                    'ic_stability': 1 / (np.std(ic_values) + 1e-8)  # IC稳定性
                }
            else:
                ic_results[factor_name] = {
                    'mean_ic': 0,
                    'std_ic': 0,
                    'ic_ir': 0,
                    'ic_stability': 0
                }
        
        return ic_results
    
    def calculate_factor_returns(self, factor_matrix: pd.DataFrame) -> pd.DataFrame:
        """计算因子收益率"""
        # 获取因子列
        factor_columns = [col for col in factor_matrix.columns if col not in ['date', 'stock', 'crypto']]
        
        # 按日期分组计算因子收益率
        factor_returns = []
        
        for date, group in factor_matrix.groupby('date'):
            if len(group) < 5:
                continue
            
            date_returns = {'date': date}
            
            for factor_name in factor_columns:
                factor_values = group[factor_name].values
                
                # 计算因子收益率（简化：使用因子值的分位数作为收益率代理）
                if len(factor_values) > 1:
                    # 使用因子值的标准差作为收益率代理
                    factor_return = np.std(factor_values)
                    date_returns[factor_name] = factor_return
                else:
                    date_returns[factor_name] = 0
            
            factor_returns.append(date_returns)
        
        return pd.DataFrame(factor_returns)
    
    def calculate_risk_metrics(self, factor_matrix: pd.DataFrame) -> Dict[str, Dict]:
        """计算风险指标"""
        factor_columns = [col for col in factor_matrix.columns if col not in ['date', 'stock', 'crypto']]
        
        risk_metrics = {}
        
        for factor_name in factor_columns:
            factor_values = factor_matrix[factor_name].dropna()
            
            if len(factor_values) > 1:
                # 计算各种风险指标
                volatility = np.std(factor_values)
                sharpe_ratio = np.mean(factor_values) / (volatility + 1e-8)
                
                # 计算最大回撤
                cumulative = (1 + factor_values).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = drawdown.min()
                
                # 计算VaR (95%)
                var_95 = np.percentile(factor_values, 5)
                
                # 计算偏度和峰度
                skewness = factor_values.skew()
                kurtosis = factor_values.kurtosis()
                
                risk_metrics[factor_name] = {
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'var_95': var_95,
                    'skewness': skewness,
                    'kurtosis': kurtosis
                }
            else:
                risk_metrics[factor_name] = {
                    'volatility': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'var_95': 0,
                    'skewness': 0,
                    'kurtosis': 0
                }
        
        return risk_metrics
    
    def calculate_diversity_score(self, factor_matrix: pd.DataFrame) -> Dict[str, float]:
        """计算因子多样性得分"""
        factor_columns = [col for col in factor_matrix.columns if col not in ['date', 'stock', 'crypto']]
        
        # 计算因子间相关性
        correlation_matrix = factor_matrix[factor_columns].corr()
        
        diversity_scores = {}
        
        for factor_name in factor_columns:
            # 计算与其他因子的平均相关性
            other_factors = [col for col in factor_columns if col != factor_name]
            if other_factors:
                avg_correlation = correlation_matrix.loc[factor_name, other_factors].abs().mean()
                diversity_score = 1 - avg_correlation  # 相关性越低，多样性越高
            else:
                diversity_score = 1
            
            diversity_scores[factor_name] = diversity_score
        
        return diversity_scores
    
    def apply_statistical_selection(self, factor_matrix: pd.DataFrame, method: str = 'f_regression', 
                                  k: int = 10) -> List[str]:
        """应用统计方法进行因子选择"""
        factor_columns = [col for col in factor_matrix.columns if col not in ['date', 'stock', 'crypto']]
        
        # 准备数据
        X = factor_matrix[factor_columns].fillna(0)
        
        # 创建目标变量（使用因子值的组合作为目标）
        y = X.mean(axis=1)  # 简化：使用所有因子的平均值作为目标
        
        if method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=k)
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
        else:
            return factor_columns[:k]
        
        selector.fit(X, y)
        selected_indices = selector.get_support(indices=True)
        selected_factors = [factor_columns[i] for i in selected_indices]
        
        return selected_factors
    
    def apply_ml_based_selection(self, factor_matrix: pd.DataFrame, method: str = 'random_forest', 
                               k: int = 10) -> List[str]:
        """应用机器学习方法进行因子选择"""
        factor_columns = [col for col in factor_matrix.columns if col not in ['date', 'stock', 'crypto']]
        
        # 准备数据
        X = factor_matrix[factor_columns].fillna(0)
        y = X.mean(axis=1)  # 简化目标变量
        
        if method == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # 获取特征重要性
            feature_importance = model.feature_importances_
            importance_df = pd.DataFrame({
                'factor': factor_columns,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            selected_factors = importance_df.head(k)['factor'].tolist()
        
        elif method == 'lasso':
            model = LassoCV(cv=5, random_state=42)
            model.fit(X, y)
            
            # 获取非零系数的特征
            non_zero_coef = np.abs(model.coef_) > 1e-6
            selected_factors = [factor_columns[i] for i, selected in enumerate(non_zero_coef) if selected]
            
            # 如果选择的因子太少，按系数绝对值排序选择前k个
            if len(selected_factors) < k:
                coef_df = pd.DataFrame({
                    'factor': factor_columns,
                    'coef': np.abs(model.coef_)
                }).sort_values('coef', ascending=False)
                selected_factors = coef_df.head(k)['factor'].tolist()
        
        else:
            selected_factors = factor_columns[:k]
        
        return selected_factors
    
    def apply_comprehensive_selection(self, factor_matrix: pd.DataFrame, market: str, 
                                    selection_config: Dict) -> Dict:
        """应用综合因子选择"""
        print(f"\n{'='*50}")
        print(f"{market.upper()}市场因子选择")
        print(f"{'='*50}")
        
        # 计算各种指标
        print("计算IC分析...")
        ic_analysis = self.calculate_ic_analysis(factor_matrix, market)
        
        print("计算风险指标...")
        risk_metrics = self.calculate_risk_metrics(factor_matrix)
        
        print("计算多样性得分...")
        diversity_scores = self.calculate_diversity_score(factor_matrix)
        
        # 应用统计选择
        print("应用统计选择...")
        statistical_selected = self.apply_statistical_selection(
            factor_matrix, 
            method=selection_config.get('statistical_method', 'f_regression'),
            k=selection_config.get('statistical_k', 10)
        )
        
        # 应用机器学习选择
        print("应用机器学习选择...")
        ml_selected = self.apply_ml_based_selection(
            factor_matrix,
            method=selection_config.get('ml_method', 'random_forest'),
            k=selection_config.get('ml_k', 10)
        )
        
        # 综合评分
        factor_columns = [col for col in factor_matrix.columns if col not in ['date', 'stock', 'crypto']]
        
        factor_scores = {}
        for factor_name in factor_columns:
            score = 0
            
            # IC得分
            if factor_name in ic_analysis:
                ic_score = abs(ic_analysis[factor_name]['mean_ic']) * ic_analysis[factor_name]['ic_ir']
                score += ic_score * selection_config.get('ic_weight', 0.3)
            
            # 风险得分
            if factor_name in risk_metrics:
                risk_score = risk_metrics[factor_name]['sharpe_ratio'] / (abs(risk_metrics[factor_name]['max_drawdown']) + 1e-8)
                score += risk_score * selection_config.get('risk_weight', 0.2)
            
            # 多样性得分
            if factor_name in diversity_scores:
                score += diversity_scores[factor_name] * selection_config.get('diversity_weight', 0.2)
            
            # 统计选择得分
            if factor_name in statistical_selected:
                score += selection_config.get('statistical_weight', 0.15)
            
            # 机器学习选择得分
            if factor_name in ml_selected:
                score += selection_config.get('ml_weight', 0.15)
            
            factor_scores[factor_name] = score
        
        # 选择得分最高的因子
        n_final_factors = selection_config.get('final_factors', 8)
        sorted_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
        final_selected = [factor for factor, score in sorted_factors[:n_final_factors]]
        
        print(f"最终选择因子数量: {len(final_selected)}")
        print(f"选择的因子: {final_selected}")
        
        return {
            'market': market,
            'total_factors': len(factor_columns),
            'selected_factors': final_selected,
            'factor_scores': factor_scores,
            'ic_analysis': ic_analysis,
            'risk_metrics': risk_metrics,
            'diversity_scores': diversity_scores,
            'statistical_selected': statistical_selected,
            'ml_selected': ml_selected
        }
    
    def select_all_markets(self, selection_config: Dict) -> Dict:
        """选择所有市场的因子"""
        print("=" * 80)
        print("开始因子选择")
        print("=" * 80)
        
        all_results = {}
        
        for market in self.markets:
            try:
                # 加载优化后的因子
                factor_df, stats_df = self.load_optimized_factors(market)
                factor_matrix = self.convert_to_wide_format(factor_df, market)
                
                # 应用综合选择
                result = self.apply_comprehensive_selection(factor_matrix, market, selection_config)
                all_results[market] = result
                
                # 保存选择的因子
                selected_factors = result['selected_factors']
                selected_data = factor_df[factor_df['factor_name'].isin(selected_factors)]
                self.selected_factors[market] = selected_data
                
            except Exception as e:
                print(f"❌ 选择{market}市场因子时出错: {e}")
                continue
        
        return all_results
    
    def save_selected_factors(self, output_path: str = "portfolios/selected_factors"):
        """保存选择的因子"""
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        for market, factor_data in self.selected_factors.items():
            # 保存因子数据
            factor_file = os.path.join(output_path, f"{market}_selected_factors.csv")
            factor_data.to_csv(factor_file, index=False, encoding='utf-8-sig')
            
            # 生成统计信息
            stats_data = []
            for factor_name in factor_data['factor_name'].unique():
                factor_values = factor_data[factor_data['factor_name'] == factor_name]['factor_value']
                clean_values = factor_values.dropna()
                clean_values = clean_values[np.isfinite(clean_values)]
                
                if len(clean_values) > 0:
                    stats_data.append({
                        'factor_name': factor_name,
                        'count': len(clean_values),
                        'mean': np.mean(clean_values),
                        'std': np.std(clean_values),
                        'min': np.min(clean_values),
                        'max': np.max(clean_values)
                    })
            
            stats_df = pd.DataFrame(stats_data)
            stats_file = os.path.join(output_path, f"{market}_selected_factors_stats.csv")
            stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
            
            print(f"保存{market}市场选择因子: {factor_file}")
            print(f"保存{market}市场统计信息: {stats_file}")

def main():
    """主函数"""
    # 选择配置
    selection_config = {
        'statistical_method': 'f_regression',  # 统计方法: f_regression, mutual_info
        'statistical_k': 12,                   # 统计选择因子数量
        'ml_method': 'random_forest',          # 机器学习方法: random_forest, lasso
        'ml_k': 12,                            # 机器学习选择因子数量
        'final_factors': 8,                    # 最终选择因子数量
        'ic_weight': 0.3,                      # IC权重
        'risk_weight': 0.2,                    # 风险权重
        'diversity_weight': 0.2,               # 多样性权重
        'statistical_weight': 0.15,            # 统计选择权重
        'ml_weight': 0.15                      # 机器学习选择权重
    }
    
    # 创建选择器
    selector = FactorSelector()
    
    # 执行选择
    results = selector.select_all_markets(selection_config)
    
    # 保存结果
    selector.save_selected_factors()
    
    # 显示选择结果
    print(f"\n{'='*80}")
    print("选择结果总结")
    print(f"{'='*80}")
    
    for market, result in results.items():
        print(f"\n{market.upper()}市场:")
        print(f"  总因子数量: {result['total_factors']}")
        print(f"  选择因子数量: {len(result['selected_factors'])}")
        print(f"  选择的因子: {result['selected_factors']}")

if __name__ == "__main__":
    main()


