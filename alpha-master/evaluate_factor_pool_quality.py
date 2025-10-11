#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子池质量评估脚本
全面评估alpha_pool中所有因子的质量和数据准确性
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FactorPoolQualityEvaluator:
    """因子池质量评估器"""
    
    def __init__(self, alpha_pool_path: str = "alpha_pool"):
        self.alpha_pool_path = alpha_pool_path
        self.markets = ["a_share", "crypto", "us"]
        self.evaluation_results = {}
        
    def load_factor_data(self, market: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载因子数据和统计信息"""
        factor_file = os.path.join(self.alpha_pool_path, f"{market}_alpha_factors_ultra_optimized.csv")
        stats_file = os.path.join(self.alpha_pool_path, f"{market}_alpha_factors_ultra_optimized_stats.csv")
        
        if not os.path.exists(factor_file):
            raise FileNotFoundError(f"因子文件不存在: {factor_file}")
        if not os.path.exists(stats_file):
            raise FileNotFoundError(f"统计文件不存在: {stats_file}")
        
        factor_df = pd.read_csv(factor_file)
        stats_df = pd.read_csv(stats_file)
        
        return factor_df, stats_df
    
    def evaluate_data_completeness(self, factor_df: pd.DataFrame, market: str) -> Dict:
        """评估数据完整性"""
        print(f"\n{'='*50}")
        print(f"{market.upper()}市场数据完整性评估")
        print(f"{'='*50}")
        
        # 基本信息
        total_records = len(factor_df)
        unique_dates = factor_df['date'].nunique()
        unique_assets = factor_df.iloc[:, 1].nunique()  # 第二列是资产列
        unique_factors = factor_df['factor_name'].nunique()
        
        print(f"总记录数: {total_records:,}")
        print(f"唯一日期数: {unique_dates}")
        print(f"唯一资产数: {unique_assets}")
        print(f"唯一因子数: {unique_factors}")
        
        # 检查缺失值
        missing_values = factor_df['factor_value'].isna().sum()
        missing_rate = missing_values / total_records * 100
        
        print(f"缺失值数量: {missing_values:,}")
        print(f"缺失率: {missing_rate:.2f}%")
        
        # 检查无穷大值
        inf_values = np.isinf(factor_df['factor_value']).sum()
        inf_rate = inf_values / total_records * 100
        
        print(f"无穷大值数量: {inf_values:,}")
        print(f"无穷大值率: {inf_rate:.2f}%")
        
        # 数据时间范围
        factor_df['date'] = pd.to_datetime(factor_df['date'])
        date_range = factor_df['date'].max() - factor_df['date'].min()
        print(f"数据时间跨度: {date_range.days}天")
        print(f"最早日期: {factor_df['date'].min()}")
        print(f"最晚日期: {factor_df['date'].max()}")
        
        completeness_score = 100 - missing_rate - inf_rate
        completeness_score = max(0, completeness_score)
        
        print(f"数据完整性评分: {completeness_score:.1f}/100")
        
        return {
            'total_records': total_records,
            'unique_dates': unique_dates,
            'unique_assets': unique_assets,
            'unique_factors': unique_factors,
            'missing_values': missing_values,
            'missing_rate': missing_rate,
            'inf_values': inf_values,
            'inf_rate': inf_rate,
            'date_range_days': date_range.days,
            'completeness_score': completeness_score
        }
    
    def evaluate_factor_statistics(self, stats_df: pd.DataFrame, market: str) -> Dict:
        """评估因子统计信息"""
        print(f"\n{'='*50}")
        print(f"{market.upper()}市场因子统计评估")
        print(f"{'='*50}")
        
        # 基本统计
        print(f"因子数量: {len(stats_df)}")
        
        # 检查每个因子的统计信息
        quality_issues = []
        factor_quality_scores = []
        
        for _, row in stats_df.iterrows():
            factor_name = row['factor_name']
            count = row['count']
            mean_val = row['mean']
            std_val = row['std']
            min_val = row['min']
            max_val = row['max']
            
            # 计算质量评分
            quality_score = 100
            issues = []
            
            # 检查数据量
            if count < 1000:
                quality_score -= 20
                issues.append("数据量不足")
            
            # 检查标准差
            if std_val > 5:
                quality_score -= 15
                issues.append("标准差过大")
            elif std_val < 0.01:
                quality_score -= 10
                issues.append("标准差过小")
            
            # 检查极值
            if abs(max_val) > 10 or abs(min_val) > 10:
                quality_score -= 20
                issues.append("存在极值")
            
            # 检查均值
            if abs(mean_val) > 5:
                quality_score -= 10
                issues.append("均值偏离过大")
            
            # 检查数值范围合理性
            range_val = max_val - min_val
            if range_val > 20:
                quality_score -= 15
                issues.append("数值范围过大")
            elif range_val < 0.1:
                quality_score -= 10
                issues.append("数值范围过小")
            
            quality_score = max(0, quality_score)
            factor_quality_scores.append(quality_score)
            
            if issues:
                quality_issues.append({
                    'factor': factor_name,
                    'score': quality_score,
                    'issues': issues,
                    'stats': {
                        'count': count,
                        'mean': mean_val,
                        'std': std_val,
                        'min': min_val,
                        'max': max_val
                    }
                })
        
        # 计算整体质量评分
        avg_quality_score = np.mean(factor_quality_scores)
        min_quality_score = np.min(factor_quality_scores)
        
        print(f"平均质量评分: {avg_quality_score:.1f}/100")
        print(f"最低质量评分: {min_quality_score:.1f}/100")
        print(f"质量评分标准差: {np.std(factor_quality_scores):.1f}")
        
        # 显示有问题的因子
        if quality_issues:
            print(f"\n发现 {len(quality_issues)} 个有质量问题的因子:")
            for issue in quality_issues:
                print(f"  {issue['factor']}: {issue['score']:.1f}/100 - {', '.join(issue['issues'])}")
        else:
            print("\n✅ 所有因子质量良好")
        
        return {
            'factor_count': len(stats_df),
            'avg_quality_score': avg_quality_score,
            'min_quality_score': min_quality_score,
            'quality_std': np.std(factor_quality_scores),
            'problematic_factors': len(quality_issues),
            'quality_issues': quality_issues
        }
    
    def evaluate_factor_distribution(self, factor_df: pd.DataFrame, market: str) -> Dict:
        """评估因子分布特征"""
        print(f"\n{'='*50}")
        print(f"{market.upper()}市场因子分布评估")
        print(f"{'='*50}")
        
        # 按因子分组分析
        factor_groups = factor_df.groupby('factor_name')['factor_value']
        
        distribution_analysis = {}
        
        for factor_name, values in factor_groups:
            # 移除缺失值和无穷大值
            clean_values = values.dropna()
            clean_values = clean_values[np.isfinite(clean_values)]
            
            if len(clean_values) == 0:
                continue
            
            # 计算分布特征
            skewness = clean_values.skew()
            kurtosis = clean_values.kurtosis()
            
            # 检查正态性（简化的Shapiro-Wilk测试，使用样本）
            sample_size = min(5000, len(clean_values))
            sample_values = clean_values.sample(n=sample_size, random_state=42)
            
            # 计算分位数
            q25 = clean_values.quantile(0.25)
            q50 = clean_values.quantile(0.50)
            q75 = clean_values.quantile(0.75)
            iqr = q75 - q25
            
            # 异常值检测（IQR方法）
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outliers = clean_values[(clean_values < lower_bound) | (clean_values > upper_bound)]
            outlier_rate = len(outliers) / len(clean_values) * 100
            
            distribution_analysis[factor_name] = {
                'count': len(clean_values),
                'skewness': skewness,
                'kurtosis': kurtosis,
                'q25': q25,
                'q50': q50,
                'q75': q75,
                'iqr': iqr,
                'outlier_count': len(outliers),
                'outlier_rate': outlier_rate
            }
        
        # 计算整体分布特征
        all_skewness = [info['skewness'] for info in distribution_analysis.values()]
        all_kurtosis = [info['kurtosis'] for info in distribution_analysis.values()]
        all_outlier_rates = [info['outlier_rate'] for info in distribution_analysis.values()]
        
        print(f"偏度统计:")
        print(f"  平均偏度: {np.mean(all_skewness):.3f}")
        print(f"  偏度标准差: {np.std(all_skewness):.3f}")
        print(f"  偏度范围: [{np.min(all_skewness):.3f}, {np.max(all_skewness):.3f}]")
        
        print(f"峰度统计:")
        print(f"  平均峰度: {np.mean(all_kurtosis):.3f}")
        print(f"  峰度标准差: {np.std(all_kurtosis):.3f}")
        print(f"  峰度范围: [{np.min(all_kurtosis):.3f}, {np.max(all_kurtosis):.3f}]")
        
        print(f"异常值统计:")
        print(f"  平均异常值率: {np.mean(all_outlier_rates):.2f}%")
        print(f"  异常值率标准差: {np.std(all_outlier_rates):.2f}%")
        print(f"  异常值率范围: [{np.min(all_outlier_rates):.2f}%, {np.max(all_outlier_rates):.2f}%]")
        
        # 识别分布异常的因子
        distribution_issues = []
        for factor_name, info in distribution_analysis.items():
            issues = []
            
            if abs(info['skewness']) > 2:
                issues.append("严重偏斜")
            if abs(info['kurtosis']) > 5:
                issues.append("峰度过高")
            if info['outlier_rate'] > 10:
                issues.append("异常值过多")
            
            if issues:
                distribution_issues.append({
                    'factor': factor_name,
                    'issues': issues,
                    'skewness': info['skewness'],
                    'kurtosis': info['kurtosis'],
                    'outlier_rate': info['outlier_rate']
                })
        
        if distribution_issues:
            print(f"\n发现 {len(distribution_issues)} 个分布异常的因子:")
            for issue in distribution_issues:
                print(f"  {issue['factor']}: {', '.join(issue['issues'])}")
        else:
            print("\n✅ 所有因子分布正常")
        
        return {
            'avg_skewness': np.mean(all_skewness),
            'avg_kurtosis': np.mean(all_kurtosis),
            'avg_outlier_rate': np.mean(all_outlier_rates),
            'distribution_issues': len(distribution_issues),
            'distribution_analysis': distribution_analysis
        }
    
    def evaluate_factor_correlation(self, factor_df: pd.DataFrame, market: str) -> Dict:
        """评估因子相关性"""
        print(f"\n{'='*50}")
        print(f"{market.upper()}市场因子相关性评估")
        print(f"{'='*50}")
        
        # 将数据转换为宽格式
        pivot_df = factor_df.pivot_table(
            index=['date', factor_df.columns[1]],  # date和资产列
            columns='factor_name',
            values='factor_value'
        ).reset_index()
        
        # 计算因子间相关性
        factor_columns = [col for col in pivot_df.columns if col not in ['date', pivot_df.columns[1]]]
        correlation_matrix = pivot_df[factor_columns].corr()
        
        # 分析相关性
        # 获取上三角矩阵（排除对角线）
        upper_tri = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        # 计算高相关性对
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:  # 高相关性阈值
                    high_corr_pairs.append({
                        'factor1': correlation_matrix.columns[i],
                        'factor2': correlation_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        # 计算平均相关性
        all_correlations = upper_tri.values.flatten()
        all_correlations = all_correlations[~np.isnan(all_correlations)]
        
        avg_correlation = np.mean(np.abs(all_correlations))
        max_correlation = np.max(np.abs(all_correlations))
        
        print(f"因子数量: {len(factor_columns)}")
        print(f"平均绝对相关性: {avg_correlation:.3f}")
        print(f"最大绝对相关性: {max_correlation:.3f}")
        print(f"高相关性因子对数量: {len(high_corr_pairs)}")
        
        if high_corr_pairs:
            print(f"\n高相关性因子对 (|r| > 0.8):")
            for pair in high_corr_pairs[:10]:  # 只显示前10个
                print(f"  {pair['factor1']} <-> {pair['factor2']}: {pair['correlation']:.3f}")
            if len(high_corr_pairs) > 10:
                print(f"  ... 还有 {len(high_corr_pairs) - 10} 个高相关性对")
        else:
            print("\n✅ 无高相关性因子对")
        
        return {
            'factor_count': len(factor_columns),
            'avg_correlation': avg_correlation,
            'max_correlation': max_correlation,
            'high_corr_pairs': len(high_corr_pairs),
            'correlation_matrix': correlation_matrix
        }
    
    def evaluate_market_consistency(self) -> Dict:
        """评估跨市场一致性"""
        print(f"\n{'='*50}")
        print("跨市场一致性评估")
        print(f"{'='*50}")
        
        market_stats = {}
        
        # 收集各市场统计信息
        for market in self.markets:
            try:
                _, stats_df = self.load_factor_data(market)
                market_stats[market] = {
                    'factor_count': len(stats_df),
                    'factor_names': set(stats_df['factor_name']),
                    'avg_std': stats_df['std'].mean(),
                    'avg_range': (stats_df['max'] - stats_df['min']).mean()
                }
            except FileNotFoundError:
                print(f"⚠️ {market}市场数据文件不存在")
                continue
        
        if len(market_stats) < 2:
            print("⚠️ 市场数量不足，无法进行跨市场一致性评估")
            return {}
        
        # 检查因子数量一致性
        factor_counts = [stats['factor_count'] for stats in market_stats.values()]
        factor_count_consistent = len(set(factor_counts)) == 1
        
        print(f"各市场因子数量: {dict(zip(market_stats.keys(), factor_counts))}")
        print(f"因子数量一致性: {'✅' if factor_count_consistent else '❌'}")
        
        # 检查因子名称一致性
        all_factor_names = [stats['factor_names'] for stats in market_stats.values()]
        common_factors = set.intersection(*all_factor_names)
        unique_factors = set.union(*all_factor_names) - common_factors
        
        print(f"共同因子数量: {len(common_factors)}")
        print(f"独有因子数量: {len(unique_factors)}")
        
        if unique_factors:
            print(f"独有因子:")
            for market, stats in market_stats.items():
                market_unique = stats['factor_names'] - common_factors
                if market_unique:
                    print(f"  {market}: {market_unique}")
        
        # 检查统计特征一致性
        print(f"\n统计特征一致性:")
        for market, stats in market_stats.items():
            print(f"  {market}: 平均标准差={stats['avg_std']:.3f}, 平均范围={stats['avg_range']:.3f}")
        
        return {
            'market_count': len(market_stats),
            'factor_count_consistent': factor_count_consistent,
            'common_factors': len(common_factors),
            'unique_factors': len(unique_factors),
            'market_stats': market_stats
        }
    
    def generate_quality_report(self) -> Dict:
        """生成质量评估报告"""
        print(f"\n{'='*80}")
        print("因子池质量评估报告")
        print(f"{'='*80}")
        
        overall_results = {}
        
        for market in self.markets:
            try:
                print(f"\n{'='*60}")
                print(f"评估 {market.upper()} 市场")
                print(f"{'='*60}")
                
                # 加载数据
                factor_df, stats_df = self.load_factor_data(market)
                
                # 执行各项评估
                completeness = self.evaluate_data_completeness(factor_df, market)
                statistics = self.evaluate_factor_statistics(stats_df, market)
                distribution = self.evaluate_factor_distribution(factor_df, market)
                correlation = self.evaluate_factor_correlation(factor_df, market)
                
                overall_results[market] = {
                    'completeness': completeness,
                    'statistics': statistics,
                    'distribution': distribution,
                    'correlation': correlation
                }
                
            except FileNotFoundError as e:
                print(f"❌ {market}市场数据文件不存在: {e}")
                continue
            except Exception as e:
                print(f"❌ 评估{market}市场时出错: {e}")
                continue
        
        # 跨市场一致性评估
        consistency = self.evaluate_market_consistency()
        
        # 计算总体质量评分
        print(f"\n{'='*80}")
        print("总体质量评分")
        print(f"{'='*80}")
        
        total_score = 0
        market_count = 0
        
        for market, results in overall_results.items():
            # 计算市场综合评分
            completeness_score = results['completeness']['completeness_score']
            statistics_score = results['statistics']['avg_quality_score']
            
            # 分布评分（基于异常值率）
            outlier_rate = results['distribution']['avg_outlier_rate']
            distribution_score = max(0, 100 - outlier_rate * 2)  # 异常值率每1%扣2分
            
            # 相关性评分（基于高相关性对数量）
            high_corr_pairs = results['correlation']['high_corr_pairs']
            correlation_score = max(0, 100 - high_corr_pairs * 2)  # 每对高相关性扣2分
            
            # 市场综合评分
            market_score = (completeness_score + statistics_score + distribution_score + correlation_score) / 4
            
            print(f"{market.upper()}市场综合评分: {market_score:.1f}/100")
            print(f"  数据完整性: {completeness_score:.1f}/100")
            print(f"  统计质量: {statistics_score:.1f}/100")
            print(f"  分布质量: {distribution_score:.1f}/100")
            print(f"  相关性质量: {correlation_score:.1f}/100")
            
            total_score += market_score
            market_count += 1
        
        if market_count > 0:
            overall_score = total_score / market_count
            print(f"\n总体质量评分: {overall_score:.1f}/100")
            
            # 质量等级评定
            if overall_score >= 90:
                quality_grade = "优秀"
            elif overall_score >= 80:
                quality_grade = "良好"
            elif overall_score >= 70:
                quality_grade = "一般"
            elif overall_score >= 60:
                quality_grade = "较差"
            else:
                quality_grade = "差"
            
            print(f"质量等级: {quality_grade}")
        
        return {
            'overall_score': overall_score if market_count > 0 else 0,
            'quality_grade': quality_grade if market_count > 0 else "无法评估",
            'market_results': overall_results,
            'consistency': consistency
        }
    
    def save_evaluation_report(self, results: Dict, output_file: str = "factor_pool_quality_evaluation_report.md"):
        """保存评估报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 因子池质量评估报告\n\n")
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## 总体评估结果\n\n")
            f.write(f"- **总体质量评分**: {results['overall_score']:.1f}/100\n")
            f.write(f"- **质量等级**: {results['quality_grade']}\n\n")
            
            f.write("## 各市场详细评估\n\n")
            
            for market, market_results in results['market_results'].items():
                f.write(f"### {market.upper()}市场\n\n")
                
                # 数据完整性
                completeness = market_results['completeness']
                f.write(f"#### 数据完整性\n")
                f.write(f"- 总记录数: {completeness['total_records']:,}\n")
                f.write(f"- 唯一日期数: {completeness['unique_dates']}\n")
                f.write(f"- 唯一资产数: {completeness['unique_assets']}\n")
                f.write(f"- 唯一因子数: {completeness['unique_factors']}\n")
                f.write(f"- 缺失率: {completeness['missing_rate']:.2f}%\n")
                f.write(f"- 无穷大值率: {completeness['inf_rate']:.2f}%\n")
                f.write(f"- 完整性评分: {completeness['completeness_score']:.1f}/100\n\n")
                
                # 统计质量
                statistics = market_results['statistics']
                f.write(f"#### 统计质量\n")
                f.write(f"- 因子数量: {statistics['factor_count']}\n")
                f.write(f"- 平均质量评分: {statistics['avg_quality_score']:.1f}/100\n")
                f.write(f"- 最低质量评分: {statistics['min_quality_score']:.1f}/100\n")
                f.write(f"- 有问题的因子数量: {statistics['problematic_factors']}\n\n")
                
                # 分布质量
                distribution = market_results['distribution']
                f.write(f"#### 分布质量\n")
                f.write(f"- 平均偏度: {distribution['avg_skewness']:.3f}\n")
                f.write(f"- 平均峰度: {distribution['avg_kurtosis']:.3f}\n")
                f.write(f"- 平均异常值率: {distribution['avg_outlier_rate']:.2f}%\n")
                f.write(f"- 分布异常因子数量: {distribution['distribution_issues']}\n\n")
                
                # 相关性质量
                correlation = market_results['correlation']
                f.write(f"#### 相关性质量\n")
                f.write(f"- 因子数量: {correlation['factor_count']}\n")
                f.write(f"- 平均绝对相关性: {correlation['avg_correlation']:.3f}\n")
                f.write(f"- 最大绝对相关性: {correlation['max_correlation']:.3f}\n")
                f.write(f"- 高相关性因子对数量: {correlation['high_corr_pairs']}\n\n")
            
            f.write("## 跨市场一致性\n\n")
            consistency = results['consistency']
            if consistency:
                f.write(f"- 市场数量: {consistency['market_count']}\n")
                f.write(f"- 因子数量一致性: {'是' if consistency['factor_count_consistent'] else '否'}\n")
                f.write(f"- 共同因子数量: {consistency['common_factors']}\n")
                f.write(f"- 独有因子数量: {consistency['unique_factors']}\n\n")
            
            f.write("## 建议\n\n")
            if results['overall_score'] >= 90:
                f.write("✅ 因子池质量优秀，可以直接用于量化投资研究。\n")
            elif results['overall_score'] >= 80:
                f.write("✅ 因子池质量良好，建议进行少量优化后使用。\n")
            elif results['overall_score'] >= 70:
                f.write("⚠️ 因子池质量一般，建议进行优化改进。\n")
            else:
                f.write("❌ 因子池质量较差，需要重新生成或大幅优化。\n")
        
        print(f"\n评估报告已保存到: {output_file}")

def main():
    """主函数"""
    print("=" * 80)
    print("因子池质量评估")
    print("=" * 80)
    
    # 创建评估器
    evaluator = FactorPoolQualityEvaluator()
    
    # 执行评估
    results = evaluator.generate_quality_report()
    
    # 保存报告
    evaluator.save_evaluation_report(results)
    
    print(f"\n评估完成！")

if __name__ == "__main__":
    main()


