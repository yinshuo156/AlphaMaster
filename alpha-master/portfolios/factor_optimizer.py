#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子优化器
根据质量分析报告对因子进行优化，包括去相关处理和异常值优化
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class FactorOptimizer:
    """因子优化器"""
    
    def __init__(self, alpha_pool_path: str = "alpha_pool"):
        self.alpha_pool_path = alpha_pool_path
        self.markets = ["a_share", "crypto", "us"]
        self.optimized_factors = {}
        
    def load_factor_data(self, market: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载因子数据"""
        factor_file = os.path.join(self.alpha_pool_path, f"{market}_alpha_factors_ultra_optimized.csv")
        stats_file = os.path.join(self.alpha_pool_path, f"{market}_alpha_factors_ultra_optimized_stats.csv")
        
        if not os.path.exists(factor_file):
            raise FileNotFoundError(f"因子文件不存在: {factor_file}")
        if not os.path.exists(stats_file):
            raise FileNotFoundError(f"统计文件不存在: {stats_file}")
        
        factor_df = pd.read_csv(factor_file)
        stats_df = pd.read_csv(stats_file)
        
        return factor_df, stats_df
    
    def convert_to_wide_format(self, factor_df: pd.DataFrame, market: str) -> pd.DataFrame:
        """将因子数据转换为宽格式"""
        # 确定资产列名
        asset_col = 'stock' if market == 'a_share' or market == 'us' else 'crypto'
        
        # 转换为宽格式
        pivot_df = factor_df.pivot_table(
            index=['date', asset_col],
            columns='factor_name',
            values='factor_value'
        ).reset_index()
        
        return pivot_df
    
    def detect_high_correlation_factors(self, factor_matrix: pd.DataFrame, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """检测高相关性因子对"""
        # 获取因子列
        factor_columns = [col for col in factor_matrix.columns if col not in ['date', 'stock', 'crypto']]
        
        # 计算相关性矩阵
        correlation_matrix = factor_matrix[factor_columns].corr()
        
        # 找到高相关性因子对
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        corr_val
                    ))
        
        return high_corr_pairs
    
    def apply_pca_decorrelation(self, factor_matrix: pd.DataFrame, n_components: Optional[int] = None, 
                              explained_variance_threshold: float = 0.95) -> Tuple[pd.DataFrame, PCA]:
        """应用PCA进行因子去相关"""
        # 获取因子列
        factor_columns = [col for col in factor_matrix.columns if col not in ['date', 'stock', 'crypto']]
        
        # 准备数据
        factor_data = factor_matrix[factor_columns].fillna(0)
        
        # 标准化
        scaler = StandardScaler()
        factor_data_scaled = scaler.fit_transform(factor_data)
        
        # 确定主成分数量
        if n_components is None:
            pca_temp = PCA()
            pca_temp.fit(factor_data_scaled)
            cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum_variance >= explained_variance_threshold) + 1
        
        # 应用PCA
        pca = PCA(n_components=n_components)
        factor_data_pca = pca.fit_transform(factor_data_scaled)
        
        # 保持原始因子名称，选择最重要的因子
        # 根据PCA的explained_variance_ratio_选择最重要的因子
        feature_importance = np.abs(pca.components_).sum(axis=0)
        top_factor_indices = np.argsort(feature_importance)[-n_components:]
        selected_factor_names = [factor_columns[i] for i in top_factor_indices]
        
        # 创建新的DataFrame，使用原始因子名称
        result_df = factor_matrix[['date', 'stock' if 'stock' in factor_matrix.columns else 'crypto']].copy()
        for i, factor_name in enumerate(selected_factor_names):
            result_df[factor_name] = factor_data_pca[:, i]
        
        return result_df, pca
    
    def apply_factor_rotation(self, factor_matrix: pd.DataFrame, method: str = 'varimax') -> pd.DataFrame:
        """应用因子旋转技术"""
        from scipy.linalg import svd
        
        # 获取因子列
        factor_columns = [col for col in factor_matrix.columns if col not in ['date', 'stock', 'crypto']]
        factor_data = factor_matrix[factor_columns].fillna(0)
        
        # 标准化
        scaler = StandardScaler()
        factor_data_scaled = scaler.fit_transform(factor_data)
        
        # 应用PCA
        pca = PCA()
        factor_data_pca = pca.fit_transform(factor_data_scaled)
        
        # 应用Varimax旋转
        if method == 'varimax':
            rotated_factors = self._varimax_rotation(factor_data_pca)
        else:
            rotated_factors = factor_data_pca
        
        # 创建新的DataFrame
        result_df = factor_matrix[['date', 'stock' if 'stock' in factor_matrix.columns else 'crypto']].copy()
        for i in range(rotated_factors.shape[1]):
            result_df[f"Rotated_Factor_{i+1}"] = rotated_factors[:, i]
        
        return result_df
    
    def _varimax_rotation(self, factors: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
        """Varimax旋转实现"""
        n_factors = factors.shape[1]
        rotation_matrix = np.eye(n_factors)
        
        for iteration in range(max_iter):
            old_rotation = rotation_matrix.copy()
            
            for i in range(n_factors):
                for j in range(i+1, n_factors):
                    # 计算旋转角度
                    x = factors[:, [i, j]]
                    u, s, v = svd(x.T @ x)
                    rotation_angle = np.arctan2(2 * np.sum(x[:, 0] * x[:, 1]), 
                                              np.sum(x[:, 0]**2 - x[:, 1]**2)) / 4
                    
                    # 应用旋转
                    cos_angle = np.cos(rotation_angle)
                    sin_angle = np.sin(rotation_angle)
                    
                    rotation_submatrix = np.array([[cos_angle, -sin_angle],
                                                  [sin_angle, cos_angle]])
                    
                    # 更新旋转矩阵
                    rotation_matrix[:, [i, j]] = rotation_matrix[:, [i, j]] @ rotation_submatrix
            
            # 检查收敛
            if np.max(np.abs(rotation_matrix - old_rotation)) < tol:
                break
        
        return factors @ rotation_matrix
    
    def optimize_outliers(self, factor_matrix: pd.DataFrame, market: str) -> pd.DataFrame:
        """优化异常值"""
        result_df = factor_matrix.copy()
        
        # 获取因子列
        factor_columns = [col for col in factor_matrix.columns if col not in ['date', 'stock', 'crypto']]
        
        for factor_name in factor_columns:
            factor_data = factor_matrix[factor_name].fillna(0)
            
            # 计算IQR
            q25 = factor_data.quantile(0.25)
            q75 = factor_data.quantile(0.75)
            iqr = q75 - q25
            
            # 定义异常值边界
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            # 对异常值进行Winsorization
            factor_data_optimized = factor_data.clip(lower=lower_bound, upper=upper_bound)
            
            # 应用更严格的Winsorization（0.5%）
            lower_percentile = factor_data_optimized.quantile(0.005)
            upper_percentile = factor_data_optimized.quantile(0.995)
            factor_data_optimized = factor_data_optimized.clip(lower=lower_percentile, upper=upper_percentile)
            
            result_df[factor_name] = factor_data_optimized
        
        return result_df
    
    def apply_factor_selection(self, factor_matrix: pd.DataFrame, method: str = 'variance', 
                             n_factors: Optional[int] = None) -> pd.DataFrame:
        """应用因子选择"""
        # 获取因子列
        factor_columns = [col for col in factor_matrix.columns if col not in ['date', 'stock', 'crypto']]
        
        if method == 'variance':
            # 基于方差的因子选择
            variances = factor_matrix[factor_columns].var()
            if n_factors is None:
                n_factors = len(factor_columns) // 2  # 选择一半因子
            
            selected_factors = variances.nlargest(n_factors).index.tolist()
        
        elif method == 'correlation':
            # 基于相关性的因子选择
            correlation_matrix = factor_matrix[factor_columns].corr()
            
            # 计算每个因子的平均相关性
            avg_correlations = correlation_matrix.abs().mean()
            
            # 选择相关性较低的因子
            if n_factors is None:
                n_factors = len(factor_columns) // 2
            
            selected_factors = avg_correlations.nsmallest(n_factors).index.tolist()
        
        elif method == 'clustering':
            # 基于聚类的因子选择
            factor_data = factor_matrix[factor_columns].fillna(0)
            
            # 标准化
            scaler = StandardScaler()
            factor_data_scaled = scaler.fit_transform(factor_data)
            
            # K-means聚类
            if n_factors is None:
                n_clusters = len(factor_columns) // 2
            else:
                n_clusters = n_factors
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(factor_data_scaled)
            
            # 从每个聚类中选择代表性因子
            selected_factors = []
            for cluster_id in range(n_clusters):
                cluster_factors = [factor_columns[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                if cluster_factors:
                    # 选择距离聚类中心最近的因子
                    cluster_center = kmeans.cluster_centers_[cluster_id]
                    distances = [np.linalg.norm(factor_data_scaled[i] - cluster_center) 
                               for i, factor in enumerate(factor_columns) if factor in cluster_factors]
                    closest_factor_idx = np.argmin(distances)
                    selected_factors.append(cluster_factors[closest_factor_idx])
        
        else:
            selected_factors = factor_columns
        
        # 创建结果DataFrame
        result_df = factor_matrix[['date', 'stock' if 'stock' in factor_matrix.columns else 'crypto']].copy()
        for factor in selected_factors:
            result_df[factor] = factor_matrix[factor]
        
        return result_df
    
    def optimize_market_factors(self, market: str, optimization_config: Dict) -> Dict:
        """优化单个市场的因子"""
        print(f"\n{'='*60}")
        print(f"优化 {market.upper()} 市场因子")
        print(f"{'='*60}")
        
        # 加载数据
        factor_df, stats_df = self.load_factor_data(market)
        print(f"原始因子数量: {len(stats_df)}")
        
        # 转换为宽格式
        factor_matrix = self.convert_to_wide_format(factor_df, market)
        print(f"数据形状: {factor_matrix.shape}")
        
        # 检测高相关性因子
        high_corr_pairs = self.detect_high_correlation_factors(factor_matrix, threshold=0.8)
        print(f"高相关性因子对数量: {len(high_corr_pairs)}")
        
        # 应用优化步骤
        optimized_matrix = factor_matrix.copy()
        
        # 1. 异常值优化
        if optimization_config.get('optimize_outliers', True):
            print("应用异常值优化...")
            optimized_matrix = self.optimize_outliers(optimized_matrix, market)
        
        # 2. 因子选择
        if optimization_config.get('apply_selection', True):
            selection_method = optimization_config.get('selection_method', 'variance')
            n_factors = optimization_config.get('n_factors', None)
            print(f"应用因子选择 (方法: {selection_method})...")
            optimized_matrix = self.apply_factor_selection(optimized_matrix, method=selection_method, n_factors=n_factors)
            print(f"选择后因子数量: {len([col for col in optimized_matrix.columns if col not in ['date', 'stock', 'crypto']])}")
        
        # 3. 去相关处理
        if optimization_config.get('apply_decorrelation', True):
            decorrelation_method = optimization_config.get('decorrelation_method', 'pca')
            if decorrelation_method == 'pca':
                print("应用PCA去相关...")
                optimized_matrix, pca_model = self.apply_pca_decorrelation(
                    optimized_matrix, 
                    n_components=optimization_config.get('pca_components', None),
                    explained_variance_threshold=optimization_config.get('explained_variance_threshold', 0.95)
                )
            elif decorrelation_method == 'rotation':
                print("应用因子旋转...")
                optimized_matrix = self.apply_factor_rotation(optimized_matrix)
        
        # 计算优化后的相关性
        final_high_corr_pairs = self.detect_high_correlation_factors(optimized_matrix, threshold=0.8)
        print(f"优化后高相关性因子对数量: {len(final_high_corr_pairs)}")
        
        # 转换回长格式
        asset_col = 'stock' if market == 'a_share' or market == 'us' else 'crypto'
        factor_columns = [col for col in optimized_matrix.columns if col not in ['date', asset_col]]
        
        optimized_long_format = []
        for _, row in optimized_matrix.iterrows():
            for factor_name in factor_columns:
                optimized_long_format.append({
                    'date': row['date'],
                    asset_col: row[asset_col],
                    'factor_name': factor_name,
                    'factor_value': row[factor_name]
                })
        
        optimized_df = pd.DataFrame(optimized_long_format)
        
        return {
            'market': market,
            'original_factors': len(stats_df),
            'optimized_factors': len(factor_columns),
            'original_high_corr_pairs': len(high_corr_pairs),
            'optimized_high_corr_pairs': len(final_high_corr_pairs),
            'optimized_data': optimized_df,
            'optimized_matrix': optimized_matrix
        }
    
    def optimize_all_markets(self, optimization_config: Dict) -> Dict:
        """优化所有市场的因子"""
        print("=" * 80)
        print("开始因子优化")
        print("=" * 80)
        
        all_results = {}
        
        for market in self.markets:
            try:
                result = self.optimize_market_factors(market, optimization_config)
                all_results[market] = result
                self.optimized_factors[market] = result['optimized_data']
            except Exception as e:
                print(f"❌ 优化{market}市场时出错: {e}")
                continue
        
        return all_results
    
    def save_optimized_factors(self, output_path: str = "portfolios/optimized_factors"):
        """保存优化后的因子"""
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        for market, factor_data in self.optimized_factors.items():
            # 保存因子数据
            factor_file = os.path.join(output_path, f"{market}_optimized_factors.csv")
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
            stats_file = os.path.join(output_path, f"{market}_optimized_factors_stats.csv")
            stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
            
            print(f"保存{market}市场优化因子: {factor_file}")
            print(f"保存{market}市场统计信息: {stats_file}")

def main():
    """主函数"""
    # 优化配置
    optimization_config = {
        'optimize_outliers': True,           # 优化异常值
        'apply_selection': True,             # 应用因子选择
        'selection_method': 'variance',      # 选择方法: variance, correlation, clustering
        'n_factors': 15,                     # 选择因子数量
        'apply_decorrelation': True,         # 应用去相关
        'decorrelation_method': 'pca',       # 去相关方法: pca, rotation
        'pca_components': 10,                # PCA主成分数量
        'explained_variance_threshold': 0.95 # 解释方差阈值
    }
    
    # 创建优化器
    optimizer = FactorOptimizer()
    
    # 执行优化
    results = optimizer.optimize_all_markets(optimization_config)
    
    # 保存结果
    optimizer.save_optimized_factors()
    
    # 显示优化结果
    print(f"\n{'='*80}")
    print("优化结果总结")
    print(f"{'='*80}")
    
    for market, result in results.items():
        print(f"\n{market.upper()}市场:")
        print(f"  原始因子数量: {result['original_factors']}")
        print(f"  优化后因子数量: {result['optimized_factors']}")
        print(f"  原始高相关性对: {result['original_high_corr_pairs']}")
        print(f"  优化后高相关性对: {result['optimized_high_corr_pairs']}")
        print(f"  相关性改善: {result['original_high_corr_pairs'] - result['optimized_high_corr_pairs']}")

if __name__ == "__main__":
    main()

