# -*- coding: utf-8 -*-
"""
因子评估器
评估因子的质量，计算IC、收益率、Sharpe比率等指标
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dual_chain.factor_evaluator')

class FactorEvaluator:
    """
    因子评估器
    评估因子的质量，计算各种性能指标
    """
    
    def __init__(self, returns_data: pd.DataFrame):
        """
        初始化因子评估器
        
        Args:
            returns_data: 收益率数据
        """
        self.returns_data = returns_data
        logger.info(f"✅ 因子评估器初始化成功")
    
    def calculate_ic(self, factor_data: pd.DataFrame, forward_returns: int = 1) -> float:
        """
        计算因子的信息系数(IC)
        
        Args:
            factor_data: 因子数据
            forward_returns: 前瞻收益率天数
            
        Returns:
            平均IC值
        """
        # 确保索引一致
        factor_data = factor_data.dropna()
        
        # 获取收益率数据
        returns = self.returns_data.shift(-forward_returns).dropna()
        
        # 对齐数据
        common_index = factor_data.index.intersection(returns.index)
        factor_data = factor_data.loc[common_index]
        returns = returns.loc[common_index]
        
        # 计算每日IC
        daily_ics = []
        for date in common_index:
            if date not in factor_data.index or date not in returns.index:
                continue
                
            # 获取当日的因子值和收益率
            factor_values = factor_data.loc[date].dropna()
            ret_values = returns.loc[date].dropna()
            
            # 对齐股票
            common_stocks = factor_values.index.intersection(ret_values.index)
            if len(common_stocks) < 5:  # 确保有足够的股票进行相关性计算
                continue
                
            factor_values = factor_values.loc[common_stocks]
            ret_values = ret_values.loc[common_stocks]
            
            # 计算Spearman相关系数
            if factor_values.std() > 0 and ret_values.std() > 0:
                ic = factor_values.rank().corr(ret_values.rank(), method='spearman')
                daily_ics.append(ic)
        
        if not daily_ics:
            logger.warning("⚠️  无法计算IC，数据不足")
            return 0.0
        
        avg_ic = np.mean(daily_ics)
        logger.info(f"📊 IC计算完成，平均IC: {avg_ic:.4f}")
        return avg_ic
    
    def calculate_ic_ir(self, factor_data: pd.DataFrame, forward_returns: int = 1) -> float:
        """
        计算因子的IC信息比率
        
        Args:
            factor_data: 因子数据
            forward_returns: 前瞻收益率天数
            
        Returns:
            IC-IR值
        """
        # 确保索引一致
        factor_data = factor_data.dropna()
        
        # 获取收益率数据
        returns = self.returns_data.shift(-forward_returns).dropna()
        
        # 对齐数据
        common_index = factor_data.index.intersection(returns.index)
        factor_data = factor_data.loc[common_index]
        returns = returns.loc[common_index]
        
        # 计算每日IC
        daily_ics = []
        for date in common_index:
            if date not in factor_data.index or date not in returns.index:
                continue
                
            # 获取当日的因子值和收益率
            factor_values = factor_data.loc[date].dropna()
            ret_values = returns.loc[date].dropna()
            
            # 对齐股票
            common_stocks = factor_values.index.intersection(ret_values.index)
            if len(common_stocks) < 5:
                continue
                
            factor_values = factor_values.loc[common_stocks]
            ret_values = ret_values.loc[common_stocks]
            
            # 计算Spearman相关系数
            if factor_values.std() > 0 and ret_values.std() > 0:
                ic = factor_values.rank().corr(ret_values.rank(), method='spearman')
                daily_ics.append(ic)
        
        if not daily_ics:
            logger.warning("⚠️  无法计算IC-IR，数据不足")
            return 0.0
        
        ic_ir = np.mean(daily_ics) / (np.std(daily_ics) + 1e-8)
        logger.info(f"📊 IC-IR计算完成: {ic_ir:.4f}")
        return ic_ir
    
    def calculate_factor_returns(self, factor_data: pd.DataFrame, 
                               n_groups: int = 5, 
                               group_num: int = 1) -> pd.Series:
        """
        计算因子分组收益
        
        Args:
            factor_data: 因子数据
            n_groups: 分组数量
            group_num: 组号（1为最高分组，n_groups为最低分组）
            
        Returns:
            分组收益率序列
        """
        # 确保索引一致
        factor_data = factor_data.dropna()
        returns = self.returns_data.dropna()
        
        # 对齐数据
        common_index = factor_data.index.intersection(returns.index)
        factor_data = factor_data.loc[common_index]
        returns = returns.loc[common_index]
        
        # 计算分组收益
        group_returns = []
        for date in common_index:
            if date not in factor_data.index or date not in returns.index:
                continue
                
            # 获取当日的因子值和收益率
            factor_values = factor_data.loc[date].dropna()
            ret_values = returns.loc[date].dropna()
            
            # 对齐股票
            common_stocks = factor_values.index.intersection(ret_values.index)
            if len(common_stocks) < n_groups * 2:
                continue
                
            factor_values = factor_values.loc[common_stocks]
            ret_values = ret_values.loc[common_stocks]
            
            # 分组
            quantiles = factor_values.quantile(np.linspace(0, 1, n_groups + 1))
            
            # 处理极端情况
            if len(quantiles.unique()) < n_groups:
                continue
                
            # 根据组号选择股票
            if group_num == 1:
                selected_stocks = factor_values[factor_values >= quantiles.iloc[-2]].index
            elif group_num == n_groups:
                selected_stocks = factor_values[factor_values <= quantiles.iloc[1]].index
            else:
                selected_stocks = factor_values[
                    (factor_values > quantiles.iloc[group_num - 1]) & 
                    (factor_values <= quantiles.iloc[group_num])
                ].index
            
            # 计算平均收益率
            if selected_stocks.empty:
                group_returns.append(0)
            else:
                avg_return = ret_values.loc[selected_stocks].mean()
                group_returns.append(avg_return)
        
        if not group_returns:
            logger.warning("⚠️  无法计算分组收益，数据不足")
            return pd.Series()
        
        returns_series = pd.Series(group_returns, index=common_index[:len(group_returns)])
        logger.info(f"📊 分组收益计算完成，组号: {group_num}, 平均日收益: {np.mean(group_returns):.6f}")
        return returns_series
    
    def calculate_sharpe_ratio(self, returns_series: pd.Series, annualization: int = 252) -> float:
        """
        计算夏普比率
        
        Args:
            returns_series: 收益率序列
            annualization: 年化因子
            
        Returns:
            夏普比率
        """
        if returns_series.empty:
            logger.warning("⚠️  无法计算夏普比率，收益率序列为空")
            return 0.0
        
        daily_return = returns_series.mean()
        daily_vol = returns_series.std()
        
        if daily_vol == 0:
            return 0.0
        
        sharpe = (daily_return * annualization) / (daily_vol * np.sqrt(annualization))
        logger.info(f"📊 夏普比率计算完成: {sharpe:.4f}")
        return sharpe
    
    def evaluate_factor(self, factor_data: pd.DataFrame, factor_name: str) -> Dict[str, float]:
        """
        完整评估因子质量
        
        Args:
            factor_data: 因子数据
            factor_name: 因子名称
            
        Returns:
            评估指标字典
        """
        logger.info(f"🔍 开始评估因子: {factor_name}")
        
        # 计算IC
        ic = self.calculate_ic(factor_data)
        ic_ir = self.calculate_ic_ir(factor_data)
        
        # 计算分组收益
        top_group_returns = self.calculate_factor_returns(factor_data, group_num=1)
        bottom_group_returns = self.calculate_factor_returns(factor_data, group_num=5)
        
        # 计算多空收益
        if not top_group_returns.empty and not bottom_group_returns.empty:
            # 对齐索引
            common_dates = top_group_returns.index.intersection(bottom_group_returns.index)
            long_short_returns = top_group_returns.loc[common_dates] - bottom_group_returns.loc[common_dates]
            total_return = long_short_returns.sum()
            annual_return = long_short_returns.mean() * 252
            annual_volatility = long_short_returns.std() * np.sqrt(252)
            sharpe = self.calculate_sharpe_ratio(long_short_returns)
        else:
            total_return = 0
            annual_return = 0
            annual_volatility = 0
            sharpe = 0
        
        # 构建评估结果
        metrics = {
            "ic": ic,
            "ic_ir": ic_ir,
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe": sharpe
        }
        
        logger.info(f"✅ 因子评估完成: {factor_name}")
        logger.info(f"📊 评估结果: IC={ic:.4f}, Sharpe={sharpe:.4f}, 年化收益={annual_return:.4f}")
        
        return metrics
    
    def determine_factor_quality(self, metrics: Dict[str, float]) -> Tuple[bool, str]:
        """
        判断因子质量，决定是否进入有效池
        
        Args:
            metrics: 评估指标
            
        Returns:
            (是否有效, 原因)
        """
        ic_threshold = 0.01  # IC阈值
        sharpe_threshold = 0.3  # 夏普比率阈值
        
        # 检查IC
        if metrics["ic"] < ic_threshold:
            return False, f"IC值过低，不达标 ({metrics['ic']:.4f} < {ic_threshold})"
        
        # 检查夏普比率
        if metrics["sharpe"] < sharpe_threshold:
            return False, f"夏普比率过低，不达标 ({metrics['sharpe']:.4f} < {sharpe_threshold})"
        
        # 检查年化收益
        if metrics["annual_return"] < 0:
            return False, f"年化收益为负，不达标 ({metrics['annual_return']:.4f} < 0)"
        
        # 所有指标通过
        return True, "因子质量达标"
    
    def get_factor_quality_score(self, metrics: Dict[str, float]) -> float:
        """
        计算因子质量综合得分
        
        Args:
            metrics: 评估指标
            
        Returns:
            质量得分 (0-100)
        """
        # IC得分 (0-40分)
        ic_score = min(40, max(0, metrics["ic"] * 2000))  # IC=0.02得满分40
        
        # 夏普比率得分 (0-30分)
        sharpe_score = min(30, max(0, metrics["sharpe"] * 10))  # 夏普=3得满分30
        
        # 年化收益得分 (0-30分)
        annual_return_score = min(30, max(0, metrics["annual_return"] * 100))  # 年化收益30%得满分30
        
        total_score = ic_score + sharpe_score + annual_return_score
        logger.info(f"📊 因子质量得分: {total_score:.1f}/100")
        
        return total_score