#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Alpha因子生成器 - 超优化版
深度优化所有问题因子，确保数值稳定性和质量
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CryptoDataLoader:
    """Crypto数据加载器"""
    
    def __init__(self, data_path: str = "data/crypto"):
        self.data_path = data_path
        self.crypto_data = {}
        self.load_all_data()
    
    def load_all_data(self):
        """加载所有Crypto数据"""
        csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
        
        for file_path in csv_files:
            try:
                # 从文件名提取交易对
                filename = os.path.basename(file_path)
                symbol = filename.replace('.csv', '')
                
                # 读取数据
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # 计算收益率
                df['returns'] = df['close'].pct_change()
                
                self.crypto_data[symbol] = df
                print(f"加载Crypto数据: {symbol}, 数据量: {len(df)}")
                
            except Exception as e:
                print(f"加载文件 {file_path} 失败: {e}")
    
    def get_crypto_list(self) -> List[str]:
        """获取Crypto交易对列表"""
        return list(self.crypto_data.keys())
    
    def get_data_matrix(self, field: str = 'close') -> pd.DataFrame:
        """获取数据矩阵"""
        data_dict = {}
        for symbol, df in self.crypto_data.items():
            data_dict[symbol] = df[field]
        
        return pd.DataFrame(data_dict)
    
    def get_returns_matrix(self) -> pd.DataFrame:
        """获取收益率矩阵"""
        return self.get_data_matrix('returns')

class UltraFactorNormalizer:
    """超优化因子标准化器"""
    
    @staticmethod
    def ultra_winsorize(data: pd.DataFrame, limits: Tuple[float, float] = (0.005, 0.005)) -> pd.DataFrame:
        """超严格Winsorization处理，限制极值"""
        if data.empty:
            return data
        
        # 处理无穷大值
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # 对每列进行超严格Winsorization
        result = data.copy()
        for col in data.columns:
            values = data[col].dropna()
            if len(values) > 0:
                # 使用更严格的限制
                lower_limit = values.quantile(limits[0])
                upper_limit = values.quantile(1 - limits[1])
                result[col] = result[col].clip(lower=lower_limit, upper=upper_limit)
        
        return result.fillna(0)
    
    @staticmethod
    def ultra_normalize(data: pd.DataFrame, method: str = 'ultra_robust') -> pd.DataFrame:
        """超优化因子标准化接口"""
        if data.empty:
            return data
        
        # 处理无穷大值
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # 如果数据全为NaN，返回零矩阵
        if data.isna().all().all():
            return pd.DataFrame(0, index=data.index, columns=data.columns)
        
        # 先进行超严格Winsorization
        data = UltraFactorNormalizer.ultra_winsorize(data, limits=(0.005, 0.005))
        
        if method == 'ultra_robust':
            # 超稳健标准化
            median = data.median()
            mad = np.median(np.abs(data - median), axis=0)
            mad = np.where(mad == 0, 1, mad)
            result = (data - median) / (1.4826 * mad)
            # 再次限制范围
            result = result.clip(-5, 5)
            return result.fillna(0)
        elif method == 'ultra_zscore':
            # 超Z-score标准化
            mean = data.mean()
            std = data.std()
            std = np.where(std == 0, 1, std)
            result = (data - mean) / std
            # 再次限制范围
            result = result.clip(-5, 5)
            return result.fillna(0)
        elif method == 'ultra_rank':
            # 超排名标准化
            result = data.rank(pct=True)
            result = 2 * result - 1  # 转换为-1到1
            return result.fillna(0)
        else:
            return data.fillna(0)

class CryptoAlphaGFNGenerator:
    """基于GFlowNet的Crypto Alpha因子生成器 - 超优化版"""
    
    def __init__(self, data_loader: CryptoDataLoader):
        self.data_loader = data_loader
        self.window_size = 5
        self.normalizer = UltraFactorNormalizer()
        
    def generate_factors(self) -> Dict[str, pd.DataFrame]:
        """生成GFlowNet风格的crypto alpha因子 - 超优化版"""
        factors = {}
        
        # 获取数据矩阵
        close_data = self.data_loader.get_data_matrix('close')
        volume_data = self.data_loader.get_data_matrix('volume')
        high_data = self.data_loader.get_data_matrix('high')
        low_data = self.data_loader.get_data_matrix('low')
        
        # 数据预处理 - 超严格处理
        close_data = close_data.fillna(method='ffill').fillna(method='bfill').fillna(1)
        volume_data = volume_data.fillna(method='ffill').fillna(method='bfill').fillna(1)
        high_data = high_data.fillna(method='ffill').fillna(method='bfill').fillna(1)
        low_data = low_data.fillna(method='ffill').fillna(method='bfill').fillna(1)
        
        # 因子1: 价格动量因子 - 超优化版
        returns_data = close_data.pct_change().fillna(0)
        momentum_factor = returns_data.rolling(window=self.window_size).mean().fillna(0)
        momentum_factor = self.normalizer.ultra_normalize(momentum_factor, 'ultra_robust')
        factors['GFN_Crypto_Momentum'] = momentum_factor
        
        # 因子2: 价格波动因子 - 超优化版
        volatility_factor = returns_data.rolling(window=self.window_size).std().fillna(0)
        volatility_factor = self.normalizer.ultra_normalize(volatility_factor, 'ultra_robust')
        factors['GFN_Crypto_Volatility'] = volatility_factor
        
        # 因子3: 成交量价格相关性因子 - 超优化版
        volume_returns = volume_data.pct_change().fillna(0)
        volume_corr_factor = returns_data.rolling(window=self.window_size).corr(volume_returns).fillna(0)
        volume_corr_factor = self.normalizer.ultra_normalize(volume_corr_factor, 'ultra_robust')
        factors['GFN_Crypto_VolumeCorr'] = volume_corr_factor
        
        # 因子4: 对数价格因子 - 超优化版
        log_returns = np.log(close_data / close_data.shift(1)).fillna(0)
        log_returns = self.normalizer.ultra_normalize(log_returns, 'ultra_robust')
        factors['GFN_Crypto_LogPrice'] = log_returns
        
        # 因子5: 复合技术因子 - 超优化版
        price_norm = self.normalizer.ultra_normalize(close_data, 'ultra_rank')
        volume_norm = self.normalizer.ultra_normalize(volume_data, 'ultra_rank')
        composite_factor = price_norm * volume_norm
        composite_factor = self.normalizer.ultra_normalize(composite_factor, 'ultra_robust')
        factors['GFN_Crypto_Composite'] = composite_factor
        
        return factors

class CryptoAlphaAgentGenerator:
    """基于AlphaAgent的Crypto因子生成器 - 超优化版"""
    
    def __init__(self, data_loader: CryptoDataLoader):
        self.data_loader = data_loader
        self.normalizer = UltraFactorNormalizer()
    
    def generate_factors(self) -> Dict[str, pd.DataFrame]:
        """生成AlphaAgent风格的crypto因子 - 超优化版"""
        factors = {}
        
        close_data = self.data_loader.get_data_matrix('close')
        volume_data = self.data_loader.get_data_matrix('volume')
        high_data = self.data_loader.get_data_matrix('high')
        low_data = self.data_loader.get_data_matrix('low')
        
        # 数据预处理
        close_data = close_data.fillna(method='ffill').fillna(method='bfill').fillna(1)
        volume_data = volume_data.fillna(method='ffill').fillna(method='bfill').fillna(1)
        high_data = high_data.fillna(method='ffill').fillna(method='bfill').fillna(1)
        low_data = low_data.fillna(method='ffill').fillna(method='bfill').fillna(1)
        
        # 因子1: MACD因子 - 超优化版
        def calculate_macd_ultra(data, fast=12, slow=26, signal=9):
            ema_fast = data.ewm(span=fast).mean().fillna(0)
            ema_slow = data.ewm(span=slow).mean().fillna(0)
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean().fillna(0)
            result = macd_line - signal_line
            # 超优化：直接限制范围
            return result.clip(-3, 3)
        
        macd_factor = calculate_macd_ultra(close_data)
        macd_factor = self.normalizer.ultra_normalize(macd_factor, 'ultra_robust')
        factors['Agent_Crypto_MACD'] = macd_factor
        
        # 因子2: RSI因子 - 超优化版
        def calculate_rsi_ultra(data, window=14):
            delta = data.diff().fillna(0)
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean().fillna(0)
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean().fillna(0)
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            result = (rsi - 50) / 50  # 转换为-1到1
            return result.clip(-1, 1)  # 强制限制范围
        
        rsi_factor = calculate_rsi_ultra(close_data)
        factors['Agent_Crypto_RSI'] = rsi_factor
        
        # 因子3: 布林带因子 - 超优化版
        def calculate_bollinger_ultra(data, window=20, num_std=2):
            rolling_mean = data.rolling(window=window).mean().fillna(data.mean())
            rolling_std = data.rolling(window=window).std().fillna(data.std())
            upper_band = rolling_mean + (rolling_std * num_std)
            lower_band = rolling_mean - (rolling_std * num_std)
            bollinger = (data - lower_band) / (upper_band - lower_band + 1e-8)
            result = 2 * bollinger - 1  # 转换为-1到1
            return result.clip(-1, 1)  # 强制限制范围
        
        bollinger_factor = calculate_bollinger_ultra(close_data)
        factors['Agent_Crypto_Bollinger'] = bollinger_factor
        
        # 因子4: 成交量加权平均价格因子 - 超优化版
        def calculate_vwap_ultra(data, volume, high, low):
            typical_price = (high + low + data) / 3
            vwap = (typical_price * volume).rolling(window=20).sum() / (volume.rolling(window=20).sum() + 1e-8)
            result = (data - vwap) / (vwap + 1e-8)
            return result.clip(-3, 3).fillna(0)  # 限制范围
        
        vwap_factor = calculate_vwap_ultra(close_data, volume_data, high_data, low_data)
        vwap_factor = self.normalizer.ultra_normalize(vwap_factor, 'ultra_robust')
        factors['Agent_Crypto_VWAP'] = vwap_factor
        
        # 因子5: 动量反转因子 - 超优化版
        def calculate_momentum_reversal_ultra(data, short_window=5, long_window=20):
            short_ma = data.rolling(window=short_window).mean().fillna(data.mean())
            long_ma = data.rolling(window=long_window).mean().fillna(data.mean())
            result = (short_ma - long_ma) / (long_ma + 1e-8)
            return result.clip(-3, 3).fillna(0)  # 限制范围
        
        momentum_reversal_factor = calculate_momentum_reversal_ultra(close_data)
        momentum_reversal_factor = self.normalizer.ultra_normalize(momentum_reversal_factor, 'ultra_robust')
        factors['Agent_Crypto_MomentumReversal'] = momentum_reversal_factor
        
        return factors

class CryptoAlphaGenGenerator:
    """基于AlphaGen的Crypto因子生成器 - 超优化版，深度修复波动率回归因子"""
    
    def __init__(self, data_loader: CryptoDataLoader):
        self.data_loader = data_loader
        self.normalizer = UltraFactorNormalizer()
    
    def generate_factors(self) -> Dict[str, pd.DataFrame]:
        """生成AlphaGen风格的crypto因子 - 超优化版"""
        factors = {}
        
        close_data = self.data_loader.get_data_matrix('close')
        volume_data = self.data_loader.get_data_matrix('volume')
        high_data = self.data_loader.get_data_matrix('high')
        low_data = self.data_loader.get_data_matrix('low')
        
        # 数据预处理
        close_data = close_data.fillna(method='ffill').fillna(method='bfill').fillna(1)
        volume_data = volume_data.fillna(method='ffill').fillna(method='bfill').fillna(1)
        high_data = high_data.fillna(method='ffill').fillna(method='bfill').fillna(1)
        low_data = low_data.fillna(method='ffill').fillna(method='bfill').fillna(1)
        
        # 因子1: 价格相对强度因子 - 超优化版
        def calculate_relative_strength_ultra(data, window=20):
            ma = data.rolling(window=window).mean().fillna(data.mean())
            result = data / (ma + 1e-8) - 1
            return result.clip(-2, 2).fillna(0)  # 限制范围
        
        relative_strength_factor = calculate_relative_strength_ultra(close_data)
        relative_strength_factor = self.normalizer.ultra_normalize(relative_strength_factor, 'ultra_robust')
        factors['Gen_Crypto_RelativeStrength'] = relative_strength_factor
        
        # 因子2: 成交量异常因子 - 超优化版
        def calculate_volume_anomaly_ultra(volume, window=20):
            volume_ma = volume.rolling(window=window).mean().fillna(volume.mean())
            volume_std = volume.rolling(window=window).std().fillna(volume.std())
            result = (volume - volume_ma) / (volume_std + 1e-8)
            return result.clip(-3, 3).fillna(0)  # 限制范围
        
        volume_anomaly_factor = calculate_volume_anomaly_ultra(volume_data)
        volume_anomaly_factor = self.normalizer.ultra_normalize(volume_anomaly_factor, 'ultra_robust')
        factors['Gen_Crypto_VolumeAnomaly'] = volume_anomaly_factor
        
        # 因子3: 价格加速度因子 - 超优化版
        def calculate_price_acceleration_ultra(data, window=5):
            returns = data.pct_change().fillna(0)
            acceleration = returns.diff().fillna(0)
            result = acceleration.rolling(window=window).mean().fillna(0)
            return result.clip(-2, 2)  # 限制范围
        
        price_acceleration_factor = calculate_price_acceleration_ultra(close_data)
        price_acceleration_factor = self.normalizer.ultra_normalize(price_acceleration_factor, 'ultra_robust')
        factors['Gen_Crypto_PriceAcceleration'] = price_acceleration_factor
        
        # 因子4: 波动率回归因子 - 超深度优化版
        def calculate_volatility_reversion_ultra_optimized(data, window=20):
            returns = data.pct_change().fillna(0)
            volatility = returns.rolling(window=window).std().fillna(0)
            vol_ma = volatility.rolling(window=window).mean().fillna(0)
            
            # 超深度优化：多重限制策略
            # 1. 计算比率
            vol_ratio = (volatility + 1e-8) / (vol_ma + 1e-8)
            
            # 2. 对数变换
            vol_ratio = np.log(vol_ratio)
            
            # 3. 限制对数范围
            vol_ratio = vol_ratio.clip(-2, 2)
            
            # 4. tanh变换
            vol_ratio = np.tanh(vol_ratio)
            
            # 5. 最终强制限制
            vol_ratio = vol_ratio.clip(-1, 1)
            
            return vol_ratio
        
        volatility_reversion_factor = calculate_volatility_reversion_ultra_optimized(close_data).fillna(0)
        # 不再进行额外标准化，直接使用
        factors['Gen_Crypto_VolatilityReversion'] = volatility_reversion_factor
        
        # 因子5: 多时间框架动量因子 - 超优化版
        def calculate_multi_timeframe_momentum_ultra(data):
            momentum_5 = data.pct_change(5).fillna(0)
            momentum_10 = data.pct_change(10).fillna(0)
            momentum_20 = data.pct_change(20).fillna(0)
            result = (momentum_5 + momentum_10 + momentum_20) / 3
            return result.clip(-2, 2)  # 限制范围
        
        multi_momentum_factor = calculate_multi_timeframe_momentum_ultra(close_data)
        multi_momentum_factor = self.normalizer.ultra_normalize(multi_momentum_factor, 'ultra_robust')
        factors['Gen_Crypto_MultiMomentum'] = multi_momentum_factor
        
        return factors

class CryptoAlphaMinerGenerator:
    """基于AlphaMiner的Crypto因子生成器 - 超优化版"""
    
    def __init__(self, data_loader: CryptoDataLoader):
        self.data_loader = data_loader
        self.normalizer = UltraFactorNormalizer()
    
    def generate_factors(self) -> Dict[str, pd.DataFrame]:
        """生成AlphaMiner风格的crypto因子 - 超优化版"""
        factors = {}
        
        close_data = self.data_loader.get_data_matrix('close')
        volume_data = self.data_loader.get_data_matrix('volume')
        high_data = self.data_loader.get_data_matrix('high')
        low_data = self.data_loader.get_data_matrix('low')
        
        # 数据预处理
        close_data = close_data.fillna(method='ffill').fillna(method='bfill').fillna(1)
        volume_data = volume_data.fillna(method='ffill').fillna(method='bfill').fillna(1)
        high_data = high_data.fillna(method='ffill').fillna(method='bfill').fillna(1)
        low_data = low_data.fillna(method='ffill').fillna(method='bfill').fillna(1)
        
        # 因子1: 简单移动平均交叉因子 - 超优化版
        def calculate_sma_cross_ultra(data, short_window=10, long_window=30):
            sma_short = data.rolling(window=short_window).mean().fillna(data.mean())
            sma_long = data.rolling(window=long_window).mean().fillna(data.mean())
            result = (sma_short - sma_long) / (sma_long + 1e-8)
            return result.clip(-3, 3).fillna(0)  # 限制范围
        
        sma_cross_factor = calculate_sma_cross_ultra(close_data)
        sma_cross_factor = self.normalizer.ultra_normalize(sma_cross_factor, 'ultra_robust')
        factors['Miner_Crypto_SMACross'] = sma_cross_factor
        
        # 因子2: 指数移动平均因子 - 超优化版
        def calculate_ema_factor_ultra(data, window=20):
            ema = data.ewm(span=window).mean().fillna(data.mean())
            result = (data - ema) / (ema + 1e-8)
            return result.clip(-2, 2).fillna(0)  # 限制范围
        
        ema_factor = calculate_ema_factor_ultra(close_data)
        ema_factor = self.normalizer.ultra_normalize(ema_factor, 'ultra_robust')
        factors['Miner_Crypto_EMA'] = ema_factor
        
        # 因子3: 威廉指标因子 - 超优化版
        def calculate_williams_r_ultra(high, low, close, window=14):
            highest_high = high.rolling(window=window).max().fillna(high.max())
            lowest_low = low.rolling(window=window).min().fillna(low.min())
            williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-8) * -100
            result = williams_r / 100  # 转换为-1到1
            return result.clip(-1, 1).fillna(-0.5)  # 强制限制范围
        
        williams_r_factor = calculate_williams_r_ultra(high_data, low_data, close_data)
        factors['Miner_Crypto_WilliamsR'] = williams_r_factor
        
        # 因子4: 随机指标因子 - 超优化版
        def calculate_stochastic_ultra(high, low, close, k_window=14, d_window=3):
            lowest_low = low.rolling(window=k_window).min().fillna(low.min())
            highest_high = high.rolling(window=k_window).max().fillna(high.max())
            k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-8)
            d_percent = k_percent.rolling(window=d_window).mean().fillna(50)
            result = (k_percent - d_percent) / 100  # 转换为-1到1
            return result.clip(-1, 1).fillna(0)  # 强制限制范围
        
        stochastic_factor = calculate_stochastic_ultra(high_data, low_data, close_data)
        factors['Miner_Crypto_Stochastic'] = stochastic_factor
        
        # 因子5: 商品通道指数因子 - 超优化版
        def calculate_cci_ultra(high, low, close, window=20):
            typical_price = (high + low + close) / 3
            sma_tp = typical_price.rolling(window=window).mean().fillna(typical_price.mean())
            mad = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean()))).fillna(typical_price.std())
            cci = (typical_price - sma_tp) / (0.015 * (mad + 1e-8))
            result = cci.clip(-100, 100) / 100  # 限制在-1到1
            return result.clip(-1, 1).fillna(0)  # 强制限制范围
        
        cci_factor = calculate_cci_ultra(high_data, low_data, close_data)
        factors['Miner_Crypto_CCI'] = cci_factor
        
        return factors

class CryptoGeneticAlphaGenerator:
    """基于Genetic-Alpha的Crypto因子生成器 - 超优化版"""
    
    def __init__(self, data_loader: CryptoDataLoader):
        self.data_loader = data_loader
        self.normalizer = UltraFactorNormalizer()
    
    def generate_factors(self) -> Dict[str, pd.DataFrame]:
        """生成Genetic-Alpha风格的crypto因子 - 超优化版"""
        factors = {}
        
        close_data = self.data_loader.get_data_matrix('close')
        volume_data = self.data_loader.get_data_matrix('volume')
        high_data = self.data_loader.get_data_matrix('high')
        low_data = self.data_loader.get_data_matrix('low')
        
        # 数据预处理
        close_data = close_data.fillna(method='ffill').fillna(method='bfill').fillna(1)
        volume_data = volume_data.fillna(method='ffill').fillna(method='bfill').fillna(1)
        high_data = high_data.fillna(method='ffill').fillna(method='bfill').fillna(1)
        low_data = low_data.fillna(method='ffill').fillna(method='bfill').fillna(1)
        
        # 因子1: 遗传算法风格的复合因子 - 超优化版
        def calculate_genetic_composite_ultra(close, high, low, volume):
            price_position = (close - low) / (high - low + 1e-8)
            volume_norm = self.normalizer.ultra_normalize(volume, 'ultra_rank')
            result = price_position * volume_norm
            return result.clip(-2, 2).fillna(0)  # 限制范围
        
        genetic_composite_factor = calculate_genetic_composite_ultra(close_data, high_data, low_data, volume_data)
        genetic_composite_factor = self.normalizer.ultra_normalize(genetic_composite_factor, 'ultra_robust')
        factors['Genetic_Crypto_Composite'] = genetic_composite_factor
        
        # 因子2: 自适应移动平均因子 - 超优化版
        def calculate_adaptive_ma_ultra(data, window=20):
            ma = data.rolling(window=window).mean().fillna(data.mean())
            std = data.rolling(window=window).std().fillna(data.std())
            result = (data - ma) / (std + 1e-8)
            return result.clip(-2, 2).fillna(0)  # 限制范围
        
        adaptive_ma_factor = calculate_adaptive_ma_ultra(close_data)
        adaptive_ma_factor = self.normalizer.ultra_normalize(adaptive_ma_factor, 'ultra_robust')
        factors['Genetic_Crypto_AdaptiveMA'] = adaptive_ma_factor
        
        # 因子3: 进化动量因子 - 超优化版
        def calculate_evolutionary_momentum_ultra(data):
            momentum_1 = data.pct_change(1).fillna(0)
            momentum_5 = data.pct_change(5).fillna(0)
            momentum_10 = data.pct_change(10).fillna(0)
            result = 0.5 * momentum_1 + 0.3 * momentum_5 + 0.2 * momentum_10
            return result.clip(-2, 2)  # 限制范围
        
        evolutionary_momentum_factor = calculate_evolutionary_momentum_ultra(close_data)
        evolutionary_momentum_factor = self.normalizer.ultra_normalize(evolutionary_momentum_factor, 'ultra_robust')
        factors['Genetic_Crypto_EvolutionaryMomentum'] = evolutionary_momentum_factor
        
        # 因子4: 多变量因子 - 超优化版
        def calculate_multivariate_factor_ultra(close, high, low, volume):
            price_momentum = close.pct_change(5).fillna(0)
            volatility = close.pct_change().rolling(window=10).std().fillna(0)
            volume_momentum = volume.pct_change(5).fillna(0)
            result = (price_momentum + volatility + volume_momentum) / 3
            return result.clip(-2, 2)  # 限制范围
        
        multivariate_factor = calculate_multivariate_factor_ultra(close_data, high_data, low_data, volume_data)
        multivariate_factor = self.normalizer.ultra_normalize(multivariate_factor, 'ultra_robust')
        factors['Genetic_Crypto_Multivariate'] = multivariate_factor
        
        # 因子5: 最优因子 - 超优化版
        def calculate_optimal_factor_ultra(close, high, low, volume):
            rsi = self.calculate_rsi_ultra(close)
            bollinger = self.calculate_bollinger_ultra(close)
            volume_ratio = volume / (volume.rolling(window=20).mean() + 1e-8)
            result = (rsi + bollinger + volume_ratio) / 3
            return result.clip(-2, 2).fillna(0)  # 限制范围
        
        optimal_factor = calculate_optimal_factor_ultra(close_data, high_data, low_data, volume_data)
        optimal_factor = self.normalizer.ultra_normalize(optimal_factor, 'ultra_robust')
        factors['Genetic_Crypto_Optimal'] = optimal_factor
        
        return factors
    
    def calculate_rsi_ultra(self, data, window=14):
        """计算RSI - 超优化版"""
        delta = data.diff().fillna(0)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean().fillna(0)
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean().fillna(0)
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        result = (rsi - 50) / 50  # 转换为-1到1
        return result.clip(-1, 1)  # 强制限制范围
    
    def calculate_bollinger_ultra(self, data, window=20, num_std=2):
        """计算布林带位置 - 超优化版"""
        rolling_mean = data.rolling(window=window).mean().fillna(data.mean())
        rolling_std = data.rolling(window=window).std().fillna(data.std())
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        bollinger = (data - lower_band) / (upper_band - lower_band + 1e-8)
        result = 2 * bollinger - 1  # 转换为-1到1
        return result.clip(-1, 1)  # 强制限制范围

class CryptoAlphaFactorGenerator:
    """Crypto Alpha因子生成器主类 - 超优化版"""
    
    def __init__(self, data_path: str = "data/crypto"):
        self.data_loader = CryptoDataLoader(data_path)
        self.gfn_generator = CryptoAlphaGFNGenerator(self.data_loader)
        self.agent_generator = CryptoAlphaAgentGenerator(self.data_loader)
        self.gen_generator = CryptoAlphaGenGenerator(self.data_loader)
        self.miner_generator = CryptoAlphaMinerGenerator(self.data_loader)
        self.genetic_generator = CryptoGeneticAlphaGenerator(self.data_loader)
    
    def generate_all_factors(self) -> Dict[str, pd.DataFrame]:
        """生成所有因子 - 超优化版"""
        print("开始生成Crypto Alpha因子 - 超优化版...")
        
        all_factors = {}
        
        # 生成各类因子
        print("生成GFlowNet因子...")
        gfn_factors = self.gfn_generator.generate_factors()
        all_factors.update(gfn_factors)
        print(f"GFlowNet因子: {list(gfn_factors.keys())}")
        
        print("生成AlphaAgent因子...")
        agent_factors = self.agent_generator.generate_factors()
        all_factors.update(agent_factors)
        print(f"AlphaAgent因子: {list(agent_factors.keys())}")
        
        print("生成AlphaGen因子...")
        gen_factors = self.gen_generator.generate_factors()
        all_factors.update(gen_factors)
        print(f"AlphaGen因子: {list(gen_factors.keys())}")
        
        print("生成AlphaMiner因子...")
        miner_factors = self.miner_generator.generate_factors()
        all_factors.update(miner_factors)
        print(f"AlphaMiner因子: {list(miner_factors.keys())}")
        
        print("生成Genetic-Alpha因子...")
        genetic_factors = self.genetic_generator.generate_factors()
        all_factors.update(genetic_factors)
        print(f"Genetic-Alpha因子: {list(genetic_factors.keys())}")
        
        print(f"共生成 {len(all_factors)} 个因子")
        return all_factors
    
    def save_factors(self, factors: Dict[str, pd.DataFrame], output_file: str = "alpha_pool/crypto_alpha_factors_ultra_optimized.csv"):
        """保存因子数据"""
        print(f"保存因子数据到: {output_file}")
        
        # 转换为长格式
        factor_data = []
        for factor_name, factor_df in factors.items():
            print(f"处理因子: {factor_name}, 形状: {factor_df.shape}")
            count = 0
            for date in factor_df.index:
                for crypto in factor_df.columns:
                    value = factor_df.loc[date, crypto]
                    if not pd.isna(value) and not np.isinf(value):
                        factor_data.append({
                            'date': date,
                            'crypto': crypto,
                            'factor_name': factor_name,
                            'factor_value': value
                        })
                        count += 1
            print(f"  有效记录数: {count}")
        
        # 创建DataFrame并保存
        result_df = pd.DataFrame(factor_data)
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"因子数据已保存，共 {len(result_df)} 条记录")
        
        # 生成统计信息
        stats_data = []
        for factor_name, factor_df in factors.items():
            values = factor_df.values.flatten()
            values = values[~pd.isna(values)]
            values = values[~np.isinf(values)]
            if len(values) > 0:
                stats_data.append({
                    'factor_name': factor_name,
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                })
        
        stats_df = pd.DataFrame(stats_data)
        stats_file = output_file.replace('.csv', '_stats.csv')
        stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
        
        print(f"统计信息已保存到: {stats_file}")
        
        return result_df, stats_df

def main():
    """主函数"""
    print("=" * 60)
    print("Crypto Alpha因子生成器 - 超优化版")
    print("深度优化所有问题因子，确保数值稳定性和质量")
    print("=" * 60)
    
    # 创建生成器
    generator = CryptoAlphaFactorGenerator()
    
    # 生成所有因子
    factors = generator.generate_all_factors()
    
    # 保存因子数据
    result_df, stats_df = generator.save_factors(factors)
    
    # 显示统计信息
    print("\n因子统计信息:")
    print("=" * 60)
    print(stats_df.to_string(index=False))
    
    print(f"\n因子生成完成！")
    print(f"共生成 {len(factors)} 个因子")
    print(f"数据记录数: {len(result_df)}")

if __name__ == "__main__":
    main()

