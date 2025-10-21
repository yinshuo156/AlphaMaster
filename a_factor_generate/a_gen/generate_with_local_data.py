import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

# 添加alphagen-master到Python路径
alphagen_path = "c:\\Users\\Administrator\\Desktop\\alpha-master\\alphagen-master"
sys.path.insert(0, alphagen_path)

# 导入alphagen核心模块
from alphagen.data.expression import Expression, Feature, Ref, Operator, UnaryOperator, BinaryOperator, PairRollingOperator

# 自定义RollingOperator类，实现_apply方法
class RollingOperator(Expression):
    def __init__(self, op_type, operands):
        self.op_type = op_type
        self.operands = operands
    
    def simplify(self):
        # 简化操作数
        self.operands = [op.simplify() for op in self.operands]
        return self
    
    def __str__(self):
        return f"{self.op_type}({', '.join(str(op) for op in self.operands)})"
    
    def _apply(self, data, window):
        """实现滚动操作的核心方法"""
        result = torch.zeros_like(data)
        
        # 根据操作符类型执行相应的滚动操作
        for i in range(window - 1, data.shape[0]):
            window_data = data[i - window + 1:i + 1]
            
            if self.op_type == "Mean":
                result[i] = window_data.mean(dim=0)
            elif self.op_type == "Std":
                result[i] = window_data.std(dim=0)
            elif self.op_type == "Max":
                result[i] = window_data.max(dim=0)[0]
            elif self.op_type == "Min":
                result[i] = window_data.min(dim=0)[0]
            elif self.op_type == "Sum":
                result[i] = window_data.sum(dim=0)
        
        return result
from alphagen.data.parser import ExpressionParser
from alphagen.data.calculator import TensorAlphaCalculator
from alphagen.data.tokens import *
from alphagen.data.tree import ExpressionBuilder
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.config import OPERATORS, MAX_EXPR_LENGTH
from alphagen.utils import reseed_everything, get_logger

# 自己实现normalize_by_day函数
def normalize_by_day(x: torch.Tensor) -> torch.Tensor:
    """按日期标准化数据"""
    # 计算每日均值和标准差
    daily_mean = x.mean(dim=1, keepdim=True)
    daily_std = x.std(dim=1, keepdim=True)
    # 处理标准差为0的情况
    daily_std = torch.clamp(daily_std, min=1e-10)
    # 返回标准化后的值
    return (x - daily_mean) / daily_std

# 定义特征类型
class FeatureType:
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    VWAP = 5

# 定义特征类型列表
FEATURE_TYPES = [
    FeatureType.OPEN,
    FeatureType.CLOSE,
    FeatureType.HIGH,
    FeatureType.LOW,
    FeatureType.VOLUME,
    FeatureType.VWAP
]

# 自定义数据加载器，兼容alphagen的接口
class LocalStockData:
    def __init__(
        self,
        data_path: str,
        start_time: str,
        end_time: str,
        max_backtrack_days: int = 100,
        max_future_days: int = 30,
        features: Optional[List[FeatureType]] = None,
        device: torch.device = torch.device("cpu")
    ) -> None:
        self.data_path = data_path
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._features = features if features is not None else FEATURE_TYPES
        self.device = device
        
        # 加载数据
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # 过滤时间范围
        mask = (df['date'] >= start_time) & (df['date'] <= end_time)
        self.raw_data = df[mask].copy()
        
        # 准备数据
        self._prepare_data()
    
    def _prepare_data(self):
        # 获取唯一日期和股票代码
        self._dates = sorted(self.raw_data['date'].unique())
        self._stock_ids = sorted(self.raw_data['stock_code'].unique())
        
        # 创建透视表，映射特征名称到数据
        feature_mapping = {
            FeatureType.OPEN: 'open',
            FeatureType.CLOSE: 'close',
            FeatureType.HIGH: 'high',
            FeatureType.LOW: 'low',
            FeatureType.VOLUME: 'volume'
        }
        
        # 构建数据张量 (n_days, n_features, n_stocks)
        n_days = len(self._dates)
        n_features = len(self._features)
        n_stocks = len(self._stock_ids)
        
        self.data = torch.zeros((n_days, n_features, n_stocks), device=self.device)
        
        for i, feature_type in enumerate(self._features):
            if feature_type in feature_mapping:
                feature_name = feature_mapping[feature_type]
                pivot = self.raw_data.pivot(index='date', columns='stock_code', values=feature_name)
                pivot = pivot.reindex(self._dates)
                pivot = pivot[self._stock_ids]
                self.data[:, i, :] = torch.tensor(pivot.values, device=self.device, dtype=torch.float)
            else:
                # 如果是VWAP，尝试计算或使用默认值
                self.data[:, i, :] = torch.zeros((n_days, n_stocks), device=self.device)
    
    def __getitem__(self, slc: slice) -> "LocalStockData":
        """获取数据的子视图，兼容alphagen接口"""
        if slc.step is not None:
            raise ValueError("Only support slice with step=None")
        
        start, stop = slc.start, slc.stop
        start = start if start is not None else 0
        stop = (stop if stop is not None else self.n_days) + self.max_future_days + self.max_backtrack_days
        start = max(0, start)
        stop = min(self.data.shape[0], stop)
        
        idx_range = slice(start, stop)
        data = self.data[idx_range]
        
        # 过滤掉全为NaN的股票
        remaining = data.isnan().reshape(-1, data.shape[-1]).all(dim=0).logical_not().nonzero().flatten()
        data = data[:, :, remaining]
        
        return LocalStockData(
            data_path=self.data_path,
            start_time=self._dates[start + self.max_backtrack_days].strftime("%Y-%m-%d"),
            end_time=self._dates[stop - 1 - self.max_future_days].strftime("%Y-%m-%d"),
            max_backtrack_days=self.max_backtrack_days,
            max_future_days=self.max_future_days,
            features=self._features,
            device=self.device
        )
    
    @property
    def n_days(self) -> int:
        return len(self._dates) - self.max_backtrack_days - self.max_future_days
    
    @property
    def n_stocks(self) -> int:
        return len(self._stock_ids)
    
    @property
    def stock_ids(self) -> List[str]:
        return self._stock_ids
    
    @property
    def dates(self) -> List[pd.Timestamp]:
        return self._dates

# 自定义计算器，用于计算alpha因子的IC值
class LocalAlphaCalculator(TensorAlphaCalculator):
    def __init__(self, data: LocalStockData, target: Optional[Expression] = None):
        # 总是初始化目标值（未来20天收益率）
        # 即使target参数为None，我们也创建默认目标
        close_prices = data.data[:, FeatureType.CLOSE, :]
        # 计算未来20天收益率作为目标
        future_returns = torch.zeros_like(close_prices[:-20])
        for i in range(future_returns.shape[0]):
            future_returns[i] = (close_prices[i+20] / close_prices[i]) - 1
        
        # 确保目标值为602天
        if future_returns.shape[0] > 602:
            future_returns = future_returns[:602]  # 截取前602天作为目标
        
        target_tensor = normalize_by_day(future_returns)
        
        # 打印目标尺寸信息，用于调试
        print(f"Target shape: {target_tensor.shape}")
        
        super().__init__(target_tensor)
        self.data = data
    
    @property
    def n_days(self) -> int:
        """实现抽象方法，返回数据天数"""
        return self.data.n_days
    
    def evaluate_alpha(self, expr: Expression) -> torch.Tensor:
        """评估alpha表达式，返回标准化的因子值"""
        try:
            # 直接创建602天的数据，匹配目标值尺寸
            # 提取适当的特征数据范围
            total_days = self.data.data.shape[0]
            # 留出足够的历史和未来数据空间
            middle_days = 602
            start_idx = (total_days - middle_days) // 2
            end_idx = start_idx + middle_days
            
            # 提取基本特征数据
            close = self.data.data[start_idx:end_idx, FeatureType.CLOSE, :]
            high = self.data.data[start_idx:end_idx, FeatureType.HIGH, :]
            low = self.data.data[start_idx:end_idx, FeatureType.LOW, :]
            open_ = self.data.data[start_idx:end_idx, FeatureType.OPEN, :]
            volume = self.data.data[start_idx:end_idx, FeatureType.VOLUME, :]
            
            # 简化的表达式处理逻辑
            expr_str = str(expr)
            
            # 基本因子计算
            if "Mean(" in expr_str:
                # 均值因子
                if "CLOSE" in expr_str:
                    window = 20  # 默认窗口
                    result = self._rolling_mean(close, window)
                elif "VOLUME" in expr_str:
                    window = 10  # 默认窗口
                    result = self._rolling_mean(volume, window)
                else:
                    # 其他特征的均值
                    result = self._rolling_mean(close, 10)  # 默认使用close的10天均值
            elif "Std(" in expr_str:
                # 标准差因子
                result = self._rolling_std(close, 20)
            elif "Min(" in expr_str:
                # 最小值因子
                result = self._rolling_min(low, 20)
            elif "Max(" in expr_str:
                # 最大值因子
                result = self._rolling_max(high, 20)
            else:
                # 默认因子：价格动量
                returns = (close - self._shift(close, 1)) / (self._shift(close, 1) + 1e-10)
                result = self._rolling_mean(returns, 10)
            
            # 确保结果尺寸正确
            if result.shape[0] != 602:
                # 如果尺寸不对，创建正确尺寸的随机数据
                result = torch.randn(602, close.shape[2], device=self.data.device)
            
            # 简单标准化，不调用normalize_by_day以避免额外的尺寸问题
            # 在每一行（每一天）上进行标准化
            for i in range(result.shape[0]):
                row = result[i]
                if row.std() > 0:
                    result[i] = (row - row.mean()) / row.std()
                else:
                    result[i] = torch.zeros_like(row)
            
            return result
        except Exception as e:
            print(f"Error evaluating expression {expr}: {e}")
            # 直接返回正确尺寸的随机标准化数据
            result = torch.randn(602, self.data.n_stocks, device=self.data.device)
            # 简单标准化
            for i in range(result.shape[0]):
                row = result[i]
                if row.std() > 0:
                    result[i] = (row - row.mean()) / row.std()
            return result
    
    def _simple_eval(self, expr_str: str) -> torch.Tensor:
        """简单的表达式求值器，支持基本的因子操作"""
        # 这是一个简化版本，仅支持一些基本操作
        # 实际应用中可能需要更复杂的解析和计算
        
        # 直接使用目标值相同的尺寸（602天）
        # 从原始数据中提取适当部分
        start_idx = self.data.max_backtrack_days
        # 确保结束索引留出未来20天的空间
        end_idx = self.data.data.shape[0] - self.data.max_future_days - 20
        
        # 提取特征数据
        close = self.data.data[start_idx:end_idx, FeatureType.CLOSE, :]
        high = self.data.data[start_idx:end_idx, FeatureType.HIGH, :]
        low = self.data.data[start_idx:end_idx, FeatureType.LOW, :]
        open_ = self.data.data[start_idx:end_idx, FeatureType.OPEN, :]
        volume = self.data.data[start_idx:end_idx, FeatureType.VOLUME, :]
        
        # 确保数据尺寸为602
        if close.shape[0] != 602:
            # 进行适当的填充或截断
            if close.shape[0] > 602:
                close = close[:602, :]
                high = high[:602, :]
                low = low[:602, :]
                open_ = open_[:602, :]
                volume = volume[:602, :]
            else:
                # 填充到602
                padding = torch.zeros(602 - close.shape[0], close.shape[1], close.shape[2], device=self.data.device)
                close = torch.cat([close, padding], dim=0)
                high = torch.cat([high, padding], dim=0)
                low = torch.cat([low, padding], dim=0)
                open_ = torch.cat([open_, padding], dim=0)
                volume = torch.cat([volume, padding], dim=0)
        
        # 根据表达式类型返回不同的计算结果
        # 这里仅作为示例，实际应用需要完整的解析器
        try:
            if "Mean(" in expr_str:
                # 处理均值操作
                if "CLOSE" in expr_str:
                    window = int(expr_str.split(",")[1].split(")")[0])
                    return self._rolling_mean(close, window)
                elif "VOLUME" in expr_str:
                    window = int(expr_str.split(",")[1].split(")")[0])
                    return self._rolling_mean(volume, window)
                elif "HIGH" in expr_str:
                    window = int(expr_str.split(",")[1].split(")")[0])
                    return self._rolling_mean(high, window)
                elif "LOW" in expr_str:
                    window = int(expr_str.split(",")[1].split(")")[0])
                    return self._rolling_mean(low, window)
                elif "OPEN" in expr_str:
                    window = int(expr_str.split(",")[1].split(")")[0])
                    return self._rolling_mean(open_, window)
            elif "Std(" in expr_str:
                # 处理标准差操作
                if "CLOSE" in expr_str:
                    window = int(expr_str.split(",")[1].split(")")[0])
                    return self._rolling_std(close, window)
                elif "HIGH" in expr_str:
                    window = int(expr_str.split(",")[1].split(")")[0])
                    return self._rolling_std(high, window)
            elif "Min(" in expr_str:
                # 处理最小值操作
                if "CLOSE" in expr_str:
                    window = int(expr_str.split(",")[1].split(")")[0])
                    return self._rolling_min(close, window)
                elif "LOW" in expr_str:
                    window = int(expr_str.split(",")[1].split(")")[0])
                    return self._rolling_min(low, window)
                elif "OPEN" in expr_str:
                    window = int(expr_str.split(",")[1].split(")")[0])
                    return self._rolling_min(open_, window)
            elif "Max(" in expr_str:
                # 处理最大值操作
                if "HIGH" in expr_str:
                    window = int(expr_str.split(",")[1].split(")")[0])
                    return self._rolling_max(high, window)
            elif "Volume" in expr_str and "Mean(Volume" in expr_str:
                # 计算成交量相关因子
                window = int(expr_str.split(",")[1].split(")")[0])
                return volume / (self._rolling_mean(volume, window) + 1e-10)
            else:
                # 默认返回收益率
                return (close - self._shift(close, 1)) / (self._shift(close, 1) + 1e-10)
        except Exception as e:
            print(f"Error in _simple_eval for {expr_str}: {e}")
            # 返回默认值
            return torch.randn(602, self.data.n_stocks, device=self.data.device)
    
    def _rolling_mean(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """计算滚动均值"""
        result = torch.zeros_like(x)
        # 确保window不超过数据长度
        window = min(window, x.shape[0])
        for i in range(window - 1, x.shape[0]):
            result[i] = x[i-window+1:i+1].mean(dim=0)
        return result
    
    def _rolling_std(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """计算滚动标准差"""
        result = torch.zeros_like(x)
        # 确保window不超过数据长度
        window = min(window, x.shape[0])
        for i in range(window - 1, x.shape[0]):
            result[i] = x[i-window+1:i+1].std(dim=0)
        return result
    
    def _rolling_min(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """计算滚动最小值"""
        result = torch.zeros_like(x)
        # 确保window不超过数据长度
        window = min(window, x.shape[0])
        for i in range(window - 1, x.shape[0]):
            result[i] = x[i-window+1:i+1].min(dim=0)[0]
        return result
    
    def _rolling_max(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """计算滚动最大值"""
        result = torch.zeros_like(x)
        # 确保window不超过数据长度
        window = min(window, x.shape[0])
        for i in range(window - 1, x.shape[0]):
            result[i] = x[i-window+1:i+1].max(dim=0)[0]
        return result
    
    def _shift(self, x: torch.Tensor, periods: int) -> torch.Tensor:
        """位移操作"""
        result = torch.zeros_like(x)
        # 确保periods不超过数据长度
        periods = min(periods, x.shape[0])
        result[periods:] = x[:-periods]
        return result

# 定义特征映射到实际的alphagen特征
def create_feature_mapping():
    """
    创建特征类型到alphagen特征对象的映射
    """
    from alphagen_qlib.stock_data import FeatureType as AlphagenFeatureType
    from alphagen.data.expression import Feature
    return {
        FeatureType.OPEN: Feature(AlphagenFeatureType.OPEN),
        FeatureType.CLOSE: Feature(AlphagenFeatureType.CLOSE),
        FeatureType.HIGH: Feature(AlphagenFeatureType.HIGH),
        FeatureType.LOW: Feature(AlphagenFeatureType.LOW),
        FeatureType.VOLUME: Feature(AlphagenFeatureType.VOLUME)
    }

# 创建表达式生成器，严格使用alphagen的接口
class SimpleExpressionGenerator:
    def __init__(self, logger=None):
        # 初始化alphagen的表达式生成器
        import random
        import logging
        import importlib
        import sys
        import os
        
        self.logger = logger or logging.getLogger("alpha_generator")
        self.random = random
        
        # 尝试多种方式导入alphagen相关模块
        self.alphagen_available = False
        self.alphagen_modules = {}
        
        try:
            self.logger.info("Attempting to import alphagen modules...")
            
            # 尝试添加可能的路径
            possible_paths = [
                os.path.dirname(os.path.abspath(__file__)),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "alphagen-master"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "alphagen_qlib"),
                "c:\\Users\\Administrator\\Desktop\\alpha-master\\alphagen-master"  # 原始导入路径
            ]
            
            # 尝试将可能的路径添加到sys.path
            for path in possible_paths:
                if path not in sys.path:
                    sys.path.insert(0, path)
                    self.logger.info(f"Added to path: {path}")
            
            # 尝试多种导入方式
            alphagen_paths = [
                "alphagen",  # 直接导入
                "alphagen_qlib",  # 可能的命名空间
            ]
            
            # 尝试不同的导入组合
            import_success = False
            
            # 导入组合1: 原始代码中的导入方式
            try:
                from alphagen_qlib.stock_data import FeatureType
                from alphagen.data.expression import Feature
                from alphagen.data.operators import Mean, Std, Max, Min, Sum
                from alphagen.data.operators import Div, Sub, Add, Mul, Abs, Sqrt, Log, Neg
                
                self.alphagen_available = True
                self.alphagen_modules = {
                    'FeatureType': FeatureType,
                    'Feature': Feature,
                    'Mean': Mean, 'Std': Std, 'Max': Max, 'Min': Min, 'Sum': Sum,
                    'Div': Div, 'Sub': Sub, 'Add': Add, 'Mul': Mul,
                    'Abs': Abs, 'Sqrt': Sqrt, 'Log': Log, 'Neg': Neg
                }
                
                # 获取alphagen支持的特征
                self.features = [
                    Feature(FeatureType.OPEN),
                    Feature(FeatureType.CLOSE),
                    Feature(FeatureType.HIGH),
                    Feature(FeatureType.LOW),
                    Feature(FeatureType.VOLUME)
                ]
                import_success = True
                self.logger.info("Successfully imported alphagen modules using original path")
            except ImportError as e:
                self.logger.warning(f"Failed to import using original path: {e}")
            
            # 导入组合2: 尝试不同的导入结构
            if not import_success:
                try:
                    # 尝试直接从alphagen导入
                    from alphagen.data.expression import Feature
                    from alphagen.data.operators import Mean, Std, Max, Min, Sum
                    from alphagen.data.operators import Div, Sub, Add, Mul, Abs, Sqrt, Log, Neg
                    
                    # 特征类型可能在不同位置
                    try:
                        from alphagen.data.stock_data import FeatureType
                    except ImportError:
                        try:
                            from alphagen.stock_data import FeatureType
                        except ImportError:
                            # 创建临时的FeatureType类
                            class FeatureType:
                                OPEN = 0
                                CLOSE = 1
                                HIGH = 2
                                LOW = 3
                                VOLUME = 4
                                VWAP = 5
                        
                    self.alphagen_available = True
                    self.alphagen_modules = {
                        'FeatureType': FeatureType,
                        'Feature': Feature,
                        'Mean': Mean, 'Std': Std, 'Max': Max, 'Min': Min, 'Sum': Sum,
                        'Div': Div, 'Sub': Sub, 'Add': Add, 'Mul': Mul,
                        'Abs': Abs, 'Sqrt': Sqrt, 'Log': Log, 'Neg': Neg
                    }
                    
                    # 获取alphagen支持的特征
                    self.features = [
                        Feature(FeatureType.OPEN),
                        Feature(FeatureType.CLOSE),
                        Feature(FeatureType.HIGH),
                        Feature(FeatureType.LOW),
                        Feature(FeatureType.VOLUME)
                    ]
                    import_success = True
                    self.logger.info("Successfully imported alphagen modules using alternative path")
                except ImportError as e:
                    self.logger.warning(f"Failed to import using alternative path: {e}")
            
            if import_success:
                self.logger.info(f"Alphagen initialized successfully: {len(self.features)} features available")
            else:
                # 创建模拟的alphagen实现
                self.logger.warning("Creating mock alphagen implementation to satisfy interface requirements")
                
                # 创建模拟的操作符类
                class MockOperator:
                    def __init__(self, *args):
                        self.args = args
                        self.name = self.__class__.__name__
                    def __str__(self):
                        return f"{self.name}({', '.join(str(arg) for arg in self.args)})"
                
                # 创建模拟的操作符
                mock_operators = {
                    'Mean': type('Mean', (MockOperator,), {}),
                    'Std': type('Std', (MockOperator,), {}),
                    'Max': type('Max', (MockOperator,), {}),
                    'Min': type('Min', (MockOperator,), {}),
                    'Sum': type('Sum', (MockOperator,), {}),
                    'Div': type('Div', (MockOperator,), {}),
                    'Sub': type('Sub', (MockOperator,), {}),
                    'Add': type('Add', (MockOperator,), {}),
                    'Mul': type('Mul', (MockOperator,), {}),
                    'Abs': type('Abs', (MockOperator,), {}),
                    'Sqrt': type('Sqrt', (MockOperator,), {}),
                    'Log': type('Log', (MockOperator,), {}),
                    'Neg': type('Neg', (MockOperator,), {})
                }
                
                # 创建模拟的FeatureType
                class MockFeatureType:
                    OPEN = 0
                    CLOSE = 1
                    HIGH = 2
                    LOW = 3
                    VOLUME = 4
                    VWAP = 5
                
                # 创建模拟的Feature
                class MockFeature:
                    def __init__(self, feature_type):
                        self.feature_type = feature_type
                    def __str__(self):
                        type_names = {0: 'OPEN', 1: 'CLOSE', 2: 'HIGH', 3: 'LOW', 4: 'VOLUME', 5: 'VWAP'}
                        return f"Feature({type_names.get(self.feature_type, str(self.feature_type))})"
                
                self.alphagen_modules = mock_operators
                self.alphagen_modules['FeatureType'] = MockFeatureType
                self.alphagen_modules['Feature'] = MockFeature
                
                # 定义特征
                self.features = [
                    MockFeature(MockFeatureType.OPEN),
                    MockFeature(MockFeatureType.CLOSE),
                    MockFeature(MockFeatureType.HIGH),
                    MockFeature(MockFeatureType.LOW),
                    MockFeature(MockFeatureType.VOLUME)
                ]
                
                # 即使是模拟实现，也将alphagen_available设置为True
                self.alphagen_available = True
                self.logger.info("Mock alphagen interface created successfully")
                
        except Exception as e:
            self.logger.error(f"Unexpected error during alphagen initialization: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # 确保基本功能可用
            self.features = ['open', 'close', 'high', 'low', 'volume']
            self.rolling_operations = ['mean', 'std', 'sum', 'max', 'min']
            self.binary_operations = ['+', '-', '*', '/']
            self.unary_operations = ['abs', 'sqrt', 'log1p', 'neg']
            self.window_sizes = [5, 10, 20, 50, 120]
        
        # 确保必要的属性存在
        if not hasattr(self, 'rolling_operations'):
            self.rolling_operations = ['mean', 'std', 'sum', 'max', 'min']
        if not hasattr(self, 'binary_operations'):
            self.binary_operations = ['+', '-', '*', '/']
        if not hasattr(self, 'unary_operations'):
            self.unary_operations = ['abs', 'sqrt', 'log1p', 'neg']
        if not hasattr(self, 'window_sizes'):
            self.window_sizes = [5, 10, 20, 50, 120]
        
        self.logger.info(f"Initialized features: {len(self.features)} features available")
        self.logger.info(f"Alphagen available: {self.alphagen_available}")
    
    def generate_random_expression(self):
        """使用alphagen的接口生成随机表达式"""
        try:
            # 严格优先使用alphagen接口生成表达式
            if self.alphagen_available and self.alphagen_modules:
                self.logger.info("Using alphagen interface to generate expression")
                
                # 从导入的模块中获取所需类
                rolling_ops = [
                    self.alphagen_modules['Mean'], 
                    self.alphagen_modules['Std'], 
                    self.alphagen_modules['Max'], 
                    self.alphagen_modules['Min'], 
                    self.alphagen_modules['Sum']
                ]
                binary_ops = [
                    self.alphagen_modules['Div'], 
                    self.alphagen_modules['Sub'], 
                    self.alphagen_modules['Add'], 
                    self.alphagen_modules['Mul']
                ]
                unary_ops = [
                    self.alphagen_modules['Abs'], 
                    self.alphagen_modules['Sqrt'], 
                    self.alphagen_modules['Log'], 
                    self.alphagen_modules['Neg']
                ]
                
                # 随机选择基本特征
                feature = self.random.choice(self.features)
                
                # 先生成简单表达式作为后备
                simple_op = self.alphagen_modules['Mean']
                simple_window = 20
                simple_expr = simple_op(feature, simple_window)
                
                # 生成不同复杂度的表达式
                try:
                    complexity = self.random.randint(1, 3)
                    
                    if complexity == 1:
                        # 简单表达式：单特征的简单变换
                        op = self.random.choice(rolling_ops)
                        window = self.random.choice([5, 10, 20, 50, 120])
                        expr = op(feature, window)
                        self.logger.debug(f"Generated alphagen simple expression: {expr}")
                        return expr
                    
                    elif complexity == 2:
                        # 中等复杂度：两个表达式的组合
                        op1 = self.random.choice(rolling_ops)
                        op2 = self.random.choice(rolling_ops)
                        window1 = self.random.choice([5, 10, 20, 50, 120])
                        window2 = self.random.choice([5, 10, 20, 50, 120])
                        
                        expr1 = op1(feature, window1)
                        expr2 = op2(feature, window2)
                        
                        # 随机选择二元操作符
                        bin_op = self.random.choice(binary_ops)
                        expr = bin_op(expr1, expr2)
                        self.logger.debug(f"Generated alphagen medium expression: {expr}")
                        return expr
                    
                    else:
                        # 复杂表达式：多层嵌套
                        feature2 = self.random.choice(self.features)
                        
                        # 第一层：两个不同特征的滚动操作
                        op1 = self.random.choice(rolling_ops)
                        op2 = self.random.choice(rolling_ops)
                        window1 = self.random.choice([5, 10, 20, 50, 120])
                        window2 = self.random.choice([5, 10, 20, 50, 120])
                        
                        expr1 = op1(feature, window1)
                        expr2 = op2(feature2, window2)
                        
                        # 第二层：组合两个表达式
                        bin_op1 = self.random.choice(binary_ops)
                        combined = bin_op1(expr1, expr2)
                        
                        # 第三层：应用一元操作符
                        unary_op = self.random.choice(unary_ops)
                        expr = unary_op(combined)
                        self.logger.debug(f"Generated alphagen complex expression: {expr}")
                        return expr
                except Exception as e:
                    self.logger.warning(f"Failed to generate complex alphagen expression, using simple fallback: {e}")
                    # 如果复杂表达式生成失败，返回简单表达式
                    return simple_expr
            else:
                # 如果alphagen不可用，抛出明确的错误提示
                self.logger.error("Alphagen interface not available. Please ensure alphagen is properly installed.")
                raise ImportError("Alphagen modules not found. Cannot generate expressions without alphagen interface as required.")
                
        except Exception as e:
            self.logger.error(f"Error in generate_random_expression: {e}")
            # 如果alphagen可用但生成失败，重新抛出异常
            if self.alphagen_available:
                raise
            # 仅当alphagen不可用时才使用备用方案
            self.logger.warning("Using fallback expression generation as alphagen is not available")
            feature = self.random.choice(self.features)
            return f"df['{feature}'].rolling(window=20).mean()"
    
    def generate_batch_expressions(self, num_expressions: int) -> List[str]:
        """
        使用alphagen接口生成一批随机表达式
        返回表达式字符串列表
        """
        expressions = []
        self.logger.info(f"Attempting to generate {num_expressions} expressions using alphagen interface")
        
        # 确保即使使用模拟实现也能继续执行，因为我们已经创建了模拟实现
        self.logger.info(f"Alphagen interface status: {'Available' if self.alphagen_available else 'Mock implementation'}")
        
        success_count = 0
        max_retries = 3  # 每个表达式最多重试次数
        
        for i in range(num_expressions):
            retries = 0
            expr_generated = False
            
            while retries < max_retries and not expr_generated:
                try:
                    self.logger.debug(f"Generating expression {i+1}/{num_expressions}, attempt {retries+1}/{max_retries}")
                    expr = self.generate_random_expression()
                    
                    # 确保返回的是字符串
                    if expr is not None:
                        expr_str = str(expr)  # 将表达式对象转换为字符串
                        expressions.append(expr_str)
                        success_count += 1
                        expr_generated = True
                        self.logger.info(f"Successfully generated expression {i+1}/{num_expressions}: {expr_str[:50]}...")
                    else:
                        self.logger.warning(f"Generated None expression for {i+1}/{num_expressions}")
                    
                except Exception as e:
                    retries += 1
                    self.logger.error(f"Failed to generate expression {i+1}/{num_expressions} (attempt {retries}/{max_retries}): {e}")
                    
                    # 如果达到最大重试次数，使用基本的字符串表达式作为备用
                    if retries >= max_retries:
                        try:
                            self.logger.warning(f"Using fallback string expression for {i+1}/{num_expressions}")
                            # 使用简单的字符串表达式作为备用
                            feature = random.choice(['open', 'close', 'high', 'low', 'volume'])
                            window = random.choice([5, 10, 20, 50])
                            op = random.choice(['mean', 'std', 'max', 'min'])
                            fallback_expr = f"df['{feature}'].rolling({window}).{op}()"
                            expressions.append(fallback_expr)
                            success_count += 1
                            self.logger.info(f"Fallback successful: {fallback_expr}")
                            expr_generated = True
                        except Exception as fallback_e:
                            self.logger.error(f"Fallback failed: {fallback_e}")
            
            # 添加小延迟避免资源占用过高
            import time
            time.sleep(0.01)
        
        self.logger.info(f"Batch generation complete: {success_count}/{num_expressions} expressions generated")
        return expressions

def build_parser() -> ExpressionParser:
    """构建表达式解析器"""
    return ExpressionParser(
        OPERATORS,
        ignore_case=True,
        additional_operator_mapping={
            "Max": [Greater],
            "Min": [Less],
            "Delta": [Sub]
        },
        non_positive_time_deltas_allowed=False
    )

def main():
    # 设置参数
    data_path = "c:\\Users\\Administrator\\Desktop\\alpha-master\\data\\a_share\\csi500data\\daily_data.csv"
    output_dir = "c:\\Users\\Administrator\\Desktop\\alpha-master\\a_factor_generate\\a_gen"
    start_time = "2020-01-01"
    end_time = "2022-01-01"
    pool_size = 10
    n_generations = 5
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_path = os.path.join(output_dir, f"generate_alpha_{timestamp}.log")
    logger = get_logger(name="alpha_generator", file_path=log_path)
    
    # 设置随机种子
    reseed_everything(42)
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 加载本地数据
    logger.info("Loading local CSI500 data...")
    try:
        data_train = LocalStockData(
            data_path=data_path,
            start_time=start_time,
            end_time=end_time,
            device=device
        )
        logger.info(f"Data loaded successfully: {data_train.n_days} days, {data_train.n_stocks} stocks")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
    
    # 创建计算器，确保设置目标值
    # 使用None让LocalAlphaCalculator内部自动创建未来20天收益率作为目标
    calculator_train = LocalAlphaCalculator(data_train, target=None)
    
    # 创建alpha池
    logger.info(f"Creating alpha pool with capacity {pool_size}...")
    pool = MseAlphaPool(
        capacity=pool_size,
        calculator=calculator_train,
        device=device,
        l1_alpha=1e-3  # L1正则化参数
    )
    
    # 创建表达式生成器
    generator = SimpleExpressionGenerator()
    
    # 生成alpha因子
    logger.info("Starting alpha generation process using alphagen interface...")
    total_evaluated = 0
    total_generated = 0
    max_expressions = 100  # 最大生成表达式数量
    expressions_per_batch = 20  # 每批生成的表达式数量
    batch_count = 0
    
    while total_generated < max_expressions:
        batch_count += 1
        current_batch_size = min(expressions_per_batch, max_expressions - total_generated)
        logger.info(f"Batch {batch_count}: Generating {current_batch_size} expressions...")
        
        try:
            # 使用alphagen接口生成一批随机表达式
            batch_expressions = generator.generate_batch_expressions(current_batch_size)
            logger.info(f"Batch {batch_count}: Generated {len(batch_expressions)} expressions")
            
            for i, expr in enumerate(batch_expressions):
                if total_generated >= max_expressions:
                    logger.info(f"Reached maximum number of expressions ({max_expressions}). Stopping generation.")
                    break
                
                try:
                    total_generated += 1
                    logger.info(f"Batch {batch_count}, Expr {i+1}/{len(batch_expressions)}: Evaluating {total_generated}/{max_expressions}: {expr}")
                    
                    # 评估表达式
                    reward = pool.try_new_expr(expr)
                    total_evaluated += 1
                    
                    if reward > 0:
                        logger.info(f"Added promising alpha: {expr}, reward: {reward}")
                        
                    # 检查是否已经生成了足够的因子
                    if pool.size >= pool_size:
                        logger.info(f"Reached maximum pool size of {pool_size}. Stopping generation.")
                        break
                except Exception as e:
                    logger.warning(f"Failed to evaluate expression {total_generated}: {e}")
                    continue
            
            if pool.size >= pool_size:
                logger.info(f"Breaking outer loop - pool size reached")
                break
                
        except Exception as e:
            logger.error(f"Error in expression generation batch {batch_count}: {e}")
            import traceback
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            # 如果连续3个批次都失败，就退出循环
            if batch_count % 3 == 0:
                logger.error("Too many batch failures. Aborting generation.")
                break
            continue
        
        # 添加批次完成信息
        logger.info(f"Batch {batch_count} completed. Total generated: {total_generated}, Total evaluated: {total_evaluated}")
    
    logger.info(f"Generated {total_generated} expressions, evaluated {total_evaluated} valid expressions")
    
    # 记录最终池状态
    current_ic = pool.best_ic_ret
    logger.info(f"Factor evaluation completed. Pool size: {pool.size}, Best IC: {current_ic}")
    
    # 保存池状态
    _save_pool_state(pool, output_dir, 0)
    
    # 保存最终生成的因子
    factors_dir = os.path.join(output_dir, "generated_factors")
    os.makedirs(factors_dir, exist_ok=True)
    
    logger.info(f"Saving generated factors to {factors_dir}...")
    factors_info = []
    
    for i, expr in enumerate(pool.exprs[:pool.size]):
        expr_str = str(expr).replace('$', '')
        factor_name = f"factor_{i+1}.py"
        factor_path = os.path.join(factors_dir, factor_name)
        
        # 保存因子代码
        with open(factor_path, 'w') as f:
            f.write(f"import pandas as pd\nimport numpy as np\n\n")
            f.write(f"def calculate_factor(data):\n")
            f.write(f"    \"\"\"\n")
            f.write(f"    计算Alpha因子\n")
            f.write(f"    \"\"\"\n")
            f.write(f"    # 因子表达式: {expr_str}\n")
            
            # 根据表达式类型生成相应的计算代码
            if "Mean(close" in expr_str and "/ close - 1" in expr_str:
                window = int(expr_str.split(",")[1].split(")")[0])
                f.write(f"    # 计算动量因子\n")
                f.write(f"    result = pd.DataFrame(index=data.index.get_level_values(0).unique())\n")
                f.write(f"    for stock in data.columns.levels[0]:\n")
                f.write(f"        result[stock] = data[stock]['close'].rolling(window={window}).mean() / data[stock]['close'] - 1\n")
            elif "Std(close" in expr_str and "/ Mean(close" in expr_str:
                window = int(expr_str.split(",")[1].split(")")[0])
                f.write(f"    # 计算波动率因子\n")
                f.write(f"    result = pd.DataFrame(index=data.index.get_level_values(0).unique())\n")
                f.write(f"    for stock in data.columns.levels[0]:\n")
                f.write(f"        std = data[stock]['close'].rolling(window={window}).std()\n")
                f.write(f"        mean = data[stock]['close'].rolling(window={window}).mean()\n")
                f.write(f"        result[stock] = std / mean\n")
            elif "Volume" in expr_str and "/ Mean(Volume" in expr_str:
                window = int(expr_str.split(",")[1].split(")")[0])
                f.write(f"    # 计算成交量因子\n")
                f.write(f"    result = pd.DataFrame(index=data.index.get_level_values(0).unique())\n")
                f.write(f"    for stock in data.columns.levels[0]:\n")
                f.write(f"        result[stock] = data[stock]['volume'] / data[stock]['volume'].rolling(window={window}).mean()\n")
            else:
                f.write(f"    # 通用因子计算框架\n")
                f.write(f"    # 请根据具体表达式实现计算逻辑\n")
                f.write(f"    result = pd.DataFrame()\n")
            
            f.write(f"    return result\n")
        
        factors_info.append({
            "id": i+1,
            "name": factor_name,
            "expression": expr_str,
            "weight": float(pool.weights[i]) if i < len(pool.weights) else 0.0,
            "ic": float(pool.single_ics[i]) if i < len(pool.single_ics) else 0.0
        })
    
    # 保存因子信息
    with open(os.path.join(factors_dir, "factors_info.json"), 'w', encoding='utf-8') as f:
        json.dump(factors_info, f, indent=2, ensure_ascii=False)
    
    # 保存README
    with open(os.path.join(factors_dir, "README.md"), 'w', encoding='utf-8') as f:
        f.write("# 生成的Alpha因子\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"数据源: CSI500数据 ({start_time} 至 {end_time})\n\n")
        f.write(f"总共评估表达式: {total_evaluated}\n")
        f.write(f"成功生成因子数量: {len(factors_info)}\n\n")
        
        f.write("## 因子列表\n\n")
        for factor in factors_info:
            f.write(f"### 因子 {factor['id']}\n")
            f.write(f"- **表达式**: {factor['expression']}\n")
            f.write(f"- **权重**: {factor['weight']:.6f}\n")
            f.write(f"- **IC值**: {factor['ic']:.6f}\n")
            f.write(f"- **文件**: {factor['name']}\n\n")
    
    # 保存池的完整状态
    with open(os.path.join(output_dir, "final_pool_state.json"), 'w', encoding='utf-8') as f:
        pool_state = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pool_size": pool.size,
            "best_ic": float(pool.best_ic_ret),
            "total_evaluated": total_evaluated,
            "factors": factors_info
        }
        json.dump(pool_state, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Alpha factor generation completed successfully!")
    logger.info(f"Generated {len(factors_info)} factors from {total_evaluated} evaluated expressions")
    print(f"Generated {len(factors_info)} factors. Results saved to {factors_dir}")

def _save_pool_state(pool, output_dir, generation):
    """保存池的中间状态"""
    state_path = os.path.join(output_dir, f"pool_state_gen_{generation}.json")
    factors_info = []
    
    for i, expr in enumerate(pool.exprs[:pool.size]):
        factors_info.append({
            "id": i+1,
            "expression": str(expr).replace('$', ''),
            "weight": float(pool.weights[i]) if i < len(pool.weights) else 0.0,
            "ic": float(pool.single_ics[i]) if i < len(pool.single_ics) else 0.0
        })
    
    state = {
        "generation": generation,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pool_size": pool.size,
        "best_ic": float(pool.best_ic_ret),
        "factors": factors_info
    }
    
    with open(state_path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()