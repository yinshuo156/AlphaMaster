# 因子1: 动量因子
def factor_momentum(data, window=20):
    """
    动量因子：衡量股票过去一段时间的价格趋势强度，基于A股市场存在显著的中期动量效应（通常1-3个月）。
    原理是投资者对信息反应不足导致价格趋势延续。适用于CSI500成分股中基本面改善但尚未充分定价的标的。

    参数:
        data: pandas.DataFrame, 包含'open', 'high', 'low', 'close', 'volume'等列
        window: int, 动量计算周期，默认20日（约一个月）

    返回:
        pandas.Series: 每只股票在当前时点的动量因子值（收盘价相对window天前的收益率）
    """
    import numpy as np
    return data['close'].pct_change(window).fillna(0)


# 因子2: 波动率调整后的动量（风险调整动量）
def factor_risk_adj_momentum(data, window=60):
    """
    风险调整动量因子：将价格动量除以波动率，识别“高收益低波动”的优质趋势股。
    在A股市场中，单纯动量易受高波动小盘股影响，该因子通过夏普比率逻辑筛选稳健上涨个股，
    降低极端波动带来的回撤风险，适合CSI500中性策略。

    参数:
        data: pandas.DataFrame, 包含价格和成交量数据
        window: int, 计算窗口，默认60日（季度频率）

    返回:
        pandas.Series: 动量与波动率之比，标准化为无量纲指标
    """
    import numpy as np
    # 计算对数收益率
    log_ret = np.log(data['close'] / data['close'].shift(1))
    # 年化动量（60日累计对数收益）
    momentum = log_ret.rolling(window).sum()
    # 年化波动率（60日标准差 * sqrt(252)）
    volatility = log_ret.rolling(window).std() * np.sqrt(252)
    # 避免除零，对极小波动率设下限
    volatility = volatility.replace(0, np.nan).fillna(volatility.mean())
    # 风险调整后动量
    ra_momentum = momentum / volatility
    return ra_momentum.fillna(0)


# 因子3: 成交量加权价格偏离因子（资金流向异常检测）
def factor_volume_price_deviation(data, window=10):
    """
    成交量加权价格偏离因子：捕捉资金主动买入推动下的价格突破行为。
    计算每日(VWAP - MA(Price)) * Volume，反映成交量配合下的价格强势程度。
    A股市场散户主导，放量上涨更具持续性；此因子识别机构资金介入迹象，尤其适用于CSI500中小市值个股。

    参数:
        data: pandas.DataFrame, 包含OHLCV字段
        window: int, 移动平均窗口，默认10日

    返回:
        pandas.Series: 过去window日内该指标的均值，代表近期资金推升强度
    """
    import numpy as np
    # 简单VWAP估计（假设日内均匀交易）
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    vwap = (typical_price * data['volume']).rolling(window).sum() / data['volume'].rolling(window).sum()
    # 价格移动平均
    price_ma = data['close'].rolling(window).mean()
    # 偏离度乘以成交量进行放大
    deviation = (vwap - price_ma) * data['volume']
    # 取过去window日均值作为因子值
    factor = deviation.rolling(window).mean()
    return factor.fillna(0)


# 因子4: 价格区间突破因子（高低点动量增强）
def factor_high_low_breakout(data, window=20):
    """
    区间突破因子：衡量当前价格相对于过去N日最高价和最低价的位置。
    基于行为金融学中的“锚定效应”，当股价突破前期高点时引发追涨情绪。
    A股T+1机制下，突破信号稳定性强，且CSI500成分股流动性适中，适合跟踪技术突破机会。

    参数:
        data: pandas.DataFrame
        window: int, 观察窗口，默认20日

    返回:
        pandas.Series: (close - low)/(high - low) 的滚动Z-score，消除不同股票价格区间差异
    """
    import numpy as np
    # 构造价格位置：(当前价 - 最低价)/(最高价 - 最低价)
    rolling_high = data['high'].rolling(window).max()
    rolling_low = data['low'].rolling(window).min()
    denominator = rolling_high - rolling_low
    denominator = denominator.replace(0, np.nan).fillna(1e-6)  # 防止除零
    price_position = (data['close'] - rolling_low) / denominator

    # Z-score标准化以跨股票比较
    mean_pp = price_position.rolling(window).mean()
    std_pp = price_position.rolling(window).std()
    z_score = (price_position - mean_pp) / std_pp.where(std_pp != 0, 1e-6)

    return z_score.fillna(0)


# 因子5: 换手率变异系数因子（流动性结构变化）
def factor_turnover_volatility(data, window=30):
    """
    换手率变异系数因子：衡量个股流动性的稳定性。变异系数越低，说明交易活跃且稳定；
    突然变高则可能预示筹码松动或主力出货。A股市场中，CSI500成分股常出现阶段性流动性突变，
    此因子可辅助判断持仓稳定性，规避“伪强势”股。

    参数:
        data: pandas.DataFrame, 必须包含'volume'列
               假设总股本不变，换手率正比于volume（可用实际流通股本进一步优化）

    返回:
        pandas.Series: 过去window日成交量变异系数（标准差/均值），取负号使低变异对应高分
    """
    import numpy as np
    vol_series = data['volume']
    rolling_mean = vol_series.rolling(window).mean()
    rolling_std = vol_series.rolling(window).std()
    # 防止除零
    turnover_cv = rolling_std / rolling_mean.where(rolling_mean != 0, 1e-6)
    # 取负值：流动性稳定的股票得分更高
    return (-turnover_cv).fillna(0)