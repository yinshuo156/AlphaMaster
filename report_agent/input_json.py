"""
创建Report Agent输入JSON文件

将优化因子和投资组合优化报告的重要数据合并成一个JSON文件，
作为report agent的输入数据。
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List


def load_factor_stats(file_path: str) -> Dict[str, Any]:
    """加载因子统计信息"""
    try:
        df = pd.read_csv(file_path)
        stats = {}
        for _, row in df.iterrows():
            factor_name = row['factor_name']
            stats[factor_name] = {
                'count': int(row['count']),
                'mean': float(row['mean']),
                'std': float(row['std']),
                'min': float(row['min']),
                'max': float(row['max']),
                'range': float(row['max']) - float(row['min']),
                'cv': float(row['std']) / abs(float(row['mean'])) if abs(float(row['mean'])) > 1e-10 else float('inf')
            }
        return stats
    except Exception as e:
        print(f"Error loading factor stats from {file_path}: {e}")
        return {}


def load_portfolio_summary(file_path: str) -> Dict[str, Any]:
    """加载投资组合优化摘要"""
    try:
        df = pd.read_csv(file_path)
        summary = {}
        for _, row in df.iterrows():
            market = row['market']
            method = row['method']
            if market not in summary:
                summary[market] = {}
            summary[market][method] = {
                'expected_return': float(row['expected_return']),
                'volatility': float(row['volatility']),
                'sharpe_ratio': float(row['sharpe_ratio']),
                'factor_count': int(row['factor_count'])
            }
        return summary
    except Exception as e:
        print(f"Error loading portfolio summary from {file_path}: {e}")
        return {}


def load_portfolio_weights(file_path: str) -> Dict[str, Any]:
    """加载投资组合权重"""
    try:
        df = pd.read_csv(file_path)
        weights = {}
        for _, row in df.iterrows():
            method = row['method']
            factor = row['factor']
            weight = float(row['weight'])
            
            if method not in weights:
                weights[method] = {}
            weights[method][factor] = {
                'weight': weight,
                'expected_return': float(row['expected_return']),
                'volatility': float(row['volatility']),
                'sharpe_ratio': float(row['sharpe_ratio'])
            }
        return weights
    except Exception as e:
        print(f"Error loading portfolio weights from {file_path}: {e}")
        return {}


def analyze_factor_quality(factor_stats: Dict[str, Any]) -> Dict[str, Any]:
    """分析因子质量"""
    quality_metrics = {}
    
    for factor_name, stats in factor_stats.items():
        # 计算质量指标
        stability_score = 1.0 / (1.0 + stats['cv']) if stats['cv'] != float('inf') else 0.0
        range_score = min(1.0, 10.0 / stats['range']) if stats['range'] > 0 else 0.0
        distribution_score = 1.0 - abs(stats['mean']) / stats['std'] if stats['std'] > 0 else 0.0
        
        quality_metrics[factor_name] = {
            'stability_score': stability_score,
            'range_score': range_score,
            'distribution_score': distribution_score,
            'overall_quality': (stability_score + range_score + distribution_score) / 3.0,
            'is_outlier_prone': stats['cv'] > 2.0,
            'is_extreme_range': stats['range'] > 10.0
        }
    
    return quality_metrics


def analyze_portfolio_performance(portfolio_summary: Dict[str, Any]) -> Dict[str, Any]:
    """分析投资组合表现"""
    performance_analysis = {}
    
    for market, methods in portfolio_summary.items():
        market_analysis = {
            'best_sharpe_method': None,
            'best_return_method': None,
            'lowest_risk_method': None,
            'method_comparison': {}
        }
        
        best_sharpe = -float('inf')
        best_return = -float('inf')
        lowest_risk = float('inf')
        
        for method, metrics in methods.items():
            # 记录最佳方法
            if metrics['sharpe_ratio'] > best_sharpe:
                best_sharpe = metrics['sharpe_ratio']
                market_analysis['best_sharpe_method'] = method
            
            if metrics['expected_return'] > best_return:
                best_return = metrics['expected_return']
                market_analysis['best_return_method'] = method
            
            if metrics['volatility'] < lowest_risk:
                lowest_risk = metrics['volatility']
                market_analysis['lowest_risk_method'] = method
            
            # 方法比较
            market_analysis['method_comparison'][method] = {
                'risk_return_ratio': metrics['expected_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0,
                'risk_adjusted_return': metrics['expected_return'] - 0.5 * metrics['volatility'],
                'volatility_rank': 0,  # 将在后续计算
                'return_rank': 0,      # 将在后续计算
                'sharpe_rank': 0       # 将在后续计算
            }
        
        # 计算排名
        volatility_values = [m['volatility'] for m in methods.values()]
        return_values = [m['expected_return'] for m in methods.values()]
        sharpe_values = [m['sharpe_ratio'] for m in methods.values()]
        
        for method, metrics in methods.items():
            market_analysis['method_comparison'][method]['volatility_rank'] = sorted(volatility_values).index(metrics['volatility']) + 1
            market_analysis['method_comparison'][method]['return_rank'] = sorted(return_values, reverse=True).index(metrics['expected_return']) + 1
            market_analysis['method_comparison'][method]['sharpe_rank'] = sorted(sharpe_values, reverse=True).index(metrics['sharpe_ratio']) + 1
        
        performance_analysis[market] = market_analysis
    
    return performance_analysis


def create_comprehensive_json():
    """创建综合的JSON输入文件"""
    
    # 数据文件路径
    base_path = Path("portfolios")
    
    # 加载所有数据
    print("Loading factor statistics...")
    a_share_stats = load_factor_stats(base_path / "optimized_factors" / "a_share_optimized_factors_stats.csv")
    crypto_stats = load_factor_stats(base_path / "optimized_factors" / "crypto_optimized_factors_stats.csv")
    us_stats = load_factor_stats(base_path / "optimized_factors" / "us_optimized_factors_stats.csv")
    
    print("Loading portfolio summary...")
    portfolio_summary = load_portfolio_summary(base_path / "optimization_reports" / "portfolio_optimization_summary.csv")
    
    print("Loading portfolio weights...")
    a_share_weights = load_portfolio_weights(base_path / "optimization_reports" / "a_share_portfolio_weights.csv")
    crypto_weights = load_portfolio_weights(base_path / "optimization_reports" / "crypto_portfolio_weights.csv")
    us_weights = load_portfolio_weights(base_path / "optimization_reports" / "us_portfolio_weights.csv")
    
    # 分析因子质量
    print("Analyzing factor quality...")
    a_share_quality = analyze_factor_quality(a_share_stats)
    crypto_quality = analyze_factor_quality(crypto_stats)
    us_quality = analyze_factor_quality(us_stats)
    
    # 分析投资组合表现
    print("Analyzing portfolio performance...")
    performance_analysis = analyze_portfolio_performance(portfolio_summary)
    
    # 创建综合JSON结构
    comprehensive_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "data_source": "portfolios/optimized_factors and portfolios/optimization_reports",
            "total_markets": 3,
            "total_factors": len(a_share_stats) + len(crypto_stats) + len(us_stats),
            "optimization_methods": ["max_sharpe", "min_volatility", "efficient_return", "hrp", "cla"]
        },
        
        "factor_analysis": {
            "a_share": {
                "factor_statistics": a_share_stats,
                "factor_quality": a_share_quality,
                "selected_factors": list(a_share_stats.keys()),
                "total_factors": len(a_share_stats),
                "high_quality_factors": [f for f, q in a_share_quality.items() if q['overall_quality'] > 0.7],
                "problematic_factors": [f for f, q in a_share_quality.items() if q['is_outlier_prone'] or q['is_extreme_range']]
            },
            "crypto": {
                "factor_statistics": crypto_stats,
                "factor_quality": crypto_quality,
                "selected_factors": list(crypto_stats.keys()),
                "total_factors": len(crypto_stats),
                "high_quality_factors": [f for f, q in crypto_quality.items() if q['overall_quality'] > 0.7],
                "problematic_factors": [f for f, q in crypto_quality.items() if q['is_outlier_prone'] or q['is_extreme_range']]
            },
            "us": {
                "factor_statistics": us_stats,
                "factor_quality": us_quality,
                "selected_factors": list(us_stats.keys()),
                "total_factors": len(us_stats),
                "high_quality_factors": [f for f, q in us_quality.items() if q['overall_quality'] > 0.7],
                "problematic_factors": [f for f, q in us_quality.items() if q['is_outlier_prone'] or q['is_extreme_range']]
            }
        },
        
        "portfolio_optimization": {
            "summary": portfolio_summary,
            "performance_analysis": performance_analysis,
            "weights": {
                "a_share": a_share_weights,
                "crypto": crypto_weights,
                "us": us_weights
            }
        },
        
        "optimization_methods": {
            "max_sharpe": {
                "description": "最大化夏普比率组合",
                "objective": "在给定风险水平下最大化预期收益",
                "risk_level": "中等"
            },
            "min_volatility": {
                "description": "最小波动率组合",
                "objective": "最小化投资组合波动率",
                "risk_level": "低"
            },
            "efficient_return": {
                "description": "高效收益组合",
                "objective": "在给定收益目标下最小化风险",
                "risk_level": "中等"
            },
            "hrp": {
                "description": "分层风险平价组合",
                "objective": "基于资产相关性进行风险分配",
                "risk_level": "低"
            },
            "cla": {
                "description": "临界线算法组合",
                "objective": "基于均值方差理论的最优组合",
                "risk_level": "中等"
            }
        },
        
        "factor_generation_methods": {
            "Alpha-GFN": {
                "description": "基于GFlowNet的深度强化学习因子挖掘",
                "factors": ["GFN_Momentum", "GFN_LogPrice", "GFN_Volatility", "GFN_Composite"],
                "methodology": "使用深度强化学习探索因子空间"
            },
            "AlphaAgent": {
                "description": "基于大语言模型的多智能体系统",
                "factors": ["Agent_MACD", "Agent_MomentumReversal"],
                "methodology": "LLM驱动的多智能体协作因子生成"
            },
            "AlphaGen": {
                "description": "基于PPO的因子生成",
                "factors": ["Gen_VolumeAnomaly", "Gen_PriceAcceleration"],
                "methodology": "近端策略优化算法生成因子"
            },
            "AlphaMiner": {
                "description": "传统技术指标挖掘",
                "factors": ["Miner_SMACross"],
                "methodology": "基于传统技术分析指标"
            },
            "Genetic-Alpha": {
                "description": "遗传编程符号回归",
                "factors": ["Genetic_Composite", "Genetic_Multivariate"],
                "methodology": "遗传算法优化因子表达式"
            }
        },
        
        "key_insights": {
            "best_performing_market": max(portfolio_summary.keys(), 
                                        key=lambda m: max([metrics['sharpe_ratio'] for metrics in portfolio_summary[m].values()])),
            "most_stable_factors": [],
            "highest_sharpe_method": "hrp",
            "risk_return_tradeoff": "HRP方法在所有市场都表现出最高的夏普比率",
            "factor_diversity": "五个不同的因子生成方法提供了良好的因子多样性"
        }
    }
    
    # 找出最稳定的因子
    all_factors = {}
    for market in ["a_share", "crypto", "us"]:
        for factor, quality in comprehensive_data["factor_analysis"][market]["factor_quality"].items():
            all_factors[f"{market}_{factor}"] = quality["overall_quality"]
    
    comprehensive_data["key_insights"]["most_stable_factors"] = sorted(
        all_factors.items(), key=lambda x: x[1], reverse=True
    )[:5]
    
    return comprehensive_data


def main():
    """主函数"""
    print("Creating comprehensive JSON input for Report Agent...")
    
    try:
        # 创建综合数据
        comprehensive_data = create_comprehensive_json()
        
        # 保存JSON文件
        output_path = Path("report_agent") / "input_data.json"
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Comprehensive JSON file created: {output_path}")
        print(f"📊 Total data points: {comprehensive_data['metadata']['total_factors']} factors across {comprehensive_data['metadata']['total_markets']} markets")
        print(f"🎯 Optimization methods: {', '.join(comprehensive_data['metadata']['optimization_methods'])}")
        
        # 显示关键统计信息
        print("\n📈 Key Statistics:")
        for market in ["a_share", "crypto", "us"]:
            market_data = comprehensive_data["factor_analysis"][market]
            print(f"  {market.upper()}: {market_data['total_factors']} factors, {len(market_data['high_quality_factors'])} high-quality")
        
        print(f"\n🏆 Best performing market: {comprehensive_data['key_insights']['best_performing_market'].upper()}")
        print(f"📊 Most stable factors: {[f[0] for f in comprehensive_data['key_insights']['most_stable_factors'][:3]]}")
        
    except Exception as e:
        print(f"❌ Error creating JSON file: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()