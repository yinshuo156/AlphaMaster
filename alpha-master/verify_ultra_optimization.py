#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证超优化效果脚本
检查所有三个市场的25个因子是否都正确生成并深度优化
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple

def check_market_ultra_optimization(market: str) -> Dict:
    """检查单个市场的超优化效果"""
    print(f"\n{'='*60}")
    print(f"检查{market}市场超优化效果")
    print(f"{'='*60}")
    
    # 文件路径
    if market == "A股":
        factor_file = "alpha_pool/a_share_alpha_factors_ultra_optimized.csv"
        stats_file = "alpha_pool/a_share_alpha_factors_ultra_optimized_stats.csv"
        expected_factors = 25
    elif market == "Crypto":
        factor_file = "alpha_pool/crypto_alpha_factors_ultra_optimized.csv"
        stats_file = "alpha_pool/crypto_alpha_factors_ultra_optimized_stats.csv"
        expected_factors = 25
    elif market == "美股":
        factor_file = "alpha_pool/us_alpha_factors_ultra_optimized.csv"
        stats_file = "alpha_pool/us_alpha_factors_ultra_optimized_stats.csv"
        expected_factors = 25
    else:
        return {"success": False, "error": f"未知市场: {market}"}
    
    results = {
        "market": market,
        "success": True,
        "factor_count": 0,
        "expected_factors": expected_factors,
        "total_records": 0,
        "volatility_reversion_optimized": False,
        "factor_ranges_ok": True,
        "extreme_factors": [],
        "issues": []
    }
    
    try:
        # 检查因子文件是否存在
        if not os.path.exists(factor_file):
            results["success"] = False
            results["issues"].append(f"因子文件不存在: {factor_file}")
            return results
        
        if not os.path.exists(stats_file):
            results["success"] = False
            results["issues"].append(f"统计文件不存在: {stats_file}")
            return results
        
        # 读取因子数据
        print(f"读取因子数据: {factor_file}")
        factor_df = pd.read_csv(factor_file)
        results["total_records"] = len(factor_df)
        print(f"总记录数: {results['total_records']}")
        
        # 读取统计信息
        print(f"读取统计信息: {stats_file}")
        stats_df = pd.read_csv(stats_file)
        results["factor_count"] = len(stats_df)
        print(f"因子数量: {results['factor_count']}")
        
        # 检查因子数量
        if results["factor_count"] != expected_factors:
            results["success"] = False
            results["issues"].append(f"因子数量不匹配: 期望{expected_factors}个，实际{results['factor_count']}个")
        
        # 显示所有因子
        print(f"\n{market}市场因子列表:")
        print("-" * 40)
        for i, factor_name in enumerate(stats_df['factor_name'], 1):
            print(f"{i:2d}. {factor_name}")
        
        # 检查波动率回归因子是否优化
        volatility_factors = [f for f in stats_df['factor_name'] if 'VolatilityReversion' in f]
        if volatility_factors:
            vol_factor = volatility_factors[0]
            vol_stats = stats_df[stats_df['factor_name'] == vol_factor].iloc[0]
            vol_max = abs(vol_stats['max'])
            vol_min = abs(vol_stats['min'])
            
            print(f"\n波动率回归因子超优化检查:")
            print(f"因子名: {vol_factor}")
            print(f"最大值: {vol_stats['max']:.4f}")
            print(f"最小值: {vol_stats['min']:.4f}")
            print(f"标准差: {vol_stats['std']:.4f}")
            
            # 检查是否在合理范围内（tanh函数限制在[-1,1]）
            if vol_max <= 1.1 and vol_min <= 1.1:  # 允许小幅超出
                results["volatility_reversion_optimized"] = True
                print("✅ 波动率回归因子已超优化，数值在合理范围内")
            else:
                results["issues"].append(f"波动率回归因子未完全优化: 最大值{vol_max:.4f}超出范围")
                print("❌ 波动率回归因子未完全优化，数值超出合理范围")
        else:
            results["issues"].append("未找到波动率回归因子")
        
        # 检查所有因子的数值范围
        print(f"\n因子数值范围超优化检查:")
        print("-" * 40)
        extreme_factors = []
        for _, row in stats_df.iterrows():
            factor_name = row['factor_name']
            max_val = abs(row['max'])
            min_val = abs(row['min'])
            std_val = row['std']
            
            # 超优化标准：更严格的极值检查
            if max_val > 5 or min_val > 5 or std_val > 3:
                extreme_factors.append({
                    'name': factor_name,
                    'max': row['max'],
                    'min': row['min'],
                    'std': std_val
                })
                results["factor_ranges_ok"] = False
        
        results["extreme_factors"] = extreme_factors
        
        if extreme_factors:
            print("⚠️ 发现需要进一步优化的因子:")
            for factor in extreme_factors:
                print(f"  {factor['name']}: max={factor['max']:.4f}, min={factor['min']:.4f}, std={factor['std']:.4f}")
            results["issues"].extend([f"需要优化因子: {f['name']}" for f in extreme_factors])
        else:
            print("✅ 所有因子数值范围已超优化")
        
        # 显示统计摘要
        print(f"\n{market}市场超优化统计摘要:")
        print("-" * 40)
        print(f"因子数量: {results['factor_count']}/{expected_factors}")
        print(f"总记录数: {results['total_records']:,}")
        print(f"波动率回归因子优化: {'✅' if results['volatility_reversion_optimized'] else '❌'}")
        print(f"数值范围超优化: {'✅' if results['factor_ranges_ok'] else '⚠️'}")
        print(f"极端值因子数量: {len(extreme_factors)}")
        
        if results["issues"]:
            print(f"\n发现的问题:")
            for issue in results["issues"]:
                print(f"  ❌ {issue}")
        else:
            print(f"\n✅ {market}市场超优化完成，无问题")
        
    except Exception as e:
        results["success"] = False
        results["issues"].append(f"检查过程出错: {str(e)}")
        print(f"❌ 检查{market}市场时出错: {e}")
    
    return results

def main():
    """主函数"""
    print("=" * 80)
    print("超优化效果验证")
    print("检查所有三个市场的25个因子是否都正确生成并深度优化")
    print("=" * 80)
    
    markets = ["A股", "Crypto", "美股"]
    all_results = []
    
    for market in markets:
        result = check_market_ultra_optimization(market)
        all_results.append(result)
    
    # 总结报告
    print(f"\n{'='*80}")
    print("超优化验证总结报告")
    print(f"{'='*80}")
    
    total_success = 0
    total_factors = 0
    total_records = 0
    total_extreme_factors = 0
    
    for result in all_results:
        market = result["market"]
        success = result["success"]
        factor_count = result["factor_count"]
        records = result["total_records"]
        extreme_count = len(result["extreme_factors"])
        
        print(f"\n{market}市场:")
        print(f"  状态: {'✅ 成功' if success else '❌ 失败'}")
        print(f"  因子数: {factor_count}/25")
        print(f"  记录数: {records:,}")
        print(f"  极端值因子: {extreme_count}个")
        
        if success:
            total_success += 1
        total_factors += factor_count
        total_records += records
        total_extreme_factors += extreme_count
    
    print(f"\n总体结果:")
    print(f"  成功市场: {total_success}/3")
    print(f"  总因子数: {total_factors}/75 (期望75个)")
    print(f"  总记录数: {total_records:,}")
    print(f"  总极端值因子: {total_extreme_factors}个")
    
    # 检查是否有问题
    has_issues = any(result["issues"] for result in all_results)
    
    if total_success == 3 and total_factors == 75 and total_extreme_factors == 0:
        print(f"\n🎉 所有市场超优化完全成功！")
        print(f"   ✅ 三个市场全部成功")
        print(f"   ✅ 每个市场25个因子")
        print(f"   ✅ 问题因子已超优化")
        print(f"   ✅ 数值范围完全正常")
        print(f"   ✅ 无极端值因子")
    elif total_success == 3 and total_factors == 75:
        print(f"\n🎯 超优化基本成功！")
        print(f"   ✅ 三个市场全部成功")
        print(f"   ✅ 每个市场25个因子")
        print(f"   ✅ 大部分因子已优化")
        print(f"   ⚠️ 还有{total_extreme_factors}个因子需要进一步优化")
    else:
        print(f"\n⚠️  超优化未完全成功，需要进一步检查")
        for result in all_results:
            if not result["success"] or result["issues"]:
                print(f"   {result['market']}: {', '.join(result['issues'])}")

if __name__ == "__main__":
    main()

