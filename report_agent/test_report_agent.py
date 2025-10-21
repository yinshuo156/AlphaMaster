#!/usr/bin/env python3
"""
Report Agent 测试脚本
用于验证Report Agent的基本功能
"""

import os
import sys
import json
import logging
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入Report Agent模块
from report_generator import ReportGenerator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('report_agent.test')

def create_test_data() -> dict:
    """
    创建测试数据
    
    Returns:
        测试数据字典
    """
    return {
        "metadata": {
            "generated_at": "2025-09-30T23:43:13.863061",
            "data_source": "test_data_source",
            "total_markets": 1,
            "total_factors": 3,
            "optimization_methods": ["max_sharpe", "min_volatility"]
        },
        "factor_analysis": {
            "a_share": {
                "factor_statistics": {
                    "MomentumFactor": {
                        "count": 1000,
                        "mean": 0.0,
                        "std": 1.0,
                        "min": -3.0,
                        "max": 3.0,
                        "range": 6.0,
                        "cv": 1.0
                    },
                    "ValueFactor": {
                        "count": 1000,
                        "mean": 0.0,
                        "std": 0.8,
                        "min": -2.5,
                        "max": 2.5,
                        "range": 5.0,
                        "cv": 0.8
                    },
                    "SizeFactor": {
                        "count": 1000,
                        "mean": 0.0,
                        "std": 0.9,
                        "min": -2.8,
                        "max": 2.8,
                        "range": 5.6,
                        "cv": 0.9
                    }
                },
                "factor_quality": {
                    "MomentumFactor": {
                        "stability_score": 0.8,
                        "range_score": 0.9,
                        "distribution_score": 0.7,
                        "overall_quality": 0.8,
                        "is_outlier_prone": False,
                        "is_extreme_range": False
                    },
                    "ValueFactor": {
                        "stability_score": 0.9,
                        "range_score": 0.8,
                        "distribution_score": 0.8,
                        "overall_quality": 0.83,
                        "is_outlier_prone": False,
                        "is_extreme_range": False
                    },
                    "SizeFactor": {
                        "stability_score": 0.7,
                        "range_score": 0.85,
                        "distribution_score": 0.75,
                        "overall_quality": 0.77,
                        "is_outlier_prone": False,
                        "is_extreme_range": False
                    }
                },
                "selected_factors": ["MomentumFactor", "ValueFactor", "SizeFactor"],
                "problematic_factors": []
            }
        },
        "optimization_results": {
            "max_sharpe": {
                "return": 0.12,
                "sharpe_ratio": 1.5,
                "max_drawdown": -0.08,
                "ic": 0.045,
                "factor_weights": {
                    "MomentumFactor": 0.4,
                    "ValueFactor": 0.4,
                    "SizeFactor": 0.2
                }
            },
            "min_volatility": {
                "return": 0.08,
                "sharpe_ratio": 1.6,
                "max_drawdown": -0.05,
                "ic": 0.038,
                "factor_weights": {
                    "MomentumFactor": 0.2,
                    "ValueFactor": 0.5,
                    "SizeFactor": 0.3
                }
            }
        }
    }

def save_test_data(output_file: str):
    """
    保存测试数据到文件
    
    Args:
        output_file: 输出文件路径
    """
    test_data = create_test_data()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ 测试数据已保存至: {output_file}")

def test_report_agent():
    """
    测试Report Agent的基本功能
    """
    try:
        logger.info("🚀 开始测试Report Agent")
        
        # 创建测试数据
        test_data_file = "test_input_data.json"
        save_test_data(test_data_file)
        
        # 初始化报告生成器（使用默认配置）
        llm_config = {
            "model_name": "gpt-4",  # 可以根据需要修改
            "temperature": 0.1
        }
        
        logger.info("📋 初始化报告生成器")
        report_generator = ReportGenerator(llm_config)
        
        # 加载测试数据
        logger.info(f"📥 加载测试数据: {test_data_file}")
        test_data = report_generator.load_input_data(test_data_file)
        
        # 生成Markdown报告
        logger.info("📝 生成Markdown报告")
        markdown_output = "test_report.md"
        report_content = report_generator.generate_report(test_data, "markdown")
        
        # 保存报告
        with open(markdown_output, 'wb') as f:
            f.write(report_content)
        
        logger.info(f"✅ Markdown报告已保存至: {markdown_output}")
        
        # 尝试生成PDF报告（可选）
        try:
            logger.info("📊 尝试生成PDF报告")
            pdf_output = "test_report.pdf"
            pdf_content = report_generator.generate_report(test_data, "pdf")
            
            with open(pdf_output, 'wb') as f:
                f.write(pdf_content)
            
            logger.info(f"✅ PDF报告已保存至: {pdf_output}")
        except Exception as e:
            logger.warning(f"⚠️ PDF生成失败: {str(e)}")
            logger.warning("请确保已安装PDF引擎（如wkhtmltopdf）")
        
        logger.info("🎉 测试完成！")
        logger.info(f"📋 生成的文件:")
        logger.info(f"  - {test_data_file}")
        logger.info(f"  - {markdown_output}")
        if os.path.exists("test_report.pdf"):
            logger.info(f"  - test_report.pdf")
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {str(e)}")
        raise


if __name__ == "__main__":
    test_report_agent()