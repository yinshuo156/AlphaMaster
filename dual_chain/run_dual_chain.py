# -*- coding: utf-8 -*-
"""
双链协同架构启动脚本
用于启动因子生成、评估、优化和更新因子池的自动化闭环流程
"""

import os
import sys
import argparse
import logging
import json
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dual_chain.dual_chain_manager import DualChainManager

# 确保logs目录存在
os.makedirs("logs", exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join("logs", f"dual_chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    ]
)

logger = logging.getLogger('dual_chain.runner')

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='运行双链协同因子优化流程')
    
    parser.add_argument('--data-path', 
                       type=str, 
                       default='data/a_share',
                       help='数据路径')
    
    parser.add_argument('--pool-dir', 
                       type=str, 
                       default='dual_chain/pools',
                       help='因子池目录')
    
    parser.add_argument('--llm-model', 
                       type=str, 
                       default='qwen-plus',
                       help='LLM模型名称')
    
    parser.add_argument('--llm-provider', 
                       type=str, 
                       default='dashscope',
                       help='LLM提供商，支持openai、dashscope(阿里百炼)、deepseek等')
    
    parser.add_argument('--iterations', 
                       type=int, 
                       default=5,
                       help='迭代次数')
    
    parser.add_argument('--factors-per-iteration', 
                       type=int, 
                       default=5,
                       help='每次迭代生成的因子数量')
    
    parser.add_argument('--config', 
                       type=str,
                       help='配置文件路径')
    
    # 新增参数支持运行标准化评估流程
    parser.add_argument('--alpha-master-dir', 
                       type=str, 
                       default='c:/Users/Administrator/Desktop/alpha-master',
                       help='Alpha Master项目根目录')
    
    parser.add_argument('--run-standardized-eval', 
                       action='store_true',
                       help='运行标准化评估流程（评估已有因子池）')
    
    parser.add_argument('--factors-file', 
                       type=str,
                       default='c:/Users/Administrator/Desktop/alpha-master/a_factor_generate/all_factors_expressions.json',
                       help='现有因子文件路径')
    
    parser.add_argument('--complementary-count', 
                       type=int, 
                       default=50,
                       help='生成的互补因子数量')
    
    parser.add_argument('--mock-mode', 
                       action='store_true', 
                       help='使用模拟模式（不调用真实API）')
    
    return parser.parse_args()

def load_config(config_path):
    """
    从配置文件加载参数
    """
    if not os.path.exists(config_path):
        logger.error(f"❌ 配置文件不存在: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"❌ 加载配置文件失败: {e}")
        return {}

def ensure_directories():
    """
    确保必要的目录存在
    """
    directories = [
        os.path.join("dual_chain", "logs"),
        os.path.join("dual_chain", "pools"),
        os.path.join("dual_chain", "pools", "effective"),
        os.path.join("dual_chain", "pools", "discarded"),
        os.path.join("dual_chain", "pools", "output")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logger.info("必要的目录已创建")

def print_header():
    """
    打印程序头部信息
    """
    header = """
    ============================================================================
                双链协同因子优化系统 v1.0
    ============================================================================
    功能: 实现"生成→评估→优化→再评估→更新因子池"的自动化闭环流程
    架构: 有效因子池(effective_pool) + 废弃因子池(discarded_pool) + LLM智能优化
    作者: 量化研究员团队
    日期: 2024
    ============================================================================
    """
    print(header)
    logger.info("双链协同因子优化系统启动")

def print_summary(report):
    """
    打印运行结果摘要
    """
    summary = f"""
    ============================================================================
                📊 双链协同流程运行结果 📊
    ============================================================================
    迭代次数: {report['total_iterations']}
    生成因子总数: {report['total_generated_factors']}
    有效因子总数: {report['total_effective_factors']}
    废弃因子总数: {report['total_discarded_factors']}
    
    📈 有效因子池状态:
    """
    
    # 处理空因子池情况
    if report['total_effective_factors'] > 0 and 'final_pool_statistics' in report:
        stats = report['final_pool_statistics']
        # 确保所有需要的键都存在
        if all(key in stats for key in ['effective_factors_count', 'avg_effective_ic', 'avg_effective_ic_ir', 'avg_effective_sharpe']):
            summary += f"      - 有效因子数量: {stats['effective_factors_count']}\n"
            summary += f"      - 平均IC: {stats['avg_effective_ic']:.4f}\n"
            summary += f"      - 平均IC-IR: {stats['avg_effective_ic_ir']:.4f}\n"
            summary += f"      - 平均Sharpe: {stats['avg_effective_sharpe']:.4f}\n"
        else:
            summary += "      - 有效因子池统计数据不完整\n"
    else:
        summary += "      - 有效因子池为空，无统计数据\n"
    
    summary += "\n    📉 废弃因子池状态:\n"
    
    # 处理废弃因子池
    if 'final_pool_statistics' in report:
        stats = report['final_pool_statistics']
        # 确保所有需要的键都存在
        if all(key in stats for key in ['discarded_factors_count', 'avg_discarded_ic', 'avg_discarded_sharpe']) and stats['discarded_factors_count'] > 0:
            summary += f"      - 废弃因子数量: {stats['discarded_factors_count']}\n"
            summary += f"      - 平均IC: {stats['avg_discarded_ic']:.4f}\n"
            summary += f"      - 平均Sharpe: {stats['avg_discarded_sharpe']:.4f}\n"
        else:
            summary += "      - 废弃因子池为空或统计数据不完整\n"
    else:
        summary += "      - 废弃因子池为空，无统计数据\n"
    
    summary += "\n    🚀 迭代详情:\n"
    
    # 添加迭代详情
    for i, iteration in enumerate(report['iteration_details']):
        summary += f"      迭代 {i+1}: 生成{iteration['generated_factors']}个因子, "
        summary += f"有效{iteration['effective_factors']}个, "
        summary += f"废弃{iteration['discarded_factors']}个\n"
    
    summary += "==========================================================================\n"
    
    print(summary)
    logger.info("双链协同流程运行完成，已生成摘要")

def main():
    """
    主函数
    """
    try:
        # 打印头部信息
        print_header()
        
        # 确保必要的目录存在
        ensure_directories()
        
        # 解析命令行参数
        args = parse_arguments()
        
        # 从配置文件加载参数
        config = {}
        if args.config:
            config = load_config(args.config)
        
        # 合并参数
        data_path = config.get('data-path', args.data_path)
        pool_dir = config.get('pool-dir', args.pool_dir)
        llm_model = config.get('llm-model', args.llm_model)
        llm_provider = config.get('llm-provider', args.llm_provider)
        iterations = config.get('iterations', args.iterations)
        factors_per_iteration = config.get('factors-per-iteration', args.factors_per_iteration)
        alpha_master_dir = config.get('alpha-master-dir', args.alpha_master_dir)
        run_standardized_eval = config.get('run-standardized-eval', args.run_standardized_eval)
        factors_file = config.get('factors-file', args.factors_file)
        complementary_count = config.get('complementary-count', args.complementary_count)
        # 从配置文件中读取mock-mode设置（同时支持连字符和下划线格式）
        mock_mode = config.get('mock-mode', config.get('mock_mode', args.mock_mode))
        
        # 从环境变量获取API密钥
        llm_api_key = None
        if llm_provider == 'dashscope':
            llm_api_key = os.environ.get('DASHSCOPE_API_KEY')
        elif llm_provider == 'openai':
            llm_api_key = os.environ.get('OPENAI_API_KEY')
        elif llm_provider == 'deepseek':
            llm_api_key = os.environ.get('DEEPSEEK_API_KEY')
        
        # 如果环境变量中没有API密钥，从配置文件中尝试获取
        if not llm_api_key:
            llm_api_key = config.get('llm_api_key')
        
        logger.info(f"🔧 配置参数: 数据路径={data_path}, 迭代次数={iterations}, 每次生成因子数={factors_per_iteration}")
        logger.info(f"🔧 LLM提供商: {llm_provider}, 模型: {llm_model}")
        logger.info(f"🔧 API密钥: {'已配置' if llm_api_key else '未配置'}")
        
        # 初始化双链管理器
        logger.info("🚀 初始化双链协同管理器...")
        manager = DualChainManager(
            data_path=data_path,
            pool_dir=pool_dir,
            llm_model=llm_model,
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            alpha_master_dir=alpha_master_dir,
            mock_mode=mock_mode
        )
        
        # 根据参数选择运行不同的流程
        if run_standardized_eval:
            logger.info("🔄 开始运行标准化评估流程...")
            logger.info(f"📊 评估文件: {factors_file}")
            
            # 运行标准化评估流程
            report = manager.run_standardized_evaluation_pipeline(
                factors_file=factors_file,
                complementary_count=complementary_count
            )
            
            # 打印评估结果摘要
            logger.info("📈 标准化评估结果摘要：")
            logger.info(f"✅ 有效因子数量: {len(report.get('effective_factors', []))}")
            logger.info(f"❌ 废弃因子数量: {len(report.get('discarded_factors', []))}")
            logger.info(f"✨ 补充因子数量: {len(report.get('complementary_factors', []))}")
            logger.info(f"📊 最终有效因子数量: {len(report.get('final_factors', []))}")
            
            # 如果有有效因子，打印前几个的性能
            if report.get('effective_factors'):
                # 按IC绝对值从大到小排序因子
                sorted_factors = sorted(
                    report['effective_factors'],
                    key=lambda x: abs(x.get('metrics', {}).get('ic', 0)),
                    reverse=True
                )
                
                logger.info("\n🏆 前5个最佳因子（按IC绝对值排序）：")
                for i, factor in enumerate(sorted_factors[:5]):
                    # 使用正确的键名'metrics'而不是'evaluation_metrics'
                    ic = factor.get('metrics', {}).get('ic', 0)
                    sharpe = factor.get('metrics', {}).get('sharpe', 0)
                    logger.info(f"  {i+1}. {factor.get('name')} - IC: {ic:.4f}, Sharpe: {sharpe:.4f}")
        else:
            # 运行完整流程
            logger.info("🔄 开始运行双链协同流程...")
            report = manager.run_pipeline(
                iterations=iterations,
                num_factors_per_iteration=factors_per_iteration
            )
            
            # 打印运行结果摘要
            print_summary(report)
        
        logger.info("🎉 流程运行完成")
        
    except KeyboardInterrupt:
        logger.info("⏹️  程序被用户中断")
        print("\n程序被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 程序运行出错: {e}", exc_info=True)
        print(f"\n程序运行出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()