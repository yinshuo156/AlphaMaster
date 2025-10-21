#!/usr/bin/env python3
"""
Report Agent 主入口脚本
用于从命令行生成Alpha因子分析报告
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional

# 导入报告生成器
try:
    from report_generator import ReportGenerator
except ImportError:
    # 如果直接运行脚本，尝试添加当前目录到Python路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from report_generator import ReportGenerator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('report_agent.main')


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description='Alpha因子分析报告生成工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        示例:
          python main.py --input input_data.json --output report.md --format markdown
          python main.py --input input_data.json --output report.pdf --format pdf
          python main.py --input input_data.json --output report.md --provider openai --model gpt-4 --temperature 0.1
          python main.py --input input_data.json --output report.md --provider dashscope --model qwen-plus
          python main.py --input input_data.json --output report.md --provider deepseek --model deepseek-chat
        """
    )
    
    # 必需参数
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入JSON数据文件路径'
    )
    
    # 可选参数
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='输出报告文件路径，默认与输入文件同名但扩展名不同'
    )
    
    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['markdown', 'pdf'],
        default='markdown',
        help='输出报告格式，支持markdown或pdf，默认markdown'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='配置文件路径，包含LLM设置等（配置模板：config_template.json）'
    )
    
    # LLM相关参数
    parser.add_argument(
        '--provider',
        type=str,
        choices=['openai', 'dashscope', 'deepseek'],
        help='LLM提供商，支持openai、dashscope(阿里百炼)、deepseek'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='LLM模型名称，如gpt-4'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='LLM API密钥'
    )
    
    parser.add_argument(
        '--base-url',
        type=str,
        help='LLM API基础URL'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        help='LLM温度参数，控制输出的随机性'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        help='LLM最大token数'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试日志'
    )
    
    return parser.parse_args()


def load_config(config_file: Optional[str]) -> dict:
    """
    加载配置文件
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        配置字典
    """
    config = {}
    
    if config_file and os.path.exists(config_file):
        logger.info(f"📁 加载配置文件: {config_file}")
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info("✅ 配置文件加载成功")
        except Exception as e:
            logger.error(f"❌ 加载配置文件失败: {str(e)}")
            raise Exception(f"加载配置文件失败: {str(e)}")
    
    return config


def build_llm_config(config: dict, args: argparse.Namespace) -> dict:
    """
    构建LLM配置字典
    
    Args:
        config: 从配置文件加载的配置
        args: 命令行参数
        
    Returns:
        LLM配置字典
    """
    llm_config = config.get('llm', {})
    
    # 命令行参数优先级高于配置文件
    if args.provider:
        llm_config['provider'] = args.provider
    if args.model:
        llm_config['model_name'] = args.model
    if args.api_key:
        llm_config['api_key'] = args.api_key
    if args.base_url:
        llm_config['base_url'] = args.base_url
    if args.temperature is not None:
        llm_config['temperature'] = args.temperature
    if args.max_tokens is not None:
        llm_config['max_tokens'] = args.max_tokens
    
    # 如果没有指定provider，默认为openai
    if 'provider' not in llm_config:
        llm_config['provider'] = 'openai'
    
    return llm_config


def determine_output_path(input_path: str, output_path: Optional[str], format_type: str) -> str:
    """
    确定输出文件路径
    
    Args:
        input_path: 输入文件路径
        output_path: 指定的输出文件路径
        format_type: 输出格式
        
    Returns:
        输出文件路径
    """
    if output_path:
        return output_path
    
    # 如果没有指定输出路径，使用输入文件名但更改扩展名
    input_name = os.path.splitext(os.path.basename(input_path))[0]
    extension = 'md' if format_type == 'markdown' else 'pdf'
    
    # 输出文件放在输入文件的同一目录
    input_dir = os.path.dirname(input_path) or '.'
    
    return os.path.join(input_dir, f"{input_name}_report.{extension}")


def save_report(content: bytes, output_path: str):
    """
    保存报告文件
    
    Args:
        content: 报告内容
        output_path: 输出文件路径
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存文件
    with open(output_path, 'wb') as f:
        f.write(content)
    
    logger.info(f"✅ 报告已保存至: {os.path.abspath(output_path)}")

def main():
    """
    主函数
    """
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 如果启用调试模式，设置日志级别为DEBUG
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # 加载配置
        config = load_config(args.config)
        
        # 构建LLM配置
        llm_config = build_llm_config(config, args)
        
        # 确定输出路径
        output_path = determine_output_path(args.input, args.output, args.format)
        
        logger.info(f"🚀 开始生成{args.format.upper()}格式的Alpha因子分析报告")
        logger.info(f"🤖 使用LLM提供商: {llm_config.get('provider', 'openai')}, 模型: {llm_config.get('model_name', 'gpt-4')}")
        logger.info(f"📥 输入文件: {os.path.abspath(args.input)}")
        logger.info(f"📤 输出文件: {os.path.abspath(output_path)}")
        
        # 初始化报告生成器
        report_generator = ReportGenerator(llm_config)
        
        # 加载输入数据
        input_data = report_generator.load_input_data(args.input)
        
        # 生成报告
        report_content = report_generator.generate_report(input_data, args.format)
        
        # 保存报告
        save_report(report_content, output_path)
        
        logger.info("🎉 报告生成完成!")
        
    except KeyboardInterrupt:
        logger.info("⚠️ 操作被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 发生错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()