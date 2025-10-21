#!/usr/bin/env python3
"""
报告生成器模块
负责将Alpha因子分析数据转换为结构化报告
"""

import os
import json
import logging
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

# 导入LLM适配器
try:
    from llm_adapter import create_llm_adapter
except ImportError:
    import sys
    import os
    # 如果直接运行脚本，尝试添加当前目录到Python路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from llm_adapter import create_llm_adapter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('report_agent.report_generator')


class ReportGenerator:
    """
    Alpha因子分析报告生成器
    """
    
    def __init__(self, llm_config: Optional[Dict] = None):
        """
        初始化报告生成器
        
        Args:
            llm_config: LLM配置字典
        """
        self.llm_adapter = create_llm_adapter(llm_config)
        self.report_sections = []
        logger.info("✅ 报告生成器初始化成功")
    
    def load_input_data(self, input_file: str) -> Dict[str, Any]:
        """
        加载输入数据文件
        
        Args:
            input_file: 输入文件路径
            
        Returns:
            加载的数据字典
        """
        logger.info(f"📁 加载输入数据: {input_file}")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"✅ 输入数据加载成功，包含{data['metadata']['total_markets']}个市场，{data['metadata']['total_factors']}个因子")
            return data
        except Exception as e:
            logger.error(f"❌ 加载输入数据失败: {str(e)}")
            raise Exception(f"加载输入数据失败: {str(e)}")
    
    def generate_report(self, input_data: Dict[str, Any], output_format: str = 'markdown') -> bytes:
        """
        生成完整的分析报告
        
        Args:
            input_data: 输入数据字典
            output_format: 输出格式，支持'markdown'和'pdf'
            
        Returns:
            生成的报告内容（字节形式）
        """
        logger.info(f"🚀 开始生成{output_format.upper()}格式的Alpha因子分析报告")
        
        # 生成报告各个部分
        report_content = self._generate_markdown_report(input_data)
        
        if output_format.lower() == 'markdown':
            return report_content.encode('utf-8')
        elif output_format.lower() == 'pdf':
            return self._convert_to_pdf(report_content)
        else:
            raise ValueError(f"不支持的输出格式: {output_format}")
    
    def _generate_markdown_report(self, input_data: Dict[str, Any]) -> str:
        """
        生成Markdown格式的报告
        
        Args:
            input_data: 输入数据字典
            
        Returns:
            Markdown格式的报告内容
        """
        # 生成报告元数据
        metadata = input_data.get('metadata', {})
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 初始化报告内容
        report = [
            "# Alpha因子分析报告",
            "",
            f"**生成时间**: {timestamp}",
            f"**数据源**: {metadata.get('data_source', 'N/A')}",
            f"**分析市场**: {metadata.get('total_markets', 0)}个",
            f"**总因子数**: {metadata.get('total_factors', 0)}个",
            ""
        ]
        
        # 生成报告各个部分
        report.extend(self._generate_summary_section(input_data))
        report.extend(self._generate_factor_analysis_section(input_data))
        report.extend(self._generate_optimization_results_section(input_data))
        report.extend(self._generate_conclusion_section(input_data))
        report.extend(self._generate_disclaimer_section())
        
        # 合并报告内容
        return "\n".join(report)
    
    def _generate_summary_section(self, input_data: Dict[str, Any]) -> List[str]:
        """
        生成报告摘要部分
        """
        logger.info("📊 生成报告摘要部分")
        
        # 构建系统提示词
        system_prompt = """
        你是一名专业的量化分析师，擅长撰写Alpha因子分析报告。
        请基于提供的数据，生成一份简洁明了的报告摘要，避免使用过多的bullet point，增强逻辑性，降低AI生成的痕迹。
        重点突出分析的主要发现、关键因子和整体表现。
        """
        
        # 构建用户提示词
        user_prompt = """
        请为以下Alpha因子分析数据生成一份专业的报告摘要。
        摘要应包括：
        1. 分析概述：包括分析的市场数量、因子数量和主要分析方法
        2. 关键发现：最重要的2-3个发现
        3. 因子表现：整体因子表现的简要总结
        4. 投资建议：基于分析结果的初步投资建议
        """
        
        # 生成摘要内容
        summary_content = self.llm_adapter.generate_report_section(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            data=input_data
        )
        
        return [
            "## 📋 报告摘要",
            "",
            summary_content,
            ""
        ]
    
    def _generate_factor_analysis_section(self, input_data: Dict[str, Any]) -> List[str]:
        """
        生成因子分析部分
        """
        logger.info("🔍 生成因子分析部分")
        
        factor_sections = ["## 🔬 Alpha因子详细分析"]
        
        # 分析每个市场的因子
        for market_name, market_data in input_data.get('factor_analysis', {}).items():
            factor_sections.append("")
            factor_sections.append(f"### {self._format_market_name(market_name)}市场因子分析")
            factor_sections.append("")
            
            # 生成该市场的因子分析
            market_analysis = self._generate_market_factor_analysis(market_name, market_data)
            factor_sections.extend(market_analysis)
        
        return factor_sections
    
    def _format_market_name(self, market_name: str) -> str:
        """
        格式化市场名称
        """
        market_name_map = {
            'a_share': 'A股',
            'crypto': '加密货币',
            'us_stock': '美股'
        }
        return market_name_map.get(market_name, market_name)
    
    def _generate_market_factor_analysis(self, market_name: str, market_data: Dict) -> List[str]:
        """
        生成特定市场的因子分析
        """
        # 构建系统提示词
        system_prompt = """
        你是一名专业的量化分析师，精通Alpha因子分析。
        请基于提供的因子数据，撰写一份专业、深入的因子分析，重点解释因子原理、实现方式、优化过程和最终表现。
        避免使用过多的bullet point，增加叙述的连贯性和逻辑性，降低AI生成的痕迹。
        请使用表格展示关键统计数据，使报告更具专业性。
        """
        
        # 构建用户提示词
        user_prompt = f"""
        请对{self._format_market_name(market_name)}市场的Alpha因子进行详细分析。
        分析应包括以下方面：
        1. 因子概述：所选因子的类型和特点
        2. 因子原理：每个因子的理论基础和预期效果
        3. 因子统计：关键统计指标的解读（均值、标准差、极值等）
        4. 因子质量：因子质量评分的分析和解读
        5. 因子选择：为什么选择这些因子，它们的优势是什么
        6. 因子优化：优化过程中的关键决策和调整
        """
        
        # 生成市场因子分析内容
        analysis_content = self.llm_adapter.generate_report_section(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            data=market_data
        )
        
        return analysis_content.split('\n')
    
    def _generate_optimization_results_section(self, input_data: Dict[str, Any]) -> List[str]:
        """
        生成优化结果部分
        """
        logger.info("📈 生成组合优化结果部分")
        
        # 构建系统提示词
        system_prompt = """
        你是一名专业的量化投资组合经理，精通因子组合优化。
        请基于提供的优化结果数据，撰写一份专业、详细的投资组合优化分析。
        重点解释不同优化方法的结果对比、风险收益特征分析以及最终投资组合的构建逻辑。
        使用表格清晰展示各个优化方法的关键指标（如收益率、夏普比率、最大回撤等）。
        避免使用过多的bullet point，增加内容的连贯性和专业性。
        """
        
        # 构建用户提示词
        user_prompt = """
        请对Alpha因子组合的优化结果进行详细分析。
        分析应包括以下方面：
        1. 优化方法概述：使用了哪些优化方法及其原理
        2. 结果对比：不同优化方法的收益率、夏普比率、最大回撤等指标对比
        3. IC分析：信息系数的解读和意义
        4. 因子权重：最终选择的因子权重分配逻辑
        5. 组合特征：最优组合的风险收益特征
        6. 稳定性分析：不同市场环境下的表现稳定性
        """
        
        # 提取优化结果数据
        optimization_data = input_data.get('optimization_results', {})
        
        # 生成优化结果分析内容
        optimization_content = self.llm_adapter.generate_report_section(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            data=optimization_data
        )
        
        return [
            "## 📊 组合优化结果分析",
            "",
            optimization_content,
            ""
        ]
    
    def _generate_conclusion_section(self, input_data: Dict[str, Any]) -> List[str]:
        """
        生成结论部分
        """
        logger.info("📝 生成结论和建议部分")
        
        # 构建系统提示词
        system_prompt = """
        你是一名资深的量化投资专家，擅长总结分析结果并提供专业建议。
        请基于整个Alpha因子分析报告的内容，生成一份专业、有深度的结论和建议。
        避免使用过多的bullet point，增加内容的连贯性和逻辑性。
        建议应具体、可操作，并结合分析结果给出合理的投资策略建议。
        """
        
        # 构建用户提示词
        user_prompt = """
        请基于前面的分析，生成Alpha因子分析报告的结论和建议部分。
        内容应包括：
        1. 主要发现：总结分析的关键发现
        2. 因子评价：对生成的Alpha因子整体质量的评价
        3. 投资建议：基于因子分析的具体投资策略建议
        4. 风险提示：潜在风险因素的说明
        5. 未来优化方向：如何进一步改进因子和策略
        """
        
        # 生成结论内容
        conclusion_content = self.llm_adapter.generate_report_section(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            data=input_data
        )
        
        return [
            "## 🎯 结论与投资建议",
            "",
            conclusion_content,
            ""
        ]
    
    def _generate_disclaimer_section(self) -> List[str]:
        """
        生成免责声明部分
        """
        return [
            "## ⚠️ 风险提示与免责声明",
            "",
            "**投资风险提示**:",
            "- 本报告仅作为量化分析示例，不构成任何投资建议",
            "- 过往表现不代表未来收益，投资决策需谨慎",
            "- 量化模型存在固有局限性，市场环境变化可能导致模型失效",
            "- 实际投资前请咨询专业投资顾问",
            "",
            "**免责声明**:",
            "- 本报告基于历史数据和统计模型生成",
            "- 报告内容仅供参考，投资者需自行承担投资风险",
            "- 报告生成时间: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        ]
    
    def _convert_to_pdf(self, markdown_content: str) -> bytes:
        """
        将Markdown内容转换为PDF格式
        
        Args:
            markdown_content: Markdown格式的文本内容
            
        Returns:
            PDF文件的字节内容
        """
        logger.info("🔄 将Markdown转换为PDF格式")
        
        try:
            # 检查pypandoc是否可用
            import pypandoc
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                output_file = tmp_file.name
            
            # 使用pypandoc转换
            extra_args = ['--from=markdown-yaml_metadata_block']
            
            # 尝试使用不同的PDF引擎
            pdf_engines = ['wkhtmltopdf', 'weasyprint', None]
            last_error = None
            
            for engine in pdf_engines:
                try:
                    engine_args = extra_args.copy()
                    if engine:
                        engine_args.append(f'--pdf-engine={engine}')
                    
                    pypandoc.convert_text(
                        markdown_content,
                        'pdf',
                        format='markdown',
                        outputfile=output_file,
                        extra_args=engine_args
                    )
                    
                    # 读取生成的PDF文件
                    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                        with open(output_file, 'rb') as f:
                            pdf_content = f.read()
                        
                        # 清理临时文件
                        os.unlink(output_file)
                        
                        logger.info(f"✅ PDF生成成功，使用引擎: {engine or '默认'}")
                        return pdf_content
                    else:
                        raise Exception("PDF文件生成失败或为空")
                        
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"⚠️ PDF引擎 {engine or '默认'} 失败: {e}")
                    continue
            
            # 清理临时文件
            if os.path.exists(output_file):
                os.unlink(output_file)
            
            # 如果所有引擎都失败，抛出异常
            raise Exception(f"PDF生成失败，最后错误: {last_error}\n\n请确保安装了wkhtmltopdf或LaTeX")
            
        except ImportError:
            logger.error("❌ pypandoc模块不可用，请安装pypandoc")
            raise Exception("生成PDF失败，pypandoc模块不可用")
        except Exception as e:
            logger.error(f"❌ PDF生成失败: {str(e)}")
            raise Exception(f"生成PDF失败: {str(e)}")