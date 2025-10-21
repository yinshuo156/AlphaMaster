# -*- coding: utf-8 -*-
"""
LLM因子适配器
使用大语言模型生成和优化因子表达式
"""

import os
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
import random
import re

from report_agent.llm_adapter import BaseLLMAdapter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dual_chain.llm_factor_adapter')

class LLMToolError(Exception):
    """LLM工具错误"""
    pass

class LLMFactorAdapter(BaseLLMAdapter):
    """
    LLM因子适配器
    使用大语言模型生成和优化因子
    """
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.3, mock_mode: bool = False, provider: str = None, api_key: str = None):
        """
        初始化LLM因子适配器
        
        Args:
            model_name: 模型名称
            temperature: 温度参数
            mock_mode: 是否启用模拟模式（用于测试，不调用真实API）
            provider: LLM提供商，支持'openai'、'dashscope'(阿里百炼)、'deepseek'等
            api_key: API密钥，如果不提供则从环境变量获取
        """
        self.mock_mode = mock_mode
        self.provider = provider
        self.api_key = api_key
        
        if not mock_mode:
            try:
                # 根据模型名称自动识别提供商
                if provider is None:
                    if 'qwen' in model_name.lower() or 'dashscope' in model_name.lower():
                        self.provider = 'dashscope'
                    elif 'deepseek' in model_name.lower():
                        self.provider = 'deepseek'
                    else:
                        self.provider = 'openai'
                
                logger.info(f"📡 正在初始化{self.provider}模型: {model_name}")
                
                # 根据提供商选择对应的适配器
                if self.provider == 'dashscope':
                    from report_agent.llm_adapter import DashScopeAdapter
                    self.llm_adapter = DashScopeAdapter(
                        model_name=model_name,
                        api_key=api_key,
                        temperature=temperature,
                        max_tokens=2000
                    )
                elif self.provider == 'deepseek':
                    from report_agent.llm_adapter import DeepSeekAdapter
                    self.llm_adapter = DeepSeekAdapter(
                        model_name=model_name,
                        api_key=api_key,
                        temperature=temperature,
                        max_tokens=2000
                    )
                else:
                    # 默认使用OpenAI
                    from report_agent.llm_adapter import OpenAIAdapter
                    self.llm_adapter = OpenAIAdapter(
                        model_name=model_name,
                        api_key=api_key,
                        temperature=temperature,
                        max_tokens=2000
                    )
                    
            except Exception as e:
                logger.warning(f"⚠️  LLM初始化失败，切换到模拟模式: {e}")
                self.mock_mode = True
        
        if self.mock_mode:
            logger.info("ℹ️  LLMFactorAdapter: 已启用模拟模式，不会调用真实LLM API")
        
        self.prompt_templates = self._load_prompt_templates()
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """
        加载提示词模板
        """
        templates = {
            "factor_generation": """
目标
您是量化交易阿尔法因子生成领域的专家。您的任务是设计一个新的横截面阿尔法因子—— 这是一种数学表达式，能够基于股票近期的市场数据，在每个交易日为每只股票分配一个分数。该分数将用于当日对股票进行排名与筛选（即跨市场横截面操作）。
您将获得：
可用数据字段与运算符列表。
有效与无效因子的示例集合。
您的目标是生成满足以下要求的因子表达式：
与示例因子不同。
从有效因子中获取灵感。
潜在预测能力优于无效因子。

可用数据字段
您可使用以下数据字段构建表达式。请确保在表达式中直接使用这些字段名：
- close: 收盘价
- open: 开盘价
- high: 最高价
- low: 最低价
- volume: 成交量

可用运算符
您可使用pandas的标准操作符和函数，如pct_change、rolling、mean、std、diff等

参考因子
请参考以下因子：
有效因子：
{effective_factors}

无效因子：
{discarded_factors}

要求
从有效因子中获取灵感，避免无效因子中出现的模式（如低排序信息系数（RankIC）或低信息比（IR））。
因子应生成无量纲值，不受价格尺度或成交量单位的影响。
仅使用提供的数据字段与运算符。
无需使用所有数据字段，因子表达式中使用的数据字段总数不超过 3 个。
避免过度嵌套运算符或构建过于复杂的表达式，以降低过拟合风险。
表达式应简洁、易读且可解释。

输出格式
请返回以下格式：
表达式: your_factor_expression
解释: your_factor_explanation
            """,
            
            "factor_optimization": """
目标
您是量化交易阿尔法因子优化领域的专家。您的任务是优化一个现有的横截面阿尔法因子—— 这是一种数学表达式，能够基于股票近期的市场数据，在每个交易日为每只股票分配一个分数。该分数将用于当日对股票进行排名与筛选（即跨市场横截面操作）。
您将获得：
可用数据字段与运算符列表。
现有因子的表达式及其性能（如排序信息系数（RankIC）、排序信息比（RankIR）、换手率（Turnover）、多样性（Diversity））。
该因子的优化历史。
您的目标是优化因子表达式，提升其性能。

可用数据字段
您可使用以下数据字段构建表达式。请确保在表达式中直接使用这些字段名：
- close: 收盘价
- open: 开盘价
- high: 最高价
- low: 最低价
- volume: 成交量

可用运算符
您可使用pandas的标准操作符和函数，如pct_change、rolling、mean、std、diff等

现有因子信息
因子表达式: {factor_expression}
因子解释: {factor_explanation}
评估结果: {evaluation_results}

优化历史
优化历史：暂无

要求
优质因子应满足：
排序信息系数（RankIC）> 0.015
排序信息比（RankIR）> 0.2
换手率（Turnover）< 1.5
多样性（Diversity）> 0.2
您的目标是基于上述性能指标优化因子。通常，提升排序信息系数（RankIC）与排序信息比（RankIR）是首要目标，换手率与多样性为次要目标。
因子应生成无量纲值，不受价格尺度或成交量单位的影响。
仅使用提供的数据字段与运算符。
无需使用所有数据字段，因子表达式中使用的数据字段总数不超过 3 个。
避免过度嵌套运算符或构建过于复杂的表达式，以降低过拟合风险。
表达式应简洁、易读且可解释。
避免重复先前的尝试。
检查因子表达式，防止过拟合。

输出格式
请返回以下格式：
优化后表达式: your_optimized_expression
改进说明: your_improvement_explanation
            """,
            
            "complementary_factor_generation": """
目标
您是量化交易阿尔法因子生成领域的专家。您的任务是设计一个与现有因子互补的新横截面阿尔法因子—— 这是一种数学表达式，能够基于股票近期的市场数据，在每个交易日为每只股票分配一个分数。该分数将用于当日对股票进行排名与筛选（即跨市场横截面操作）。
您将获得：
可用数据字段与运算符列表。
现有有效因子的集合及其性能。
您的目标是生成一个逻辑上与现有因子互补的新因子，捕捉不同的市场特征。

可用数据字段
您可使用以下数据字段构建表达式。请确保在表达式中直接使用这些字段名：
- close: 收盘价
- open: 开盘价
- high: 最高价
- low: 最低价
- volume: 成交量

可用运算符
您可使用pandas的标准操作符和函数，如pct_change、rolling、mean、std、diff等

现有因子分析
现有有效因子：
{effective_factors}

现有因子表达式示例（避免重复）：
{existing_expressions}

要求
生成的因子应与现有因子在逻辑上互补，捕捉不同的市场特征：
- 如果现有因子主要是动量类因子，优先生成均值回归类因子
- 如果现有因子主要是趋势类因子，优先生成波动率类因子
- 如果现有因子主要是价格类因子，优先生成成交量类因子
因子应生成无量纲值，不受价格尺度或成交量单位的影响。
仅使用提供的数据字段与运算符。
无需使用所有数据字段，因子表达式中使用的数据字段总数不超过 3 个。
避免过度嵌套运算符或构建过于复杂的表达式，以降低过拟合风险。
表达式应简洁、易读且可解释。
避免重复现有因子的核心逻辑和模式。

输出格式
请返回以下格式：
表达式: your_factor_expression
解释: your_factor_explanation
            """
        }
        return templates
    
    def _init_llm(self):
        """
        初始化LLM实例
        """
        # 如果是模拟模式，直接返回None
        if hasattr(self, 'mock_mode') and self.mock_mode:
            return None
            
        try:
            # 尝试导入LangChain OpenAI集成
            from langchain_openai import ChatOpenAI
            
            # 获取API密钥
            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL")
            
            # 检查API密钥
            if not api_key:
                logger.warning("⚠️ OPENAI_API_KEY环境变量未设置，使用模拟模式")
                self.mock_mode = True
                return None
            
            # 创建LLM实例
            llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=api_key,
                base_url=base_url if base_url else None
            )
            
            return llm
        except ImportError:
            # 尝试使用其他LLM提供商
            try:
                # 可以添加其他模型如DeepSeek、阿里百炼等
                logger.warning(f"⚠️  OpenAI集成不可用，尝试使用备用模型")
                
                # 如果没有可用的LLM，使用模拟响应
                from langchain_core.language_models import FakeListChatModel
                
                # 预定义的模拟响应
                responses = [
                    "表达式: close.pct_change(10).rolling(window=5).mean() * volume.pct_change(5).rolling(window=10).mean()\n解释: 结合动量和成交量变化的复合因子，捕捉价格和成交量的协同效应。",
                    "表达式: (close / close.rolling(window=50).mean() - 1) * (close - low) / (high - low + 1e-8)\n解释: 结合相对强弱和日内波动特性的因子，捕捉超买超卖状态。",
                    "表达式: close.pct_change().rolling(window=20).std() / close.pct_change().rolling(window=60).std()\n解释: 短期波动率与长期波动率的比率，捕捉市场情绪变化。"
                ]
                
                return FakeListChatModel(responses=responses)
            except Exception as e:
                logger.error(f"❌ 初始化LLM失败: {e}")
                # 失败时启用模拟模式
                self.mock_mode = True
                logger.warning("⚠️ 切换到模拟模式")
                return None
    
    def generate_factor_expression(self, 
                                effective_factors: List[Dict[str, Any]] = None,
                                discarded_factors: List[Dict[str, Any]] = None) -> Tuple[str, str]:
        """
        使用LLM生成新的因子表达式
        
        Args:
            effective_factors: 有效因子列表作为正向参考
            discarded_factors: 废弃因子列表作为负向参考
            
        Returns:
            (因子表达式, 因子解释)
        """
        if effective_factors is None:
            effective_factors = []
        if discarded_factors is None:
            discarded_factors = []
        
        # 格式化参考因子
        effective_factors_str = "\n".join([
            f"- {f.get('name', 'Unknown')}: {f.get('expression', 'Unknown')} (IC: {f.get('evaluation_metrics', {}).get('ic', 0):.4f})" 
            for f in effective_factors
        ])
        
        discarded_factors_str = "\n".join([
            f"- {f.get('name', 'Unknown')}: {f.get('expression', 'Unknown')} (IC: {f.get('evaluation_metrics', {}).get('ic', 0):.4f}, 原因: {f.get('reason', 'Unknown')})" 
            for f in discarded_factors
        ])
        
        # 如果没有参考因子，使用默认文本
        if not effective_factors_str:
            effective_factors_str = "暂无有效参考因子"
        if not discarded_factors_str:
            discarded_factors_str = "暂无废弃参考因子"
        
        # 构建提示词
        prompt = self.prompt_templates["factor_generation"].format(
            effective_factors=effective_factors_str,
            discarded_factors=discarded_factors_str
        )
        
        logger.info(f"📝 调用LLM生成因子表达式")
        
        # 调用LLM
        try:
            # 检查是否在模拟模式
            if hasattr(self, 'mock_mode') and self.mock_mode:
                # 模拟响应
                mock_responses = [
                    "表达式: close.pct_change(20).rolling(window=10).mean()\n解释: 基于20日价格变化的移动平均因子，捕捉中短期价格动量。",
                    "表达式: (close - low) / (high - low + 1e-8) * volume.pct_change(5)\n解释: 结合价格位置和成交量变化的因子，捕捉买卖力量变化。",
                    "表达式: close.pct_change().rolling(window=20).std()\n解释: 20日价格波动率因子，捕捉市场不确定性。"
                ]
                response = random.choice(mock_responses)
                logger.info(f"ℹ️  使用模拟响应: {response}")
            else:
                # 调用对应的LLM适配器生成报告
                response = self.llm_adapter.generate_report_section(
                    system_prompt="你是一名专业的量化因子设计专家。",
                    user_prompt=prompt
                )
            
            # 解析响应
            expression_match = re.search(r'表达式:\s*(.*)', response, re.MULTILINE)
            explanation_match = re.search(r'解释:\s*(.*)', response, re.MULTILINE)
            
            if expression_match and explanation_match:
                expression = expression_match.group(1).strip()
                explanation = explanation_match.group(1).strip()
                
                logger.info(f"✅ 因子生成成功")
                logger.info(f"📊 表达式: {expression}")
                logger.info(f"📊 解释: {explanation}")
                
                return expression, explanation
            else:
                logger.error(f"❌ 无法解析LLM响应: {response}")
                raise ValueError("无法从LLM响应中提取因子表达式和解释")
        except Exception as e:
            logger.error(f"❌ 生成因子表达式失败: {e}")
            # 返回备用表达式
            fallback_expression = f"close.pct_change({random.randint(3, 20)}).rolling(window={random.randint(3, 10)}).mean()"
            fallback_explanation = "备用动量因子，使用随机参数"
            logger.warning(f"⚠️ 使用备用因子: {fallback_expression}")
            return fallback_expression, fallback_explanation
                
    def generate_complementary_factor(self, 
                                    effective_factors: List[Dict[str, Any]] = None,
                                    existing_expressions: List[str] = None) -> Tuple[str, str]:
        """
        使用LLM生成与现有因子互补的新因子表达式
        
        Args:
            effective_factors: 有效因子列表，用于分析现有因子特点
            existing_expressions: 现有因子表达式列表，用于避免重复
            
        Returns:
            (因子表达式, 因子解释)
        """
        if effective_factors is None:
            effective_factors = []
        if existing_expressions is None:
            existing_expressions = []
        
        # 格式化现有有效因子
        effective_factors_str = "\n".join([
            f"- {f.get('name', 'Unknown')}: {f.get('expression', 'Unknown')} (IC: {f.get('evaluation_metrics', {}).get('ic', 0):.4f}, Sharpe: {f.get('evaluation_metrics', {}).get('sharpe', 0):.4f})" 
            for f in effective_factors[:5]  # 只显示前5个因子
        ])
        
        # 格式化现有表达式
        existing_expressions_str = "\n".join([
            f"- {expr[:100]}..." if len(expr) > 100 else f"- {expr}" 
            for expr in existing_expressions[:5]  # 只显示前5个表达式
        ])
        
        # 如果没有因子或表达式，使用默认文本
        if not effective_factors_str:
            effective_factors_str = "暂无有效参考因子"
        if not existing_expressions_str:
            existing_expressions_str = "暂无现有表达式"
        
        # 构建提示词
        prompt = self.prompt_templates["complementary_factor_generation"].format(
            effective_factors=effective_factors_str,
            existing_expressions=existing_expressions_str
        )
        
        logger.info(f"📝 调用LLM生成互补因子表达式")
        
        # 调用LLM
        try:
            # 检查是否在模拟模式
            if hasattr(self, 'mock_mode') and self.mock_mode:
                # 模拟响应 - 提供互补性强的因子
                mock_responses = [
                    "表达式: (close - close.rolling(window=20).mean()) / close.rolling(window=20).std()\n解释: 基于Z-score的均值回归因子，与动量因子形成互补。",
                    "表达式: volume.pct_change(5).rolling(window=10).std()\n解释: 成交量波动率因子，捕捉流动性变化，与价格类因子形成互补。",
                    "表达式: (high - close) / (high - low + 1e-8)\n解释: 日内反转因子，捕捉短期价格反转特征，与趋势跟踪因子形成互补。"
                ]
                response = random.choice(mock_responses)
                logger.info(f"ℹ️  使用模拟响应: {response}")
            else:
                # 调用对应的LLM适配器生成报告
                response = self.llm_adapter.generate_report_section(
                    system_prompt="你是一名专业的量化因子设计专家，特别擅长设计与现有因子互补的新因子。",
                    user_prompt=prompt
                )
            
            # 解析响应
            expression_match = re.search(r'表达式:\s*(.*)', response, re.MULTILINE)
            explanation_match = re.search(r'解释:\s*(.*)', response, re.MULTILINE)
            
            if expression_match and explanation_match:
                expression = expression_match.group(1).strip()
                explanation = explanation_match.group(1).strip()
                
                logger.info(f"✅ 互补因子生成成功")
                logger.info(f"📊 表达式: {expression}")
                logger.info(f"📊 解释: {explanation}")
                
                return expression, explanation
            else:
                logger.error(f"❌ 无法解析LLM响应: {response}")
                raise ValueError("无法从LLM响应中提取因子表达式和解释")
        except Exception as e:
            logger.error(f"❌ 生成互补因子表达式失败: {e}")
            # 返回备用互补因子表达式
            fallback_expression = "(close - close.rolling(window=20).mean()) / close.rolling(window=20).std()"
            fallback_explanation = "备用均值回归因子，与动量类因子形成互补"
            logger.warning(f"⚠️ 使用备用互补因子: {fallback_expression}")
            return fallback_expression, fallback_explanation
    
    def optimize_factor_expression(self, 
                                 factor_expression: str,
                                 factor_explanation: str,
                                 evaluation_results: Dict[str, float],
                                 improvement_suggestions: str) -> Tuple[str, str]:
        """
        使用LLM优化现有因子表达式
        
        Args:
            factor_expression: 原始因子表达式
            factor_explanation: 原始因子解释
            evaluation_results: 评估结果
            improvement_suggestions: 改进建议
            
        Returns:
            (优化后的因子表达式, 改进说明)
        """
        # 格式化评估结果
        eval_results_str = "\n".join([
            f"- {key}: {value:.4f}" for key, value in evaluation_results.items()
        ])
        
        # 构建提示词
        prompt = self.prompt_templates["factor_optimization"].format(
            factor_expression=factor_expression,
            factor_explanation=factor_explanation,
            evaluation_results=eval_results_str,
            improvement_suggestions=improvement_suggestions
        )
        
        logger.info(f"📝 调用LLM优化因子表达式")
        
        # 调用LLM
        try:
            # 检查是否在模拟模式
            if hasattr(self, 'mock_mode') and self.mock_mode:
                # 模拟优化响应
                mock_responses = [
                    f"优化后表达式: {factor_expression.replace('20', '15').replace('10', '8')}\n改进说明: 调整了时间窗口参数，使因子对市场变化更敏感。",
                    f"优化后表达式: ({factor_expression}) * volume.pct_change(5).rolling(window=10).mean()\n改进说明: 结合成交量变化提高因子的有效性。",
                    f"优化后表达式: {factor_expression}.rolling(window=5).mean()\n改进说明: 添加了平滑处理，减少因子噪音。"
                ]
                response = random.choice(mock_responses)
                logger.info(f"ℹ️  使用模拟优化响应: {response}")
            else:
                # 调用对应的LLM适配器生成报告
                response = self.llm_adapter.generate_report_section(
                    system_prompt="你是一名专业的量化因子优化专家。",
                    user_prompt=prompt
                )
            
            # 解析响应
            expression_match = re.search(r'优化后表达式:\s*(.*)', response, re.MULTILINE)
            explanation_match = re.search(r'改进说明:\s*(.*)', response, re.MULTILINE)
            
            if expression_match and explanation_match:
                optimized_expression = expression_match.group(1).strip()
                improvement_explanation = explanation_match.group(1).strip()
                
                logger.info(f"✅ 因子优化成功")
                logger.info(f"📊 优化后表达式: {optimized_expression}")
                logger.info(f"📊 改进说明: {improvement_explanation}")
                
                return optimized_expression, improvement_explanation
            else:
                logger.error(f"❌ 无法解析LLM响应: {response}")
                raise ValueError("无法从LLM响应中提取优化后的表达式和说明")
        except Exception as e:
            logger.error(f"❌ 优化因子表达式失败: {e}")
            # 返回原始表达式
            logger.warning(f"⚠️ 优化失败，返回原始因子")
            return factor_expression, "优化失败，保留原始因子"
    
    def validate_factor_expression(self, expression: str) -> Tuple[bool, str]:
        """
        验证因子表达式的有效性
        
        Args:
            expression: 因子表达式
            
        Returns:
            (是否有效, 错误信息)
        """
        # 基本语法检查
        try:
            # 创建一个简单的测试环境
            import pandas as pd
            import numpy as np
            
            # 创建测试数据
            dates = pd.date_range('2020-01-01', periods=10)
            stocks = ['stock1', 'stock2']
            index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock'])
            
            # 创建测试数据框
            test_data = pd.DataFrame({
                'close': np.random.random(20),
                'open': np.random.random(20),
                'high': np.random.random(20),
                'low': np.random.random(20),
                'volume': np.random.random(20) * 1000
            }, index=index)
            
            # 尝试编译表达式
            compiled = compile(expression, '<string>', 'eval')
            
            # 检查是否使用了允许的变量
            allowed_vars = {'close', 'open', 'high', 'low', 'volume', 'np', 'pd'}
            for name in compiled.co_names:
                if name not in allowed_vars and not hasattr(pd.Series, name) and not hasattr(np, name):
                    return False, f"不允许使用的变量: {name}"
            
            # 检查是否有潜在的危险操作
            dangerous_patterns = [
                r'__[^_]+__',  # 双下划线方法
                r'import\s+',  # import语句
                r'exec\s*\(',  # exec函数
                r'eval\s*\(',  # eval函数
                r'open\s*\(',  # open函数
                r'file\s*\(',  # file函数
                r'compile\s*\(',  # compile函数
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, expression, re.IGNORECASE):
                    return False, f"潜在的危险操作: {pattern}"
            
            return True, "表达式有效"
        except SyntaxError as e:
            return False, f"语法错误: {e}"
        except Exception as e:
            return False, f"验证失败: {e}"
    
    def generate_improvement_suggestions(self, evaluation_results: Dict[str, float]) -> str:
        """
        根据评估结果生成改进建议
        
        Args:
            evaluation_results: 评估结果
            
        Returns:
            改进建议
        """
        suggestions = []
        
        # 基于IC的建议
        if evaluation_results["ic"] < 0.01:
            suggestions.append("提高因子与未来收益率的相关性，可以尝试调整时间窗口或结合其他数据源")
        elif evaluation_results["ic"] < 0.02:
            suggestions.append("IC值有提升空间，可以考虑改进因子的时间序列特性")
        
        # 基于夏普比率的建议
        if evaluation_results["sharpe"] < 0.5:
            suggestions.append("降低因子的波动性，可以尝试添加平滑处理或调整权重")
        
        # 基于收益率的建议
        if evaluation_results["annual_return"] < 0:
            suggestions.append("因子收益为负，需要重新设计因子逻辑")
        elif evaluation_results["annual_return"] < 0.1:
            suggestions.append("尝试提高因子的收益潜力，可能需要调整参数或结合非线性变换")
        
        # 综合建议
        if not suggestions:
            suggestions.append("因子表现良好，可以尝试微调参数或添加更多特征")
        
        return "\n".join(suggestions)