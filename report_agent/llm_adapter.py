#!/usr/bin/env python3
"""
LLM适配器模块
提供多种LLM提供商的统一接口，支持阿里百炼、DeepSeek、OpenAI等
"""

import os
import time
import logging
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

from langchain_core.messages import HumanMessage, SystemMessage

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('report_agent.llm_adapter')


class BaseLLMAdapter(ABC):
    """
    LLM适配器基类
    定义所有LLM适配器必须实现的接口
    """
    
    def __init__(self, 
                 model_name: str,
                 temperature: float = 0.1,
                 max_tokens: int = 4000):
        """
        初始化LLM适配器
        
        Args:
            model_name: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = self._init_llm()
        logger.info(f"✅ LLM适配器初始化成功，提供商: {self.__class__.__name__}，模型: {model_name}")
    
    @abstractmethod
    def _init_llm(self):
        """
        初始化LLM实例
        子类必须实现此方法
        """
        pass
    
    def generate_report_section(self,
                               system_prompt: str,
                               user_prompt: str,
                               data: Optional[Dict] = None) -> str:
        """
        生成报告的特定部分
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            data: 附加数据，将与用户提示词合并
            
        Returns:
            生成的文本内容
        """
        # 构建完整的用户提示词
        full_user_prompt = user_prompt
        if data:
            import json
            data_str = json.dumps(data, ensure_ascii=False, indent=2)
            full_user_prompt += f"\n\n数据:\n{data_str}"
        
        # 构建消息列表
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=full_user_prompt)
        ]
        
        # 记录请求信息
        logger.info(f"📝 生成报告部分，模型: {self.model_name}")
        logger.debug(f"系统提示词长度: {len(system_prompt)}字符")
        logger.debug(f"用户提示词长度: {len(full_user_prompt)}字符")
        
        # 调用LLM生成内容
        start_time = time.time()
        try:
            response = self.llm.invoke(messages)
            content = response.content
            
            # 记录响应信息
            elapsed_time = time.time() - start_time
            logger.info(f"✅ 报告部分生成成功，用时: {elapsed_time:.2f}秒")
            logger.debug(f"生成内容长度: {len(content)}字符")
            
            return content
        except Exception as e:
            logger.error(f"❌ 报告部分生成失败: {str(e)}")
            raise Exception(f"生成报告部分失败: {str(e)}")


class OpenAIAdapter(BaseLLMAdapter):
    """
    OpenAI适配器
    支持OpenAI的GPT模型
    """
    
    def __init__(self,
                 model_name: str = "gpt-4",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: float = 0.1,
                 max_tokens: int = 4000):
        """
        初始化OpenAI适配器
        
        Args:
            model_name: 模型名称
            api_key: API密钥
            base_url: API基础URL
            temperature: 温度参数
            max_tokens: 最大token数
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError("API密钥未找到，请设置OPENAI_API_KEY环境变量或传入api_key参数")
        
        super().__init__(model_name, temperature, max_tokens)
    
    def _init_llm(self):
        """
        初始化OpenAI LLM实例
        """
        from langchain_openai import ChatOpenAI
        
        llm_kwargs = {
            "model": self.model_name,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        # 如果提供了自定义base_url，则添加到参数中
        if self.base_url:
            llm_kwargs["base_url"] = self.base_url
        
        return ChatOpenAI(**llm_kwargs)


class DashScopeAdapter(BaseLLMAdapter):
    """
    阿里百炼适配器
    支持阿里百炼模型
    """
    
    def __init__(self,
                 model_name: str = "qwen-plus",
                 api_key: Optional[str] = None,
                 temperature: float = 0.1,
                 max_tokens: int = 4000):
        """
        初始化阿里百炼适配器
        
        Args:
            model_name: 模型名称
            api_key: API密钥
            temperature: 温度参数
            max_tokens: 最大token数
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        
        if not self.api_key:
            raise ValueError("API密钥未找到，请设置DASHSCOPE_API_KEY环境变量或传入api_key参数")
        
        super().__init__(model_name, temperature, max_tokens)
    
    def _init_llm(self):
        """
        初始化阿里百炼LLM实例
        """
        # 尝试导入DashScope相关库
        try:
            from langchain_community.chat_models import ChatDashScope
            
            return ChatDashScope(
                model=self.model_name,
                dashscope_api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        except ImportError:
            # 如果没有安装langchain_community，可以使用OpenAI兼容模式
            logger.warning("langchain_community未安装，尝试使用OpenAI兼容模式连接阿里百炼")
            from langchain_openai import ChatOpenAI
            
            return ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )


class DeepSeekAdapter(BaseLLMAdapter):
    """
    DeepSeek适配器
    支持DeepSeek模型
    """
    
    def __init__(self,
                 model_name: str = "deepseek-chat",
                 api_key: Optional[str] = None,
                 temperature: float = 0.1,
                 max_tokens: int = 4000):
        """
        初始化DeepSeek适配器
        
        Args:
            model_name: 模型名称
            api_key: API密钥
            temperature: 温度参数
            max_tokens: 最大token数
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        
        if not self.api_key:
            raise ValueError("API密钥未找到，请设置DEEPSEEK_API_KEY环境变量或传入api_key参数")
        
        super().__init__(model_name, temperature, max_tokens)
    
    def _init_llm(self):
        """
        初始化DeepSeek LLM实例
        """
        from langchain_openai import ChatOpenAI
        
        # DeepSeek使用OpenAI兼容接口
        return ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1",
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )


# LLM提供商映射
LLM_PROVIDERS = {
    "openai": OpenAIAdapter,
    "dashscope": DashScopeAdapter,
    "deepseek": DeepSeekAdapter
}


def create_llm_adapter(config: Optional[Dict] = None) -> BaseLLMAdapter:
    """
    创建LLM适配器实例
    
    Args:
        config: LLM配置字典，必须包含provider字段
        
    Returns:
        LLM适配器实例
    """
    if config is None:
        config = {}
    
    # 获取提供商类型，默认为openai
    provider = config.get("provider", "openai").lower()
    
    # 检查提供商是否支持
    if provider not in LLM_PROVIDERS:
        supported = ", ".join(LLM_PROVIDERS.keys())
        raise ValueError(f"不支持的LLM提供商: {provider}，支持的提供商有: {supported}")
    
    # 创建对应的适配器实例
    adapter_class = LLM_PROVIDERS[provider]
    
    # 移除provider字段，避免传递给适配器构造函数
    adapter_config = config.copy()
    adapter_config.pop("provider", None)
    
    # 根据不同提供商设置默认模型
    if "model_name" not in adapter_config:
        if provider == "openai":
            adapter_config["model_name"] = "gpt-4"
        elif provider == "dashscope":
            adapter_config["model_name"] = "qwen-plus"
        elif provider == "deepseek":
            adapter_config["model_name"] = "deepseek-chat"
    
    return adapter_class(**adapter_config)


# 保留旧的LLMAdapter名称以保持向后兼容性
LLMAdapter = OpenAIAdapter