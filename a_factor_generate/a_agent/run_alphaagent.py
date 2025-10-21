#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AlphaAgent因子生成器
用于通过LLM API调用生成A股市场的Alpha因子
"""

import os
import sys
import json
import logging
import subprocess
import shutil
import time
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'alphaagent_run.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AlphaAgentRunner')

class AlphaAgentRunner:
    """AlphaAgent因子生成器主类"""
    
    def __init__(self):
        """初始化AlphaAgent运行器"""
        self.logger = logger
        self.logger.info("=" * 40)
        self.logger.info("AlphaAgent因子生成器初始化中...")
        
        # 依赖列表
        self.dependencies = [
            'pandas', 
            'numpy', 
            'requests',
            'pandarallel',
            'dashscope',
            'openai'
        ]
        
        # 项目路径
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(os.path.dirname(__file__), 'generated_factors')
        self.data_dir = os.path.join(self.project_root, '..', 'data', 'a_share', 'csi500data')
    
    def install_dependencies(self):
        """安装必要的依赖包"""
        self.logger.info("开始安装依赖包...")
        try:
            for dep in self.dependencies:
                self.logger.info(f"安装依赖: {dep}")
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', 
                    '--upgrade', '--force-reinstall', '--user', dep
                ])
            self.logger.info("所有依赖包安装成功！")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"依赖安装失败: {e}")
            return False
    
    def setup_environment(self):
        """设置运行环境，包括API密钥和环境变量"""
        self.logger.info("设置运行环境...")
        
        # API密钥配置 - 使用阿里百炼
        api_key = os.environ.get('DASHSCOPE_API_KEY', '')
        if not api_key:
            self.logger.error("未设置DASHSCOPE_API_KEY环境变量！")
            self.logger.warning("请设置DASHSCOPE_API_KEY环境变量后再运行。")
            self.logger.warning("示例: set DASHSCOPE_API_KEY=your_api_key_here")
            # 为了演示，我们设置一个默认值，但实际使用时应提供真实密钥
            api_key = "demo_api_key"
            self.logger.warning("使用默认演示密钥，请确保设置真实密钥以获得正确的API调用结果")
        
        # 设置必要的环境变量
        os.environ['DASHSCOPE_API_KEY'] = api_key
        os.environ['OPENAI_API_KEY'] = api_key  # 确保OpenAI客户端也能获取到密钥
        os.environ['OPENAI_BASE_URL'] = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        os.environ['OPENAI_MODEL_NAME'] = 'qwen-plus'
        
        self.logger.info("环境变量设置完成")
        self.logger.info(f"API基础URL: {os.environ['OPENAI_BASE_URL']}")
        self.logger.info(f"模型名称: {os.environ['OPENAI_MODEL_NAME']}")
        
        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            self.logger.info(f"创建输出目录: {self.output_dir}")
    
    def prepare_data(self):
        """准备CSI500数据"""
        self.logger.info("检查数据文件...")
        
        if not os.path.exists(self.data_dir):
            self.logger.warning(f"数据目录不存在: {self.data_dir}")
            self.logger.info("尝试从上级目录复制数据...")
            
            # 尝试复制数据文件
            try:
                data_src = os.path.join(self.project_root, '..', 'data', 'csi500data')
                if os.path.exists(data_src):
                    shutil.copytree(data_src, self.data_dir)
                    self.logger.info("数据复制成功")
                else:
                    self.logger.warning("数据源目录也不存在，使用模拟数据进行处理")
            except Exception as e:
                self.logger.error(f"数据复制失败: {e}")
        
        # 检查是否有数据文件
        if os.path.exists(self.data_dir):
            files = os.listdir(self.data_dir)
            if files:
                self.logger.info(f"找到{len(files)}个数据文件")
            else:
                self.logger.warning("数据目录为空")
    
    def import_alphaagent(self):
        """动态导入AlphaAgent模块"""
        self.logger.info("尝试导入AlphaAgent模块...")
        
        # 查找AlphaAgent模块路径
        alphaagent_paths = [
            os.path.join(self.project_root, '..', 'AlphaAgent-main'),
            os.path.join(self.project_root, '..', 'AGI-Alpha-Agent-v0-main'),
        ]
        
        alphaagent_found = False
        for path in alphaagent_paths:
            if os.path.exists(path) and os.path.isdir(path):
                sys.path.append(path)
                self.logger.info(f"添加AlphaAgent路径: {path}")
                alphaagent_found = True
                break
        
        if not alphaagent_found:
            self.logger.error("未找到AlphaAgent模块，请确保AlphaAgent-main或AGI-Alpha-Agent-v0-main目录存在")
            return False
        
        try:
            # 导入必要的AlphaAgent组件
            global AlphaAgentLoop
            from alphaagent.core.loop import AlphaAgentLoop
            self.logger.info("AlphaAgent模块导入成功")
            return True
        except ImportError as e:
            self.logger.error(f"AlphaAgent模块导入失败: {e}")
            self.logger.warning("尝试另一种导入方式...")
            try:
                # 尝试从不同的路径导入
                global AlphaAgent
                import AlphaAgent
                self.logger.info("AlphaAgent备用导入成功")
                return True
            except ImportError as e2:
                self.logger.error(f"AlphaAgent备用导入也失败: {e2}")
                return False
    
    def generate_factors_with_api(self):
        """使用外部API生成因子"""
        self.logger.info("开始使用外部API生成因子...")
        
        try:
            # 1. 初始化AlphaAgent循环
            self.logger.info("初始化AlphaAgentLoop...")
            
            # 配置参数
            config = {
                "model_name": os.environ.get("OPENAI_MODEL_NAME", "qwen-plus"),
                "api_base": os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                "api_key": os.environ.get("OPENAI_API_KEY", ""),
                "max_retries": 5,
                "timeout": 60
            }
            
            # 尝试初始化AlphaAgentLoop
            try:
                loop = AlphaAgentLoop(config=config)
                self.logger.info("AlphaAgentLoop初始化成功")
            except Exception as e:
                self.logger.error(f"AlphaAgentLoop初始化失败: {e}")
                self.logger.info("尝试使用简化配置...")
                # 尝试使用最小配置初始化
                loop = AlphaAgentLoop()
            
            # 2. 运行因子生成过程
            self.logger.info("开始因子生成过程...")
            
            # 定义因子生成的指导信息
            guidance = {
                "market": "A股",
                "index": "CSI500",
                "factors_requested": 5,
                "factor_types": ["动量", "价值", "波动率", "成交量", "技术指标"],
                "requirements": "因子必须使用pandas和numpy实现，包含完整的文档字符串，支持向量化运算，有良好的回测性能。"
            }
            
            # 运行因子生成循环
            self.logger.info(f"发送因子生成请求: {guidance}")
            factors = loop.run(guidance=guidance)
            
            # 3. 保存生成的因子
            if factors and len(factors) > 0:
                self.logger.info(f"成功生成{len(factors)}个因子")
                
                for i, factor in enumerate(factors):
                    factor_name = factor.get("name", f"factor_{i+1}")
                    factor_code = factor.get("code", "")
                    factor_desc = factor.get("description", "")
                    
                    if factor_code:
                        # 保存因子代码文件
                        factor_file = os.path.join(self.output_dir, f"{factor_name}.py")
                        with open(factor_file, 'w', encoding='utf-8') as f:
                            f.write(factor_code)
                        self.logger.info(f"因子保存成功: {factor_file}")
                    
                # 生成README文件
                self._generate_readme(factors)
                return True
            else:
                self.logger.error("未生成任何因子")
                return False
                
        except Exception as e:
            self.logger.error(f"因子生成过程中发生错误: {e}")
            # 如果AlphaAgentLoop调用失败，尝试直接调用API
            self.logger.info("尝试直接调用API生成因子...")
            return self._direct_api_call()
    
    def _direct_api_call(self):
        """直接调用API生成因子"""
        self.logger.info("执行直接API调用...")
        
        try:
            # 导入requests模块
            import requests
            
            # 准备API请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('DASHSCOPE_API_KEY', '')}"
            }
            
            # 定义因子生成提示
            # 定义因子生成提示
            prompt = "作为A股量化策略专家，请为CSI500指数成分股生成5个有效的Alpha因子。\n\n每个因子要求：\n1. 使用pandas和numpy实现\n2. 包含完整的文档字符串，解释因子原理和计算方法\n3. 因子函数名以'factor_'开头\n4. 输入参数为dataframe，包含'open', 'high', 'low', 'close', 'volume'等列\n5. 返回一个series，表示因子值\n6. 因子要有明确的投资逻辑\n7. 考虑A股市场的特殊性\n\n请按以下格式返回5个因子：\n```python\n# 因子1: 动量因子\ndef factor_momentum(data, window=20):\n    \"\"\"\n    动量因子: 衡量股票价格的变化趋势\n    \n    参数:\n        data: pandas.DataFrame, 包含价格数据\n        window: int, 计算窗口\n    \n    返回:\n        pandas.Series: 因子值\n    \"\"\"\n    # 实现代码\n    return data['close'].pct_change(window)\n```"
            
            # 准备请求体
            payload = {
                "model": os.environ.get("OPENAI_MODEL_NAME", "qwen-plus"),
                "messages": [
                    {"role": "system", "content": "你是一位专业的量化分析师，精通A股市场的Alpha因子开发。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 2000
            }
            
            # 发送API请求
            self.logger.info(f"向API发送请求: {os.environ.get('OPENAI_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')}/chat/completions")
            response = requests.post(
                f"{os.environ.get('OPENAI_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                self.logger.info("API调用成功，开始解析因子代码...")
                
                # 解析生成的因子代码
                factors = self._parse_api_response(content)
                
                if factors and len(factors) > 0:
                    self.logger.info(f"成功解析{len(factors)}个因子")
                    
                    # 保存因子文件
                    for i, (factor_name, factor_code) in enumerate(factors.items()):
                        factor_file = os.path.join(self.output_dir, f"{factor_name}.py")
                        with open(factor_file, 'w', encoding='utf-8') as f:
                            f.write(factor_code)
                        self.logger.info(f"因子保存成功: {factor_file}")
                    
                    # 生成README
                    self._generate_readme_simple(factors)
                    return True
                else:
                    self.logger.error("未能从API响应中解析出有效因子")
                    return False
            else:
                self.logger.error(f"API调用失败: 状态码 {response.status_code}, 响应: {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"直接API调用失败: {e}")
            return False
    
    def _parse_api_response(self, content):
        """解析API响应中的因子代码"""
        factors = {}
        
        try:
            # 提取Python代码块
            import re
            code_blocks = re.findall(r'```python(.*?)```', content, re.DOTALL)
            
            for block in code_blocks:
                # 查找函数定义
                func_match = re.search(r'def\s+(factor_\w+)\(', block)
                if func_match:
                    factor_name = func_match.group(1)
                    factors[factor_name] = block.strip()
            
            # 如果没有找到代码块，尝试直接查找函数
            if not factors:
                functions = re.findall(r'(def\s+factor_\w+\(.*?\):\s*(?:"""[\s\S]*?"""\s*)?[\s\S]*?)(?=def\s+factor_|$)', content, re.MULTILINE)
                for func in functions:
                    func_match = re.search(r'def\s+(factor_\w+)\(', func)
                    if func_match:
                        factor_name = func_match.group(1)
                        factors[factor_name] = func.strip()
            
        except Exception as e:
            self.logger.error(f"解析API响应失败: {e}")
        
        return factors
    
    def _generate_readme(self, factors):
        """生成因子说明文档"""
        readme_path = os.path.join(self.output_dir, "README.md")
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"""# AlphaAgent生成的A股因子

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 因子列表

""")
            
            for i, factor in enumerate(factors):
                name = factor.get("name", f"因子{i+1}")
                desc = factor.get("description", "无描述")
                file_name = f"{name}.py"
                
                f.write(f"### {i+1}. {name}\n")
                f.write(f"**文件**: {file_name}\n")
                f.write(f"**描述**: {desc}\n\n")
            
            f.write("""
## 使用方法

1. 将生成的因子文件导入到您的量化策略中
2. 使用pandas读取CSI500数据
3. 调用相应的因子函数计算因子值
4. 根据因子值进行选股和回测

## 注意事项

- 因子需要与CSI500成分股数据配合使用
- 建议在使用前对因子进行回测验证
- 可以根据市场情况调整因子参数
""")
        
        self.logger.info(f"README文件生成成功: {readme_path}")
    
    def _generate_readme_simple(self, factors):
        """生成简化版README文档"""
        readme_path = os.path.join(self.output_dir, "README.md")
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"""# AlphaAgent生成的A股因子

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 因子列表

""")
            
            for i, (factor_name, _) in enumerate(factors.items()):
                file_name = f"{factor_name}.py"
                
                f.write(f"### {i+1}. {factor_name}\n")
                f.write(f"**文件**: {file_name}\n\n")
            
            f.write("""
## 使用方法

1. 将生成的因子文件导入到您的量化策略中
2. 使用pandas读取CSI500数据
3. 调用相应的因子函数计算因子值
4. 根据因子值进行选股和回测

## 注意事项

- 因子需要与CSI500成分股数据配合使用
- 建议在使用前对因子进行回测验证
- 可以根据市场情况调整因子参数
""")
        
        self.logger.info(f"简化版README文件生成成功: {readme_path}")
    
    def run(self):
        """运行完整的因子生成流程"""
        try:
            self.logger.info("开始AlphaAgent因子生成流程...")
            
            # 1. 安装依赖
            if not self.install_dependencies():
                self.logger.error("依赖安装失败，流程终止")
                return False
            
            # 2. 设置环境
            self.setup_environment()
            
            # 3. 准备数据
            self.prepare_data()
            
            # 4. 导入AlphaAgent
            imported = self.import_alphaagent()
            
            # 5. 生成因子
            success = self.generate_factors_with_api()
            
            if success:
                self.logger.info("[SUCCESS] AlphaAgent因子生成流程成功完成！")
                self.logger.info(f"因子文件保存在: {self.output_dir}")
                return True
            else:
                self.logger.error("[ERROR] AlphaAgent因子生成失败")
                return False
                
        except Exception as e:
            self.logger.error(f"运行过程中发生未预期的错误: {e}")
            return False


def main():
    """主函数"""
    runner = AlphaAgentRunner()
    success = runner.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()