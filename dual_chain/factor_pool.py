# -*- coding: utf-8 -*-
"""
因子池管理器
管理有效因子池(effective_pool)和废弃因子池(discarded_pool)
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dual_chain.factor_pool')

class FactorPool:
    """
    因子池管理器
    负责有效因子池和废弃因子池的存储、读取和更新
    """
    
    def __init__(self, pool_dir: str = "dual_chain/pools", initial_factors_file: str = None):
        """
        初始化因子池管理器
        
        Args:
            pool_dir: 因子池存储目录
            initial_factors_file: 初始因子文件路径（可选）
        """
        self.pool_dir = pool_dir
        self.effective_pool_path = os.path.join(pool_dir, "effective_pool")
        self.discarded_pool_path = os.path.join(pool_dir, "discarded_pool")
        self.metadata_path = os.path.join(pool_dir, "metadata.json")
        
        # 创建必要的目录
        os.makedirs(self.effective_pool_path, exist_ok=True)
        os.makedirs(self.discarded_pool_path, exist_ok=True)
        
        # 初始化元数据
        self._init_metadata()
        
        # 如果提供了初始因子文件，加载初始因子
        if initial_factors_file and os.path.exists(initial_factors_file):
            logger.info(f"📥 开始从初始因子文件加载因子: {initial_factors_file}")
            self.load_initial_factors(initial_factors_file)
        
        logger.info(f"✅ 因子池管理器初始化成功，有效池路径: {self.effective_pool_path}")
        logger.info(f"✅ 因子池管理器初始化成功，废弃池路径: {self.discarded_pool_path}")
    
    def _init_metadata(self):
        """
        初始化元数据
        """
        if not os.path.exists(self.metadata_path):
            metadata = {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "effective_factors_count": 0,
                "discarded_factors_count": 0,
                "total_iterations": 0
            }
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def update_metadata(self, key: str, value: Any):
        """
        更新元数据
        
        Args:
            key: 元数据键
            value: 元数据值
        """
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        metadata[key] = value
        metadata["last_updated"] = datetime.now().isoformat()
        
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def add_effective_factor(self, factor_name: str, factor_data: pd.DataFrame, 
                           factor_expression: str, evaluation_metrics: Dict[str, float]):
        """
        添加有效因子到有效因子池
        
        Args:
            factor_name: 因子名称
            factor_data: 因子数据
            factor_expression: 因子表达式/描述
            evaluation_metrics: 评估指标
        """
        # 保存因子数据
        data_path = os.path.join(self.effective_pool_path, f"{factor_name}_data.pkl")
        factor_data.to_pickle(data_path)
        
        # 保存因子元信息
        metadata = {
            "name": factor_name,
            "expression": factor_expression,
            "evaluation_metrics": evaluation_metrics,
            "added_at": datetime.now().isoformat()
        }
        
        meta_path = os.path.join(self.effective_pool_path, f"{factor_name}_metadata.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 更新总计数
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        metadata["effective_factors_count"] = len(self.get_effective_factors_list())
        self.update_metadata("effective_factors_count", metadata["effective_factors_count"])
        
        logger.info(f"✅ 因子添加到有效池: {factor_name}, IC: {evaluation_metrics.get('ic', 'N/A'):.4f}")
    
    def add_discarded_factor(self, factor_name: str, factor_data: pd.DataFrame, 
                           factor_expression: str, evaluation_metrics: Dict[str, float],
                           reason: str):
        """
        添加废弃因子到废弃因子池
        
        Args:
            factor_name: 因子名称
            factor_data: 因子数据
            factor_expression: 因子表达式/描述
            evaluation_metrics: 评估指标
            reason: 废弃原因
        """
        # 保存因子数据
        data_path = os.path.join(self.discarded_pool_path, f"{factor_name}_data.pkl")
        factor_data.to_pickle(data_path)
        
        # 保存因子元信息
        metadata = {
            "name": factor_name,
            "expression": factor_expression,
            "evaluation_metrics": evaluation_metrics,
            "discarded_at": datetime.now().isoformat(),
            "reason": reason
        }
        
        meta_path = os.path.join(self.discarded_pool_path, f"{factor_name}_metadata.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 更新总计数
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        metadata["discarded_factors_count"] = len(self.get_discarded_factors_list())
        self.update_metadata("discarded_factors_count", metadata["discarded_factors_count"])
        
        logger.info(f"❌ 因子添加到废弃池: {factor_name}, 原因: {reason}")
    
    def get_effective_factors_list(self) -> List[str]:
        """
        获取有效因子列表
        
        Returns:
            有效因子名称列表
        """
        factors = []
        for file in os.listdir(self.effective_pool_path):
            if file.endswith("_metadata.json"):
                factor_name = file.replace("_metadata.json", "")
                factors.append(factor_name)
        return factors
    
    def get_discarded_factors_list(self) -> List[str]:
        """
        获取废弃因子列表
        
        Returns:
            废弃因子名称列表
        """
        factors = []
        for file in os.listdir(self.discarded_pool_path):
            if file.endswith("_metadata.json"):
                factor_name = file.replace("_metadata.json", "")
                factors.append(factor_name)
        return factors
    
    def get_factor_metadata(self, factor_name: str, is_effective: bool = True) -> Dict[str, Any]:
        """
        获取因子元信息
        
        Args:
            factor_name: 因子名称
            is_effective: 是否为有效因子
            
        Returns:
            因子元信息
        """
        pool_path = self.effective_pool_path if is_effective else self.discarded_pool_path
        meta_path = os.path.join(pool_path, f"{factor_name}_metadata.json")
        
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"因子元信息不存在: {factor_name}")
        
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_initial_factors(self, factors_file: str) -> List[Dict[str, Any]]:
        """
        从JSON文件加载初始因子
        
        Args:
            factors_file: 因子文件路径
            
        Returns:
            加载的因子列表
        """
        try:
            with open(factors_file, 'r', encoding='utf-8') as f:
                factors_data = json.load(f)
            
            loaded_factors = []
            # 遍历所有来源的因子
            for source, factors in factors_data.items():
                logger.info(f"🔍 处理来源: {source}，包含 {len(factors)} 个因子")
                for factor in factors:
                    # 提取因子信息
                    factor_info = {
                        "name": factor.get("name", f"{source}_factor_{len(loaded_factors)+1}"),
                        "expression": factor.get("expression", ""),
                        "description": factor.get("description", ""),
                        "source": source,
                        "parameters": factor.get("parameters", {})
                    }
                    loaded_factors.append(factor_info)
            
            logger.info(f"✅ 成功加载 {len(loaded_factors)} 个初始因子")
            return loaded_factors
            
        except Exception as e:
            logger.error(f"❌ 加载初始因子失败: {e}")
            return []
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """
        获取因子池统计信息
        
        Returns:
            因子池统计信息
        """
        effective_factors = self.get_effective_factors_list()
        discarded_factors = self.get_discarded_factors_list()
        
        stats = {
            "effective_factors_count": len(effective_factors),
            "discarded_factors_count": len(discarded_factors),
            "total_factors": len(effective_factors) + len(discarded_factors)
        }
        
        # 计算平均指标
        effective_ics = []
        effective_ic_irs = []
        effective_sharpes = []
        
        for factor_name in effective_factors:
            try:
                metadata = self.get_factor_metadata(factor_name)
                metrics = metadata.get("evaluation_metrics", {})
                if "ic" in metrics:
                    effective_ics.append(metrics["ic"])
                if "ic_ir" in metrics:
                    effective_ic_irs.append(metrics["ic_ir"])
                if "sharpe" in metrics:
                    effective_sharpes.append(metrics["sharpe"])
            except Exception as e:
                logger.error(f"❌ 获取因子 {factor_name} 统计信息失败: {e}")
        
        if effective_ics:
            stats["avg_effective_ic"] = np.mean(effective_ics)
            stats["avg_effective_ic_ir"] = np.mean(effective_ic_irs) if effective_ic_irs else 0
            stats["avg_effective_sharpe"] = np.mean(effective_sharpes) if effective_sharpes else 0
        
        discarded_ics = []
        discarded_sharpes = []
        
        for factor_name in discarded_factors:
            try:
                metadata = self.get_factor_metadata(factor_name, is_effective=False)
                metrics = metadata.get("evaluation_metrics", {})
                if "ic" in metrics:
                    discarded_ics.append(metrics["ic"])
                if "sharpe" in metrics:
                    discarded_sharpes.append(metrics["sharpe"])
            except Exception as e:
                logger.error(f"❌ 获取废弃因子 {factor_name} 统计信息失败: {e}")
        
        if discarded_ics:
            stats["avg_discarded_ic"] = np.mean(discarded_ics)
            stats["avg_discarded_sharpe"] = np.mean(discarded_sharpes) if discarded_sharpes else 0
        
        return stats
    
    def get_factor_data(self, factor_name: str, is_effective: bool = True) -> pd.DataFrame:
        """
        获取因子数据
        
        Args:
            factor_name: 因子名称
            is_effective: 是否为有效因子
            
        Returns:
            因子数据
        """
        pool_path = self.effective_pool_path if is_effective else self.discarded_pool_path
        data_path = os.path.join(pool_path, f"{factor_name}_data.pkl")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"因子数据不存在: {factor_name}")
        
        return pd.read_pickle(data_path)
    
    def get_reference_factors(self, top_n: int = 5) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        获取参考因子，用于指导新因子生成
        
        Args:
            top_n: 返回前N个因子
            
        Returns:
            (有效因子列表, 废弃因子列表)
        """
        # 获取有效因子，按IC排序
        effective_factors = []
        for factor_name in self.get_effective_factors_list():
            metadata = self.get_factor_metadata(factor_name, is_effective=True)
            effective_factors.append(metadata)
        
        # 按IC降序排序
        effective_factors.sort(key=lambda x: x["evaluation_metrics"].get("ic", 0), reverse=True)
        
        # 获取废弃因子，按IC升序排序（最差的因子）
        discarded_factors = []
        for factor_name in self.get_discarded_factors_list():
            metadata = self.get_factor_metadata(factor_name, is_effective=False)
            discarded_factors.append(metadata)
        
        # 按IC升序排序
        discarded_factors.sort(key=lambda x: x["evaluation_metrics"].get("ic", 0), reverse=False)
        
        return effective_factors[:top_n], discarded_factors[:top_n]
    
    def remove_factor(self, factor_name: str, is_effective: bool = True):
        """
        从因子池中移除因子
        
        Args:
            factor_name: 因子名称
            is_effective: 是否为有效因子
        """
        pool_path = self.effective_pool_path if is_effective else self.discarded_pool_path
        
        # 删除数据文件
        data_path = os.path.join(pool_path, f"{factor_name}_data.pkl")
        if os.path.exists(data_path):
            os.remove(data_path)
        
        # 删除元信息文件
        meta_path = os.path.join(pool_path, f"{factor_name}_metadata.json")
        if os.path.exists(meta_path):
            os.remove(meta_path)
        
        # 更新计数
        if is_effective:
            self.update_metadata("effective_factors_count", len(self.get_effective_factors_list()))
        else:
            self.update_metadata("discarded_factors_count", len(self.get_discarded_factors_list()))
        
        logger.info(f"🗑️  因子已从{'有效池' if is_effective else '废弃池'}移除: {factor_name}")
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """
        获取因子池统计信息
        
        Returns:
            统计信息
        """
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 计算有效因子的平均IC
        effective_factors = self.get_effective_factors_list()
        avg_effective_ic = 0
        if effective_factors:
            ics = []
            for factor_name in effective_factors:
                meta = self.get_factor_metadata(factor_name, is_effective=True)
                ics.append(meta["evaluation_metrics"].get("ic", 0))
            avg_effective_ic = np.mean(ics)
        
        # 计算废弃因子的平均IC
        discarded_factors = self.get_discarded_factors_list()
        avg_discarded_ic = 0
        if discarded_factors:
            ics = []
            for factor_name in discarded_factors:
                meta = self.get_factor_metadata(factor_name, is_effective=False)
                ics.append(meta["evaluation_metrics"].get("ic", 0))
            avg_discarded_ic = np.mean(ics)
        
        statistics = {
            "effective_factors_count": metadata["effective_factors_count"],
            "discarded_factors_count": metadata["discarded_factors_count"],
            "total_iterations": metadata["total_iterations"],
            "created_at": metadata["created_at"],
            "last_updated": metadata["last_updated"],
            "avg_effective_ic": avg_effective_ic,
            "avg_discarded_ic": avg_discarded_ic
        }
        
        return statistics