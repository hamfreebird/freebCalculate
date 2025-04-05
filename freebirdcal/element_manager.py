import re
import csv
from collections import defaultdict
from abc import ABC, abstractmethod
from element_data import *


class ElementCompoundManager(ABC):
    """化合物管理基类"""

    def __init__(self, element_symbol: str):
        self.element_symbol = element_symbol
        self.compounds = []
        self._atomic_weights = self._get_default_atomic_weights()
        self._init_default_compounds()  # 确保调用初始化方法

    @abstractmethod
    def _get_default_atomic_weights(self) -> dict:
        """获取默认原子量（需子类实现）"""
        pass

    @abstractmethod
    def _init_default_compounds(self):
        """初始化默认化合物（需子类实现）"""
        pass

    def _parse_simple_formula(self, formula: str) -> float:
        """解析简单化学式（无括号）"""
        elements = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
        total = 0.0
        for elem, count in elements:
            if elem not in self._atomic_weights:
                raise ValueError(f"未知元素 {elem}")
            count = int(count) if count else 1
            total += self._atomic_weights[elem] * count
        return total

    def _parse_complex_formula(self, formula: str) -> float:
        """解析含括号的复杂化学式"""
        total = 0.0
        # 使用正则表达式查找所有括号及其乘数
        pattern = re.compile(r'\(([A-Za-z0-9]+)\)(\d*)')
        matches = list(pattern.finditer(formula))

        # 从右到左处理以避免影响后续匹配位置
        for match in reversed(matches):
            subunit = match.group(1)
            multiplier_str = match.group(2)
            multiplier = int(multiplier_str) if multiplier_str else 1
            subunit_weight = self._parse_simple_formula(subunit) * multiplier
            total += subunit_weight
            # 替换处理过的部分
            start = match.start()
            end = match.end()
            formula = formula[:start] + formula[end:]

        # 处理剩余部分
        return total + self._parse_simple_formula(formula)

    def add_compound(self, name: str, formula: str, oxidation_states: list,
                     phase: str, uses: list):
        """添加新化合物"""
        try:
            mw = self._parse_complex_formula(formula)
            self.compounds.append({
                "name": name,
                "formula": formula,
                "molecular_weight": round(mw, 2),
                "oxidation_states": oxidation_states,
                "phase": phase,
                "uses": uses
            })
            print(f"成功添加：{name} ({formula})")
        except (ValueError, KeyError) as e:
            print(f"添加失败：{str(e)}")

    def find_by_formula(self, formula: str) -> list:
        """根据化学式查询化合物"""
        return [c for c in self.compounds if c["formula"] == formula]

    def get_statistics(self) -> dict:
        """获取统计信息"""
        stats = {
            "element": self.element_symbol,
            "total": len(self.compounds),
            "avg_molecular_weight": 0,
            "oxidation_distribution": defaultdict(int),
            "phase_distribution": defaultdict(int)
        }

        total_weight = 0
        for c in self.compounds:
            total_weight += c["molecular_weight"]
            for state in c["oxidation_states"]:
                stats["oxidation_distribution"][state] += 1
            stats["phase_distribution"][c["phase"]] += 1

        if stats["total"] > 0:
            stats["avg_molecular_weight"] = round(total_weight / stats["total"], 2)

        return stats

    def save_to_csv(self, filename: str):
        """保存到CSV文件"""
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "name", "formula", "molecular_weight",
                "oxidation_states", "phase", "uses"
            ])
            writer.writeheader()
            for c in self.compounds:
                c_copy = c.copy()
                c_copy["oxidation_states"] = ",".join(map(str, c["oxidation_states"]))
                c_copy["uses"] = ";".join(c["uses"])
                writer.writerow(c_copy)

    def __str__(self):
        """字符串表示"""
        return f"{self.element_symbol}化合物管理系统（{len(self.compounds)}种化合物）"


class UraniumCompoundManager(ElementCompoundManager):
    """铀化合物专用管理类"""

    def _get_default_atomic_weights(self) -> dict:
        """铀相关原子量"""
        return atomic_weights

    def _init_default_compounds(self):
        """初始化默认铀化合物"""
        for name, formula, states, phase, color, uses in U_EC:
            try:
                self.add_compound(name, formula, states, phase, uses)
            except Exception as e:
                print(f"初始化错误：{name} - {str(e)}")

    def get_nuclear_properties(self):
        """获取核特性统计（铀专用方法）"""
        nuclear_uses = [
            "核燃料", "铀浓缩", "核武器", "反应堆"
        ]
        return {
            "nuclear_compounds": [
                c for c in self.compounds
                if any(use in nuclear_uses for use in c["uses"])
            ],
            "count": len([
                c for c in self.compounds
                if any(use in nuclear_uses for use in c["uses"])
            ])
        }


class CopperCompoundManager(ElementCompoundManager):
    """铜化合物专用管理类"""

    def _get_default_atomic_weights(self) -> dict:
        """铜相关原子量"""
        return atomic_weights

    def _init_default_compounds(self):
        """初始化默认铜化合物"""
        for name, formula, states, phase, color, uses in Cu_EC:
            try:
                self.add_compound(name, formula, states, phase, uses)
            except Exception as e:
                print(f"初始化错误：{name} - {str(e)}")


# 使用示例
if __name__ == "__main__":
    u_manager = UraniumCompoundManager("U")
    print(u_manager)  # 输出：U化合物管理系统

    # 添加新化合物
    u_manager.add_compound(
        name="氟化铀酰",
        formula="UO2F2",
        oxidation_states=[6],
        phase="晶体",
        uses=["催化剂"]
    )

    # 获取统计信息
    stats = u_manager.get_statistics()
    print(f"\n氧化态分布：{dict(stats['oxidation_distribution'])}")

    # 核相关统计
    nuclear_stats = u_manager.get_nuclear_properties()
    print(f"核相关化合物数量：{nuclear_stats['count']}")

    # 保存数据
    u_manager.save_to_csv("uranium_compounds.csv")

