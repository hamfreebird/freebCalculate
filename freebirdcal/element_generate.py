import math
import re
import time
import element_data
from pubchempy import get_compounds
from itertools import product
from collections import defaultdict
from rdkit import Chem


class ElementCompoundGenerate:
    def __init__(self, valence_dict):
        self.valence = valence_dict
        self.elements = list(valence_dict.keys())

    def calculate_compounds(self):
        compounds = []
        for elem1, elem2 in product(self.elements, repeat=2):
            valences1 = self.valence[elem1]
            valences2 = self.valence[elem2]
            for v1 in valences1:
                for v2 in valences2:
                    # 检查价是否为相反的符号或都为零
                    if (v1 * v2 < 0) or (v1 == 0 and v2 == 0):
                        if v1 == 0 and v2 == 0:
                            x, y = 1, 1
                            part1 = elem1 + (str(x) if x != 1 else '')
                            part2 = elem2 + (str(y) if y != 1 else '')
                            formula = part1 + part2
                        else:
                            gcd_val = math.gcd(abs(v1), abs(v2))
                            x = abs(v2) // gcd_val
                            y = abs(v1) // gcd_val
                            # 根据正价确定顺序
                            if v1 > 0:
                                part1 = elem1 + (str(x) if x != 1 else '')
                                part2 = elem2 + (str(y) if y != 1 else '')
                            else:
                                part1 = elem2 + (str(y) if y != 1 else '')
                                part2 = elem1 + (str(x) if x != 1 else '')
                            formula = part1 + part2
                        compounds.append(formula)
        # 删除重复项并返回
        unique_compounds = list(set(compounds))
        return sorted(unique_compounds)


class ThreeElementCompoundGenerate:
    def __init__(self, valence_dict):
        self.valence = valence_dict
        self.elements = list(valence_dict.keys())

    def calculate_compounds(self, max_coeff=4):
        compounds = []
        # 生成所有可能的三元素组合（允许重复）并按排序去重
        seen_combos = set()
        for combo in product(self.elements, repeat=3):
            sorted_combo = tuple(sorted(combo))
            if sorted_combo not in seen_combos:
                seen_combos.add(sorted_combo)

        # 遍历每个唯一的元素组合
        for combo in seen_combos:
            elem_counts = defaultdict(list)
            for idx, elem in enumerate(combo):
                elem_counts[elem].append(idx)

            # 生成所有可能的价态组合
            valence_choices = {}
            # 处理需要统一价态的元素
            common_valence_elems = [elem for elem, pos in elem_counts.items() if len(pos) > 1]
            common_valence_lists = [self.valence[elem] for elem in common_valence_elems]
            for common_vals in product(*common_valence_lists):
                valence_dict = dict(zip(common_valence_elems, common_vals))
                # 处理非共同元素的价态
                non_common_elems = [elem for elem in combo if elem not in common_valence_elems]
                non_common_valences = [self.valence[elem] for elem in non_common_elems]
                for non_common_vals in product(*non_common_valences):
                    # 构建完整的价态数组
                    valence = [0] * 3
                    # 填充共同元素的价态
                    for elem in common_valence_elems:
                        for pos in elem_counts[elem]:
                            valence[pos] = valence_dict[elem]
                    # 填充非共同元素的价态
                    nc_idx = 0
                    for i, elem in enumerate(combo):
                        if elem in non_common_elems:
                            valence[i] = non_common_vals[nc_idx]
                            nc_idx += 1
                    # 寻找所有可能的系数组合
                    for a in range(1, max_coeff + 1):
                        for b in range(1, max_coeff + 1):
                            for c in range(1, max_coeff + 1):
                                if a * valence[0] + b * valence[1] + c * valence[2] == 0:
                                    # 计算最简比例
                                    gcd_val = math.gcd(math.gcd(a, b), c)
                                    sa, sb, sc = a // gcd_val, b // gcd_val, c // gcd_val
                                    # 合并相同元素的系数
                                    element_counts = defaultdict(int)
                                    for elem, coeff in zip(combo, [sa, sb, sc]):
                                        element_counts[elem] += coeff
                                    # 生成化学式
                                    formula_parts = []
                                    for elem in sorted(element_counts.keys()):
                                        count = element_counts[elem]
                                        formula_parts.append(elem + (str(count) if count > 1 else ''))
                                    formula = ''.join(formula_parts)
                                    compounds.append(formula)
        # 去重并排序
        return sorted(list(set(compounds)))


def standardize_formula(formula):
    """将化学式标准化为Hill系统排序"""
    # 使用正则表达式分解元素和原子数
    elements = re.findall(r'([A-Z][a-z]*)(\d*)', formula)

    # 统计各元素总原子数
    element_counts = {}
    for elem, num in elements:
        count = int(num) if num else 1
        element_counts[elem] = element_counts.get(elem, 0) + count

    # Hill系统排序规则
    has_carbon = 'C' in element_counts
    sorted_elements = []

    if has_carbon:
        # 有机化合物：C先，H次，其他按字母序
        sorted_elements.append(('C', element_counts.pop('C')))
        if 'H' in element_counts:
            sorted_elements.append(('H', element_counts.pop('H')))

    # 剩余元素按字母顺序排序
    remaining = sorted(element_counts.items())
    sorted_elements.extend(remaining)

    # 生成标准化学式
    parts = []
    for elem, count in sorted_elements:
        parts.append(elem + (str(count) if count > 1 else ''))

    return ''.join(parts)


def is_compound_valid(formula, max_retries=3):
    """验证化学式是否存在于PubChem数据库"""
    try:
        std_formula = standardize_formula(formula)

        # 最大重试次数
        for _ in range(max_retries):
            try:
                results = get_compounds(std_formula, 'formula')
                return len(results) > 0
            except Exception as e:
                print(f"查询错误: {str(e)}，5秒后重试...")
                time.sleep(5)

        return False  # 所有重试失败

    except Exception as e:
        print(f"标准化错误: {str(e)}")
        return False

    finally:
        time.sleep(0.3)  # 请求间隔保护


def is_chemical_formula_valid(formula):
    """
    使用 RDKit 验证化学式格式是否合法
    """
    try:
        # 获取周期表实例
        pt = Chem.GetPeriodicTable()

        # 预处理：去除电荷符号（如 Fe^3+ → Fe）
        base_formula = re.sub(r'[\^±+-]\d*$', '', formula)

        # 分解括号结构并验证（如 Fe3(SO4)3 → Fe3S3O12）
        def expand_parentheses(match):
            group = match.group(1)
            count = int(match.group(2)) if match.group(2) else 1
            return group * count

        # 递归展开嵌套括号
        while True:
            new_formula = re.sub(r'\(([A-Za-z0-9]+)\)(\d*)', expand_parentheses, base_formula)
            if new_formula == base_formula:
                break
            base_formula = new_formula

        # 验证元素符号和原子数
        elements = re.findall(r'([A-Z][a-z]*)(\d*)', base_formula)
        if not elements:
            return False

        for elem, num_str in elements:
            # 检查元素是否存在
            if pt.GetAtomicNumber(elem) == 0:
                return False
            # 检查原子数是否合法
            if num_str and (not num_str.isdigit() or int(num_str) == 0):
                return False

        return True

    except Exception as e:
        print(f"验证错误: {str(e)}")
        return False


if __name__ == "__main__":
    calculator = ElementCompoundGenerate(element_data.oxidation_states)
    compounds = calculator.calculate_compounds()
    for compound in compounds:
        compound = standardize_formula(compound)
        print(compound + "  " + str(is_chemical_formula_valid(compound)))

    calculator = ThreeElementCompoundGenerate(element_data.oxidation_states)
    compounds = calculator.calculate_compounds(max_coeff=3)
    for compound in compounds:
        compound = standardize_formula(compound)
        print(compound + "  " + str(is_chemical_formula_valid(compound)))

