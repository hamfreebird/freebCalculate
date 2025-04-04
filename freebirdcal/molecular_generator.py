from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, rdMolDescriptors, rdchem, GetPeriodicTable
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Draw import MolsToGridImage, IPythonConsole
import random
from collections import defaultdict


class InorganicGeneratorExtension:
    def __init__(self):
        self._extend_periodic_table()
        self.inorganic_fragments = self._build_inorganic_fragments()
        self.metal_oxidation_states = self._get_metal_oxidation_states()

    def _extend_periodic_table(self):
        """扩展RDKit元素属性处理"""
        self.ptable = GetPeriodicTable()
        # 添加稀有气体化合物支持
        self.special_valences = {
            'U': [+4, +6],
            'Xe': [0, +2, +4, +6],  # 氙的常见氧化态
            'Kr': [0, +2],
            'Rn': [0, +2, +4]
        }

    def _build_inorganic_fragments(self):
        """构建无机物特征片段库"""
        return {
            'metal_oxides': [
                ('[O-][M]', ['Fe', 'Cu', 'U', 'Ce']),  # 金属氧化物
                ('[O-][M][O-]', ['Cr', 'Mn', 'W'])  # 多金属氧酸盐
            ],
            'halides': [
                ('[F-]', ['Au', 'Pt', 'Xe']),  # 氟化物
                ('[Cl-]', ['Pb', 'Ag', 'Hg'])  # 氯化物
            ],
            'coordination': [
                ('[N]([M])', ['Co', 'Ni'], 6),  # 六配位结构
                ('[O]([M])', ['Fe', 'Cu'], 4)  # 四配位结构
            ]
        }

    def _get_metal_oxidation_states(self):
        """常见金属氧化态数据库"""
        return {
            'Fe': [+2, +3],
            'Cu': [+1, +2],
            'U': [+4, +6],
            'Xe': [+2, +4, +6],
            'Au': [+1, +3],
            'Pt': [+2, +4]
        }

    def generate_metal_complex(self, metal=None, ligand_type='oxide'):
        """
        生成金属配合物
        :param metal: 指定金属元素符号（如'Fe'）
        :param ligand_type: 配体类型（oxide/halide/ammonia）
        :return: RDKit Mol对象
        """
        # 铀的特殊处理
        if metal == 'U' and ligand_type == 'oxide':
            return self._generate_uranyl_complex()

        # 随机选择金属
        if not metal:
            metals = [m for m in self.metal_oxidation_states.keys()
                      if GetAtomicNumber(m) > 20]  # 原子序数>20
            metal = random.choice(metals)

        # 根据氧化态确定配位数
        common_oxidation = random.choice(self.metal_oxidation_states[metal])
        coord_number = self._get_coordination_number(metal, common_oxidation)

        # 构建配位结构
        mol = Chem.RWMol()
        metal_atom = Chem.Atom(metal)
        metal_atom.SetFormalCharge(common_oxidation)
        mol.AddAtom(metal_atom)

        # 添加配体
        ligands = self._get_ligands(ligand_type, coord_number, common_oxidation)
        for ligand in ligands:
            mol.AddAtom(ligand)
            mol.AddBond(0, mol.GetNumAtoms() - 1, Chem.BondType.DATIVE)  # 配位键

        # 添加反离子平衡电荷
        self._add_counter_ions(mol, common_oxidation * -1)

        Chem.SanitizeMol(mol)
        return mol.GetMol()

    def _get_coordination_number(self, metal, oxidation):
        """根据金属和氧化态确定典型配位数"""
        coord_rules = {
            'Fe': {+2: 6, +3: 6},
            'Cu': {+1: 2, +2: 4},
            'Pt': {+2: 4, +4: 6},
            'Xe': {+4: 4, +6: 6}
        }
        return coord_rules.get(metal, {}).get(oxidation, 4)  # 默认4配位

    def _get_ligands(self, ligand_type, num, oxidation):
        """生成配体原子列表"""
        ligands = []
        if ligand_type == 'oxide':
            for _ in range(num):
                o = Chem.Atom('O')
                o.SetFormalCharge(-2)
                ligands.append(o)
        elif ligand_type == 'halide':
            charge = -1
            for _ in range(num):
                halogen = random.choice(['F', 'Cl'])
                atom = Chem.Atom(halogen)
                atom.SetFormalCharge(charge)
                ligands.append(atom)
        # 可扩展其他配体类型
        return ligands

    def _add_counter_ions(self, mol, total_charge):
        """添加反离子平衡电荷"""
        if total_charge == 0:
            return

        common_ions = {
            +1: ('Na', +1),
            -1: ('Cl', -1),
            +2: ('Ca', +2),
            -2: ('O', -2)  # 以氧化物形式
        }

        ion, charge = common_ions.get(abs(total_charge), ('Cl', -1))
        num_ions = abs(total_charge) // abs(charge)

        for _ in range(num_ions):
            atom = Chem.Atom(ion)
            atom.SetFormalCharge(charge * (-1 if total_charge < 0 else 1))
            mol.AddAtom(atom)

    def generate_noble_gas_compound(self):
        """生成稀有气体化合物"""
        gases = ['Xe', 'Kr', 'Rn']
        compound_type = random.choice(['fluoride', 'oxide'])

        mol = Chem.RWMol()
        gas_atom = Chem.Atom(random.choice(gases))

        # 设置氧化态
        possible_oxidation = self.special_valences[gas_atom.GetSymbol()]
        oxidation = random.choice([o for o in possible_oxidation if o > 0])
        gas_atom.SetFormalCharge(oxidation)
        mol.AddAtom(gas_atom)

        # 添加配位原子
        if compound_type == 'fluoride':
            ligand = 'F'
            ligand_charge = -1
            coord_num = oxidation // 1  # 例如XeF4: Xe(+4), 4 F-
        else:  # oxide
            ligand = 'O'
            ligand_charge = -2
            coord_num = oxidation // 2  # 例如XeO3: Xe(+6), 3 O^2-

        for _ in range(coord_num):
            atom = Chem.Atom(ligand)
            atom.SetFormalCharge(ligand_charge)
            mol.AddAtom(atom)
            mol.AddBond(0, mol.GetNumAtoms() - 1, Chem.BondType.SINGLE)

        # 平衡电荷
        self._add_counter_ions(mol, oxidation + coord_num * ligand_charge)

        Chem.SanitizeMol(mol)
        return mol.GetMol()

    def _generate_uranyl_complex(self):
        """生成正确的铀酰（UO2²⁺）结构"""
        uranyl = Chem.RWMol()

        # ========== 铀酰核心构建 ==========
        # 添加铀原子（+6氧化态）
        u = Chem.Atom('U')
        u.SetFormalCharge(+6)
        uranyl.AddAtom(u)

        # 添加双键氧原子（显式验证键索引）
        for _ in range(2):
            o = Chem.Atom('O')
            o.SetFormalCharge(-2)
            uranyl.AddAtom(o)
            bond_idx = uranyl.AddBond(0, uranyl.GetNumAtoms() - 1, Chem.BondType.DOUBLE)

            # 验证键索引有效性
            if bond_idx < 0 or bond_idx >= uranyl.GetNumBonds():
                raise RuntimeError(f"无效键索引: {bond_idx}")

            # 安全设置键属性
            try:
                bond = uranyl.GetBondWithIdx(bond_idx)
                bond.SetIsConjugated(True)
            except Exception as e:
                print(f"设置键属性失败: {str(e)}")
                print(f"当前键总数: {uranyl.GetNumBonds()}")
                print(Chem.MolToMolBlock(uranyl))
                raise

        # ========== 水分子配体处理 ==========
        for _ in range(4):
            try:
                # 创建配位氧原子（显式设置电荷）
                o = Chem.Atom('O')
                o.SetFormalCharge(0)  # 配位后电荷变为0
                o_idx = uranyl.AddAtom(o)

                # 添加配位键
                uranyl.AddBond(0, o_idx, Chem.BondType.DATIVE)

                # 添加两个氢原子
                h1 = Chem.Atom('H')
                h2 = Chem.Atom('H')
                uranyl.AddAtom(h1)
                uranyl.AddAtom(h2)
                uranyl.AddBond(o_idx, uranyl.GetNumAtoms() - 2, Chem.BondType.SINGLE)
                uranyl.AddBond(o_idx, uranyl.GetNumAtoms() - 1, Chem.BondType.SINGLE)

            except Exception as e:
                print(f"添加配体时发生错误: {str(e)}")

        # ========== 电荷平衡 ==========
        # 计算总电荷
        total_charge = sum(a.GetFormalCharge() for a in uranyl.GetAtoms())
        required_anions = abs(total_charge)
        # 添加硝酸根反离子
        for _ in range(required_anions):
            nitrate = Chem.MolFromSmiles('[O-][N+](=O)[O-]')
            uranyl.InsertMol(nitrate)
        # 二次电荷验证
        new_charge = sum(a.GetFormalCharge() for a in uranyl.GetAtoms())
        if new_charge != 0:
            raise ValueError(f"电荷平衡失败 (当前电荷: {new_charge})")

        # ========== 结构优化 ==========
        try:
            AllChem.SanitizeMol(uranyl)
            return uranyl.GetMol()
        except Exception as e:
            print(f"分子验证失败: {str(e)}")
            print("调试信息:")
            print(Chem.MolToMolBlock(uranyl))
            return None

    def validate_uranyl(self, mol):
        """铀酰结构验证"""
        if not mol:
            return ["无效分子"]

            # 检查铀原子
        u_atoms = [a for a in mol.GetAtoms() if a.GetSymbol() == 'U']
        if not u_atoms:
            return ["未找到铀原子"]
        u = u_atoms[0]

        # 检查配位环境
        issues = []
        bonds = u.GetBonds()

        # 双键氧检查
        double_bonds = [b for b in bonds
                        if b.GetBondType() == Chem.BondType.DOUBLE]
        if len(double_bonds) != 2:
            issues.append(f"需要2个双键氧，实际{len(double_bonds)}个")

        # 配位键检查
        dative_bonds = [b for b in bonds
                        if b.GetBondType() == Chem.BondType.DATIVE]
        if len(dative_bonds) != 4:
            issues.append(f"需要4个配位键，实际{len(dative_bonds)}个")

        # 电荷检查
        total_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
        if total_charge != 0:
            issues.append(f"总电荷不平衡 ({total_charge})")

        return issues if issues else ["验证通过"]


class AdvancedMolecularGenerator(InorganicGeneratorExtension):
    def __init__(self):
        super().__init__()
        self.fragment_library = self._build_fragment_library()
        self.reaction_rules = self._build_reaction_rules()
        self.alert_filter = self._create_chemical_filters()
        self.fragment_library.update(self.inorganic_fragments)

    def _build_fragment_library(self):
        """构建常见化学片段库"""
        return {
            'alkyl': ['[CH3]-', '-[CH2]-', '-[CH1]-'],
            'aryl': ['c1ccccc1', 'c1ccc(O)cc1'],
            'func_group': [
                '-O-', '-C(=O)O-', '-NH2', '-COOH',
                '-C#N', '-NO2', '-SO3H', '-COOR'
            ]
        }

    def _build_reaction_rules(self):
        """构建化学反应规则库"""
        return {
            'water': {
                'triggers': ['C(=O)OR', 'C(=O)Cl', 'CN'],
                'reaction': '水解反应',
                'conditions': '酸性/碱性条件'
            },
            'oxygen': {
                'triggers': ['C=C', 'C-H', 'OH'],
                'reaction': '氧化反应',
                'conditions': '加热/催化剂'
            },
            'metal': {
                'triggers': ['COOH', 'OH'],
                'reaction': '置换反应',
                'conditions': '金属钠/钾'
            }
        }

    def _create_chemical_filters(self):
        """创建化学合理性过滤器"""
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        return FilterCatalog(params)

    def generate_by_fragments(self, num_fragments=3):
        """基于片段的分子生成方法"""
        mol = Chem.RWMol()
        fragments = []

        # 选择初始片段
        frag_type = random.choice(list(self.fragment_library.keys()))
        frag_smarts = random.choice(self.fragment_library[frag_type])
        frag = Chem.MolFromSmarts(frag_smarts)
        mol.InsertMol(frag)

        # 逐步添加其他片段
        for _ in range(num_fragments - 1):
            try:
                # 选择连接点
                anchor = random.choice([atom for atom in mol.GetAtoms()
                                        if atom.GetDegree() < 2])

                # 选择新片段
                new_frag_type = random.choice(list(self.fragment_library.keys()))
                new_frag_smarts = random.choice(self.fragment_library[new_frag_type])
                new_frag = Chem.MolFromSmarts(new_frag_smarts)

                # 连接片段
                mol.InsertMol(new_frag)
                bond_order = random.choice([Chem.BondType.SINGLE,
                                            Chem.BondType.DOUBLE])
                mol.AddBond(anchor.GetIdx(),
                mol.GetNumAtoms() - 1,
                bond_order)

            except Exception:
                continue

        # 生成最终分子
        Chem.SanitizeMol(mol)
        return mol.GetMol()

    def validate_molecule(self, mol):
        """化学合理性验证"""
        validation = defaultdict(list)

        # 1. 结构合理性检查
        validation['structure'] = self._check_structure(mol)

        # 2. 化学稳定性检查
        validation['stability'] = self._check_stability(mol)

        # 3. 合成可行性检查
        validation['synthesizability'] = self._check_synthesizability(mol)

        return validation

    def _check_structure(self, mol):
        """结构合理性验证"""
        issues = []
        # 检查电荷平衡
        charge = Chem.GetFormalCharge(mol)
        if charge != 0:
            issues.append(f"电荷不平衡 ({charge})")

        # 检查原子价态
        for atom in mol.GetAtoms():
            if not atom.GetExplicitValence() == atom.GetImplicitValence():
                issues.append(f"原子 {atom.GetIdx()} 价态异常")

        return issues if issues else ["结构合理"]

    def _check_stability(self, mol):
        """化学稳定性检查"""
        alerts = []
        # 检查不稳定基团
        if self.alert_filter.HasMatch(mol):
            alerts.append("含有不稳定/反应性基团")

        # 检查环张力
        if rdMolDescriptors.CalcNumAmideBonds(mol) > 0:
            alerts.append("酰胺键存在水解风险")

        return alerts if alerts else ["稳定性良好"]

    def predict_reactions(self, mol):
        """预测分子反应性"""
        reactions = defaultdict(list)
        smi = Chem.MolToSmiles(mol)

        # 匹配反应规则
        for reagent, rule in self.reaction_rules.items():
            for pattern in rule['triggers']:
                if Chem.MolFromSmarts(pattern).HasSubstructMatch(mol):
                    reactions[reagent].append({
                        'type': rule['reaction'],
                        'conditions': rule['conditions'],
                        'probability': '高' if '*' in pattern else '中'
                    })

        # 特殊反应预测
        if Descriptors.MolLogP(mol) < -1:
            reactions['water'].append({'type': '易溶', 'conditions': '常温'})

        return dict(reactions)

    def generate_mixed_molecule(self):
        """生成有机-无机杂化分子"""
        # 50%概率生成纯有机/纯无机
        if random.random() < 0.5:
            return self.generate_by_fragments()
        else:
            # 生成金属配合物核心
            core = self.generate_metal_complex()
            # 添加有机配体
            return self._add_organic_ligands(core)

    def _add_organic_ligands(self, mol):
        """在金属核心上添加有机配体"""
        rw_mol = Chem.RWMol(mol)
        metal_idx = [atom.GetIdx() for atom in rw_mol.GetAtoms()
                     if atom.GetAtomicNum() > 20][0]

        # 添加羧酸配体示例
        cooh = Chem.MolFromSmiles('C(=O)O')
        rw_mol.InsertMol(cooh)
        rw_mol.AddBond(metal_idx, rw_mol.GetNumAtoms() - 2, Chem.BondType.DATIVE)

        Chem.SanitizeMol(rw_mol)
        return rw_mol.GetMol()


# 使用示例
if __name__ == "__main__":
    gen = AdvancedMolecularGenerator()

    # 生成铀酰配合物
    uranyl = gen.generate_metal_complex('U', ligand_type='oxide')
    # uranyl = gen.generate_metal_complex()
    print(Chem.MolToSmiles(uranyl))  # 例如O=[U]=O

    # 生成氙氟化合物
    xef = gen.generate_noble_gas_compound()
    print(Chem.MolToSmiles(xef))  # 例如F[Xe](F)(F)F

    # 生成有机-无机杂化分子
    hybrid = gen.generate_mixed_molecule()
    Draw.MolToImage(hybrid).show()

