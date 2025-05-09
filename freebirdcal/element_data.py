# 原子量字典
atomic_weights = {
    'H':  1.008,  'He': 4.0026, 'Li': 6.94,   'Be': 9.0122,
    'B':  10.81,  'C':  12.011, 'N':  14.007, 'O':  16.00,
    'F':  19.00,  'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305,
    'Al': 26.982, 'Si': 28.085, 'P':  30.974, 'S':  32.06,
    'Cl': 35.45,  'Ar': 39.948, 'K':  39.098, 'Ca': 40.078,
    'Sc': 44.956, 'Ti': 47.867, 'V':  50.942, 'Cr': 52.00,
    'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693,
    'Cu': 63.546, 'Zn': 65.38,  'Ga': 69.723, 'Ge': 72.630,
    'As': 74.922, 'Se': 78.971, 'Br': 79.904, 'Kr': 83.798,
    'Rb': 85.468, 'Sr': 87.62,  'Y':  88.906, 'Zr': 91.224,
    'Nb': 92.906, 'Mo': 95.95,  'Tc': 98,     'Ru': 101.07,
    'Rh': 102.91, 'Pd': 106.42, 'Ag': 107.87, 'Cd': 112.41,
    'In': 114.82, 'Sn': 118.71, 'Sb': 121.76, 'Te': 127.60,
    'I':  126.90, 'Xe': 131.29,
    'Cs': 132.91, 'Ba': 137.33, 'La': 138.91, 'Ce': 140.12,
    'Pr': 140.91, 'Nd': 144.24, 'Pm': 145,    'Sm': 150.36,
    'Eu': 151.96, 'Gd': 157.25, 'Tb': 158.93, 'Dy': 162.50,
    'Ho': 164.93, 'Er': 167.26, 'Tm': 168.93, 'Yb': 173.05,
    'Lu': 174.97, 'Hf': 178.49, 'Ta': 180.95, 'W':  183.84,
    'Re': 186.21, 'Os': 190.23, 'Ir': 192.22, 'Pt': 195.08,
    'Au': 196.97, 'Hg': 200.59, 'Tl': 204.38, 'Pb': 207.2,
    'Bi': 208.98, 'Po': 209,    'At': 210,    'Rn': 222,
    'Fr': 223,    'Ra': 226,    'Ac': 227,    'Th': 232.04,
    'Pa': 231.04, 'U':  238.03, 'Np': 237,    'Pu': 244,
    'Am': 243,    'Cm': 247,    'Bk': 247,    'Cf': 251,
    'Es': 252,    'Fm': 257,    'Md': 258,    'No': 259,
    'Lr': 266,    'Rf': 267,    'Db': 270,    'Sg': 271,
    'Bh': 270,    'Hs': 277,    'Mt': 278,    'Ds': 281,
    'Rg': 282,    'Cn': 285,    'Nh': 286,    'Fl': 289,
    'Mc': 289,    'Lv': 293,    'Ts': 294,    'Og': 294
}

# 化合价字典（列出常见氧化态）
oxidation_states = {
    # 第1周期
    'H':  [-1, +1],                # 氢（H⁻, H⁺）
    'He': [0],                     # 氦（仅单质）
    # 第2周期
    'Li': [+1],                    # 锂
    'Be': [+2],                    # 铍
    'B':  [+3],                    # 硼
    'C':  [-4, -3, -2, -1, 0, +2, +4],       # 碳（CH₄: -4, CO: +2, CO₂: +4）
    'N':  [-3, -2, -1, +1, +2, +3, +4, +5],  # 氮（NH₃: -3, NO₂: +4, HNO₃: +5）
    'O':  [-2, -1, 0, +1, +2],     # 氧（H₂O: -2, O₂: 0, OF₂: +2）
    'F':  [-1, 0],                 # 氟（F⁻: -1, F₂: 0）
    'Ne': [0],                     # 氖（仅单质）
    # 第3周期
    'Na': [+1],                    # 钠
    'Mg': [+2],                    # 镁
    'Al': [+3],                    # 铝
    'Si': [-4, +4],                # 硅（SiH₄: -4, SiO₂: +4）
    'P':  [-3, +3, +5],            # 磷（PH₃: -3, H₃PO₄: +5）
    'S':  [-2, +4, +6],            # 硫（H₂S: -2, SO₂: +4, H₂SO₄: +6）
    'Cl': [-1, +1, +3, +5, +7],    # 氯（HCl: -1, HClO: +1, HClO₄: +7）
    'Ar': [0],                     # 氩（仅单质）
    # 第4周期
    'K':  [+1],                    # 钾
    'Ca': [+2],                    # 钙
    'Sc': [+3],                    # 钪
    'Ti': [+2, +3, +4],            # 钛（TiO: +2, Ti₂O₃: +3, TiO₂: +4）
    'V':  [+2, +3, +4, +5],        # 钒（VO: +2, V₂O₅: +5）
    'Cr': [+2, +3, +6],            # 铬（CrO: +2, Cr₂O₃: +3, CrO₃: +6）
    'Mn': [+2, +3, +4, +6, +7],    # 锰（Mn²⁺, MnO₄⁻: +7）
    'Fe': [+2, +3],                # 铁（Fe²⁺, Fe³⁺）
    'Co': [+2, +3],                # 钴
    'Ni': [+2, +3],                # 镍
    'Cu': [+1, +2],                # 铜（Cu⁺, Cu²⁺）
    'Zn': [+2],                    # 锌
    'Ga': [+3],                    # 镓
    'Ge': [-4, +2, +4],            # 锗
    'As': [-3, +3, +5],            # 砷（AsH₃: -3, H₃AsO₄: +5）
    'Se': [-2, +4, +6],            # 硒（H₂Se: -2, SeO₃²⁻: +4）
    'Br': [-1, +1, +3, +5, +7],    # 溴（Br⁻: -1, HBrO₃: +5）
    'Kr': [0, +2],                 # 氪（KrF₂: +2）
    # 第5周期
    'Rb': [+1],                    # 铷
    'Sr': [+2],                    # 锶
    'Y':  [+3],                    # 钇
    'Zr': [+4],                    # 锆
    'Nb': [+3, +5],                # 铌
    'Mo': [+2, +3, +4, +5, +6],    # 钼（MoO₃: +6）
    'Tc': [+4, +6, +7],            # 锝（TcO₄⁻: +7）
    'Ru': [+2, +3, +4, +6, +8],    # 钌（RuO₄: +8）
    'Rh': [+3],                    # 铑
    'Pd': [+2, +4],                # 钯
    'Ag': [+1],                    # 银（Ag⁺）
    'Cd': [+2],                    # 镉
    'In': [+3],                    # 铟
    'Sn': [+2, +4],                # 锡（Sn²⁺, Sn⁴⁺）
    'Sb': [-3, +3, +5],            # 锑（SbH₃: -3, Sb₂O₅: +5）
    'Te': [-2, +4, +6],            # 碲（H₂Te: -2, TeO₃: +4）
    'I':  [-1, +1, +5, +7],        # 碘（I⁻: -1, HIO₃: +5）
    'Xe': [0, +2, +4, +6],         # 氙（XeF₂: +2, XeO₃: +6）
    # 第6周期（含镧系）
    'Cs': [+1],                   # 铯
    'Ba': [+2],                   # 钡
    'La': [+3],                   # 镧
    'Ce': [+3, +4],               # 铈（Ce³⁺, CeO₂: +4）
    'Pr': [+3],                   # 镨
    'Nd': [+3],                   # 钕
    'Pm': [+3],                   # 钷
    'Sm': [+2, +3],               # 钐
    'Eu': [+2, +3],               # 铕
    'Gd': [+3],                   # 钆
    'Tb': [+3, +4],               # 铽
    'Dy': [+3],                   # 镝
    'Ho': [+3],                   # 钬
    'Er': [+3],                   # 铒
    'Tm': [+3],                   # 铥
    'Yb': [+2, +3],               # 镱
    'Lu': [+3],                   # 镥
    'Hf': [+4],                   # 铪
    'Ta': [+5],                   # 钽
    'W':  [+4, +6],               # 钨（WO₃: +6）
    'Re': [+4, +6, +7],           # 铼（ReO₄⁻: +7）
    'Os': [+3, +4, +6, +8],       # 锇（OsO₄: +8）
    'Ir': [+3, +4],               # 铱
    'Pt': [+2, +4],               # 铂（Pt²⁺, PtCl₆²⁻: +4）
    'Au': [+1, +3],               # 金（Au⁺, Au³⁺）
    'Hg': [+1, +2],               # 汞（Hg₂²⁺: +1, Hg²⁺: +2）
    'Tl': [+1, +3],               # 铊（Tl⁺, Tl³⁺）
    'Pb': [+2, +4],               # 铅（Pb²⁺, PbO₂: +4）
    'Bi': [+3],                   # 铋（Bi³⁺）
    'Po': [-2, +2, +4],           # 钋（Po²⁺, PoO₃: +4）
    'At': [-1, +1, +5],           # 砹（At⁻, AtO₃⁻: +5）
    'Rn': [0, +2],                # 氡（RnF₂: +2）
    # 第7周期（含锕系及超重元素）
    'Fr': [+1],                   # 钫
    'Ra': [+2],                   # 镭
    'Ac': [+3],                   # 锕
    'Th': [+4],                   # 钍（Th⁴⁺）
    'Pa': [+4, +5],               # 镤
    'U':  [+3, +4, +5, +6],       # 铀（U³⁺, UO₂²⁺: +6）
    'Np': [+3, +4, +5, +6],       # 镎
    'Pu': [+3, +4, +5, +6],       # 钚
    'Am': [+3, +4, +5],           # 镅
    'Cm': [+3],                   # 锔
    'Bk': [+3, +4],               # 锫
    'Cf': [+3],                   # 锎
    'Es': [+3],                   # 锿
    'Fm': [+3],                   # 镄
    'Md': [+3],                   # 钔
    'No': [+2, +3],               # 锘
    'Lr': [+3],                   # 铹
    'Rf': [+4],                   # 𬬻（理论预测）
    'Db': [+5],                   # 𬭊（理论预测）
    'Sg': [+6],                   # 𬭳（理论预测）
    'Bh': [+7],                   # 𬭛（理论预测）
    'Hs': [+8],                   # 𬭶（理论预测）
    'Mt': [+1, +3],               # 鿏（理论预测）
    'Ds': [0, +4],                # 𫟼（理论预测）
    'Rg': [+1, +3],               # 𬬭（理论预测）
    'Cn': [+2],                   # 鎶（理论预测）
    'Nh': [+1],                   # 鉨（理论预测）
    'Fl': [+2],                   # 𫓧（理论预测）
    'Mc': [+1],                   # 镆（理论预测）
    'Lv': [+2, +4],               # 𫟷（理论预测）
    'Ts': [-1, +1, +5],           # 鿬（理论预测）
    'Og': [0, +2, +4]             # 鿫（理论预测）
}

# 化合物
#     名称            化学式                化合价    形态    颜色      用途
U_EC = [
    ("二氧化铀",      "UO2",               [4],     "固体", "黑色",   ["核燃料"]),
    ("八氧化三铀",     "U3O8",              [4, 6], "固体", "墨绿色", ["铀矿石"]),
    ("三氧化铀",      "UO3",               [6],     "固体", "橙黄色", ["燃料制备"]),
    ("六氟化铀",      "UF6",               [6],     "晶体", "白色",   ["铀浓缩"]),
    ("四氟化铀",      "UF4",               [4],     "固体", "绿色",   ["中间产物"]),
    ("硝酸铀酰",      "UO2(NO3)2",         [6],     "晶体", "黄色",   ["分析试剂"]),
    ("硫酸铀酰",      "UO2SO4",            [6],     "晶体", "黄绿色", ["电镀"]),
    ("醋酸铀酰",      "UO2(CH3COO)2",      [6],     "晶体", "黄色",   ["显微镜染色"]),
    ("氯化铀酰",      "UO2Cl2",            [6],     "晶体", "黄色",   ["催化剂"]),
    ("碳酸铀酰",      "UO2CO3",            [6],     "粉末", "黄色",   ["地质研究"]),
    ("磷酸铀酰",      "UO2HPO4",           [6],     "晶体", "浅黄色", ["分析化学"]),
    ("砷酸铀酰",      "UO2AsO4",           [6],     "粉末", "黄色",   ["分析试剂"]),
    ("钼酸铀酰",      "UO2MoO4",           [6],     "固体", "黄色",   ["催化剂"]),
    ("钨酸铀酰",      "UO2WO4",            [6],     "粉末", "黄色",   ["陶瓷颜料"]),
    ("硒酸铀酰",      "UO2SeO4",           [6],     "晶体", "黄色",   ["半导体材料"]),
    ("溴化铀酰",      "UO2Br2",            [6],     "晶体", "红色",   ["化学合成"]),
    ("碘化铀酰",      "UO2I2",             [6],     "晶体", "棕黑色", ["辐射检测"]),
    ("草酸铀酰",      "UO2C2O4",           [6],     "晶体", "黄色",   ["核废料处理"]),
    ("硅酸铀酰",      "UO2SiO3",           [6],     "矿物", "黄色",   ["地质研究"]),
    ("过氧铀酸",      "UO4",               [6],     "固体", "淡黄色", ["氧化剂"]),
    ("铀酸",         "H2UO4",              [6],    "溶液", "黄色",   ["实验室试剂"]),
    ("氟化铀酰钾",    "K(UO2)F3",           [6],    "晶体", "黄色",   ["激光材料"]),
    ("六水合硝酸铀酰", "UO2(NO3)2·6H2O",    [6],     "晶体", "黄绿色", ["分析标准"]),
    ("三水合醋酸铀酰", "UO2(CH3COO)2·3H2O", [6],     "晶体", "黄色",   ["电镀"]),
    ("九水合硫酸铀酰", "UO2SO4·9H2O",       [6],     "晶体", "绿色",  ["荧光材料"]),
    ("重铀酸铵",      "(NH4)2U2O7",        [6],     "粉末", "黄色",  ["黄饼"]),
    ("铀酸钠",        "Na2UO4",            [6],    "固体", "黄色",   ["玻璃着色"]),
    ("铀酸钾",        "K2UO4",             [6],    "晶体", "橙黄色", ["陶瓷釉料"]),
    ("碳化铀",        "UC",                [4],    "陶瓷", "灰色",   ["高温燃料"]),
    ("氮化铀",        "UN",                [3],    "晶体", "棕色",   ["核燃料"]),
    ("硫化铀",        "US",                [2],    "固体", "黑色",   ["半导体"]),
    ("氢化铀",        "UH3",               [3],    "粉末", "黑色",   ["储氢材料"])
]

Cu_EC = [
    ("氧化亚铜",       "Cu2O",              [1], "晶体", "砖红色", ["玻璃着色", "船底涂料"]),
    ("氧化铜",         "CuO",              [2], "粉末", "黑色",   ["陶瓷釉料", "催化剂"]),
    ("过氧化铜",       "CuO2",             [2], "固体", "棕黑色",  ["氧化剂"]),
    ("硫化亚铜",       "Cu2S",             [1], "矿物", "灰黑色",  ["炼铜原料"]),
    ("硫化铜",         "CuS",              [2], "固体", "黑色",   ["半导体材料"]),
    ("硫酸铜",         "CuSO4",            [2], "粉末", "白色",   ["杀菌剂", "电镀"]),
    ("五水硫酸铜",      "CuSO4·5H2O",       [2], "晶体", "蓝色",   ["教学试剂", "游泳池消毒"]),
    ("碱式硫酸铜",      "Cu3(OH)4SO4",      [2], "固体", "浅绿色", ["木材防腐"]),
    ("硝酸铜",         "Cu(NO3)2",         [2], "晶体", "深蓝色",  ["陶瓷颜料", "氧化剂"]),
    ("三水硝酸铜",      "Cu(NO3)2·3H2O",    [2], "晶体", "蓝色",   ["催化剂制备"]),
    ("氯化亚铜",       "CuCl",              [1], "晶体", "白色",   ["有机合成催化剂"]),
    ("氯化铜",         "CuCl2",            [2], "晶体", "黄棕色",  ["媒染剂", "木材防腐"]),
    ("二水氯化铜",      "CuCl2·2H2O",       [2], "晶体", "蓝绿色",  ["电镀添加剂"]),
    ("氢氧化亚铜",      "CuOH",             [1], "沉淀", "黄色",   ["暂时性存在"]),
    ("氢氧化铜",       "Cu(OH)2",          [2], "沉淀", "蓝色",    ["杀菌剂", "颜料"]),
    ("碱式碳酸铜",     "Cu2(OH)2CO3",       [2], "固体", "绿色",   ["铜锈主要成分", "颜料"]),
    ("醋酸铜",        "Cu(CH3COO)2",       [2], "晶体", "深绿色",  ["杀虫剂", "催化剂"]),
    ("水合醋酸铜",     "Cu(CH3COO)2·H2O",   [2], "晶体", "蓝绿色",  ["纺织品媒染剂"]),
    ("草酸铜",         "CuC2O4",           [2], "粉末", "浅蓝色",  ["分析试剂"]),
    ("甲酸铜",         "Cu(HCOO)2",        [2], "晶体", "蓝色",    ["防腐剂"]),
    ("四氨合硫酸铜",    "[Cu(NH3)4]SO4",    [2], "晶体", "深蓝色",  ["铜氨纤维", "定性分析"]),
    ("六氰合铁酸铜",    "Cu2[Fe(CN)6]",     [1], "沉淀", "红棕色",  ["颜料(普鲁士蓝)"]),
    ("氰化亚铜",       "CuCN",             [1], "粉末", "白色",    ["电镀", "冶金"]),
    ("磷酸铜",        "Cu3(PO4)2",        [2], "固体", "浅蓝色",  ["阻燃剂"]),
    ("硅酸铜",        "CuSiO3",           [2], "矿物", "绿色",    ["陶瓷釉料"]),
    ("溴化铜",        "CuBr2",            [2], "晶体", "黑色",    ["有机合成"]),
    ("碘化亚铜",      "CuI",               [1], "粉末", "白色",    ["人工降雨", "光敏材料"]),
    ("氧化铜镍矿",    "(Cu,Ni)O",          [2], "矿物", "黑色",    ["镍铜冶炼"]),
    ("硫化铜铁",      "CuFeS2",            [2], "矿物", "金黄色",  ["黄铜矿主要成分"]),
    ("碱式氯化铜",    "Cu2(OH)3Cl",        [2], "固体", "绿色",    ["防腐涂料"]),
    ("雷酸铜",        "Cu(ONC)2",         [2], "晶体", "棕红色",   ["起爆药(已淘汰)"]),
    ("酞菁铜",        "C32H16CuN8",       [2], "晶体", "深蓝色",   ["有机半导体", "染料"]),
    ("孔雀石",        "CuCO3·Cu(OH)2",    [2], "矿物", "翠绿色",   ["装饰石材", "炼铜原料"]),
    ("蓝铜矿",        "2CuCO3·Cu(OH)2",   [2], "矿物", "深蓝色",   ["颜料原料"]),
    ("赤铜矿",        "Cu2O",             [1], "矿物", "红色",    ["炼铜原料"])
]

