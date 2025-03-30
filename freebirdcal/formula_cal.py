import math
import numpy as np
from scipy import integrate
from scipy.optimize import newton
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpmath import mp
mp.dps = 50  # 设置50位精度


# 常数 (CODATA 2019)
CONSTANTS = {
    'G': 6.67430e-11,       # 引力常数 [m³ kg⁻¹ s⁻²]
    'c': 299792458.0,       # 光速 [m/s]
    'sigma': 5.670374419e-8, # 斯特藩-玻尔兹曼常数 [W m⁻² K⁻⁴]
    'M_sun': 1.98847e30,    # 太阳质量 [kg]
    'R_sun': 6.957e8,       # 太阳半径 [m]
    'H0': 70.0 * 1000 / 3.0856775814913673e22,  # 哈勃常数 [s⁻¹]
    'sigma_T': 6.652458732e-29,  # 汤姆逊散射截面 [m²]
    'm_p': 1.67262192369e-27,    # 质子质量 [kg]
    'pc_to_m': 3.0856775814913673e16,  # 1秒差距 → 米
    'ε0': 8.8541878128e-12,    # 真空介电常数 [F/m]
    'μ0': 4e-7 * np.pi,        # 真空磁导率 [H/m]
    'e': 1.602176634e-19,      # 元电荷 [C]
    'Z0': 376.730313668,       # 真空阻抗 [Ω]
    'ke': 8.9875517923e9,       # 库仑常数 1/(4πε0)
    'g': 9.80665,        # 重力加速度 [m/s²]
    'k_B': 1.380649e-23,  # 玻尔兹曼常数 [J/K]
    'h': 6.62607015e-34,     # 普朗克常数 [J·s]
    'e0': 8.854187817e-12,   # 真空介电常数 [F/m]
    'n_air': 1.000293,       # 空气折射率 (标准条件)
    'sigma_SB': 5.670374419e-8,  # 斯特藩-玻尔兹曼常数 [W/m²K⁴]
    'R_inf': 10973731.568160  # 里德伯常数 [m⁻¹]
}

# 常用流体属性 (国际单位制)
FLUID_PROPS = {
    'water': {'rho': 997, 'mu': 8.9e-4},    # 水 20°C
    'air': {'rho': 1.225, 'mu': 1.8e-5},    # 空气 15°C
    'oil': {'rho': 900, 'mu': 0.1}          # 典型润滑油
}


def newton_gravitation(M1, M2, r):
    """牛顿万有引力计算
    Args:
        M1, M2 (kg): 质量
        r (m): 距离
    Returns:
        F (N): 引力值
    """
    return CONSTANTS['G'] * M1 * M2 / np.square(r)


def kepler_third_law(a, M_total):
    """开普勒第三定律计算轨道周期
    Args:
        a (m): 轨道半长轴
        M_total (kg): 系统总质量
    Returns:
        P (sec): 轨道周期
    """
    return np.sqrt(4 * np.pi**2 * np.power(a, 3) / (CONSTANTS['G'] * M_total))


def escape_velocity(M, R):
    """计算天体逃逸速度
    Args:
        M (kg): 天体质量
        R (m): 天体半径
    Returns:
        v_esc (m/s): 逃逸速度
    """
    return np.sqrt(2 * CONSTANTS['G'] * M / R)


# ------------------------- 恒星物理 -------------------------
def stefan_boltzmann_luminosity(R, T_eff):
    """斯特藩-玻尔兹曼定律计算光度
    Args:
        R (m): 恒星半径
        T_eff (K): 有效温度
    Returns:
        L (W): 光度
    """
    return 4 * np.pi * np.square(R) * CONSTANTS['sigma'] * np.power(T_eff, 4)


def mass_luminosity_relation(M):
    """主序星质量-光度关系
    Args:
        M (kg): 恒星质量
    Returns:
        L (W): 光度
    """
    return 3.828e26 * np.power(M / CONSTANTS['M_sun'], 3.5)


# ------------------------- 宇宙学 -------------------------
def hubble_flow_velocity(distance):
    """哈勃定律计算退行速度
    Args:
        distance (m): 共动距离
    Returns:
        v (m/s): 退行速度
    """
    return CONSTANTS['H0'] * distance


def cosmological_redshift(wl_obs, wl_emit):
    """计算宇宙学红移
    Args:
        wl_obs (m): 观测波长
        wl_emit (m): 发射波长
    Returns:
        z: 红移值
    """
    return (wl_obs - wl_emit) / wl_emit


# ------------------------- 相对论天体物理 --------------------
def schwarzschild_radius(M):
    """计算史瓦西半径
    Args:
        M (kg): 天体质量
    Returns:
        R_s (m): 事件视界半径
    """
    return 2 * CONSTANTS['G'] * M / np.square(CONSTANTS['c'])


def eddington_luminosity(M):
    """计算爱丁顿光度
    Args:
        M (kg): 天体质量
    Returns:
        L_Edd (W): 爱丁顿光度
    """
    return (4 * np.pi * CONSTANTS['G'] * M * CONSTANTS['m_p'] * CONSTANTS['c']
            ) / CONSTANTS['sigma_T']


def schwarzschild_radius_high_precision(M):
    """
    超高精度引力波模拟
    """
    return 2 * mp.mpf(CONSTANTS['G']) * M / mp.power(mp.mpf(CONSTANTS['c']), 2)


# ------------------------- 单位转换工具 ---------------------
def solar_mass_to_kg(M_solar):
    """太阳质量转换为千克"""
    return M_solar * CONSTANTS['M_sun']


def parsec_to_meters(pc):
    """秒差距转换为米"""
    return pc * CONSTANTS['pc_to_m']


# ========================= 静电学 =========================
def coulomb_force(q1, q2, r1, r2):
    """计算两个点电荷之间的库仑力 (矢量形式)

    参数:
        q1, q2 (C): 电荷量
        r1, r2 (m): 位置坐标 (三维数组)

    返回:
        F (N): 力矢量 [Fx, Fy, Fz]
    """
    r_vec = np.array(r2) - np.array(r1)
    r = np.linalg.norm(r_vec)
    if r == 0:
        return np.zeros(3)
    unit_vec = r_vec / r
    F_mag = CONSTANTS['ke'] * q1 * q2 / r ** 2
    return F_mag * unit_vec


def electric_field(q, r_source, r_obs):
    """计算点电荷在观测点产生的电场

    参数:
        q (C): 源电荷
        r_source (m): 源电荷位置
        r_obs (m): 观测点位置

    返回:
        E (V/m): 电场矢量
    """
    r_vec = np.array(r_obs) - np.array(r_source)
    r = np.linalg.norm(r_vec)
    if r == 0:
        return np.zeros(3)
    return CONSTANTS['ke'] * q * r_vec / r ** 3


# ========================= 静磁学 =========================
def biot_savart(I, dl, r_obs):
    """毕奥-萨伐尔定律计算磁场

    参数:
        I (A): 电流
        dl (m): 电流元矢量
        r_obs (m): 观测点位置

    返回:
        dB (T): 磁场增量
    """
    r = np.linalg.norm(r_obs)
    if r == 0:
        return np.zeros(3)
    return (CONSTANTS['μ0'] / (4 * np.pi)) * I * np.cross(dl, r_obs) / r ** 3


def lorentz_force(q, E, v, B):
    """计算洛伦兹力

    参数:
        q (C): 电荷量
        E (V/m): 电场矢量
        v (m/s): 速度矢量
        B (T): 磁感应强度矢量

    返回:
        F (N): 总作用力
    """
    return q * (np.array(E) + np.cross(v, B))


# ========================= 电路理论 =======================
def ohms_law(V=None, I=None, R=None):
    """欧姆定律计算器 (自动解析缺失量)"""
    if sum(x is None for x in [V, I, R]) != 1:
        raise ValueError("必须且只能缺失一个参数")
    if V is None: return I * R
    if I is None: return V / R
    if R is None: return V / I


def kirchhoff_voltage(emfs, voltage_drops):
    """基尔霍夫电压定律验证"""
    return np.isclose(sum(emfs) - sum(voltage_drops), 0, atol=1e-9)


# ========================= 电磁波 =========================
def wave_impedance(epsilon_r=1, mu_r=1):
    """计算介质中的波阻抗"""
    return np.sqrt(CONSTANTS['μ0'] * mu_r / (CONSTANTS['ε0'] * epsilon_r))


def skin_depth(f, sigma, mu_r=1):
    """计算趋肤深度"""
    omega = 2 * np.pi * f
    return np.sqrt(2 / (CONSTANTS['μ0'] * mu_r * sigma * omega))


# ========================= 相对论变换 =====================
def lorentz_transform_fields(E, B, v):
    """电磁场洛伦兹变换 (沿x轴方向运动)"""
    gamma = 1 / np.sqrt(1 - (v ** 2 / CONSTANTS['c'] ** 2))
    E_prime = np.array([
        E[0],
        gamma * (E[1] - v * B[2]),
        gamma * (E[2] + v * B[1])
    ])
    B_prime = np.array([
        B[0],
        gamma * (B[1] + (v * E[2] / CONSTANTS['c'] ** 2)),
        gamma * (B[2] - (v * E[1] / CONSTANTS['c'] ** 2))
    ])
    return E_prime, B_prime


# ======================= 流体静力学 ========================
def hydrostatic_pressure(rho, h, g=9.81):
    """静水压力计算
    Args:
        rho (kg/m³): 流体密度
        h (m): 深度
        g (m/s²): 重力加速度
    Returns:
        P (Pa): 静水压力
    """
    return rho * g * h


# ======================= 流体运动学 ========================
def reynolds_number(rho, V, L, mu):
    """雷诺数计算
    Args:
        V (m/s): 特征流速
        L (m): 特征长度
        mu (Pa·s): 动力粘度
    Returns:
        Re: 无量纲数
    """
    return rho * V * L / mu


def mach_number(V, T, gamma=1.4, R=287):
    """马赫数计算 (理想气体)
    Args:
        T (K): 温度
        gamma: 比热比
        R (J/kg·K): 气体常数
    Returns:
        Ma: 马赫数
    """
    c = np.sqrt(gamma * R * T)  # 声速
    return V / c


# ======================= 流体动力学 ========================
def bernoulli_total_pressure(rho, V, P_static):
    """伯努利总压计算
    Args:
        V (m/s): 流速
        P_static (Pa): 静压
    Returns:
        P_total (Pa): 总压
    """
    return P_static + 0.5 * rho * V**2


def darcy_weisbach(f, L, D, V, rho):
    """达西-魏斯巴赫管流压降计算
    Args:
        f: 摩擦系数
        L (m): 管长
        D (m): 管径
        V (m/s): 平均流速
    Returns:
        delta_P (Pa): 压力损失
    """
    return f * (L/D) * 0.5 * rho * V**2


# ======================= 边界层理论 ========================
def blasius_solution(x, mu, rho, U_inf):
    """平板层流边界层Blasius解
    Args:
        x (m): 平板前缘距离
        U_inf (m/s): 自由流速度
    Returns:
        delta (m): 边界层厚度
        tau_w (Pa): 壁面剪切应力
    """
    Re_x = rho * U_inf * x / mu
    delta = 4.91 * x / np.sqrt(Re_x)
    tau_w = 0.332 * rho * U_inf**2 / np.sqrt(Re_x)
    return delta, tau_w


# ======================= 可压缩流 ========================
def isentropic_flow(A_ratio, Ma, gamma=1.4):
    """等熵流面积-马赫数关系 (数值求解)
    Args:
        A_ratio: 面积比 A/A*
    Returns:
        Ma: 马赫数
    """
    func = lambda M: (1/M**2) * ((2/(gamma+1))*(1 + (gamma-1)/2*M**2))**((gamma+1)/(gamma-1)) - A_ratio**2
    return newton(func, 2.0)  # 初始猜测值2.0(超音速解)


# ======================= 多相流 ========================
def stokes_law(V, d, mu, rho_p, rho_f):
    """斯托克斯定律 (球体终端速度)
    Args:
        d (m): 颗粒直径
        rho_p (kg/m³): 颗粒密度
        rho_f (kg/m³): 流体密度
    Returns:
        V_terminal (m/s): 终端速度
    """
    g = 9.81
    return (d**2 * (rho_p - rho_f) * g) / (18 * mu)


# ======================= 湍流模型 ========================
def k_epsilon_model(k, epsilon, L, C_mu=0.09):
    """标准k-epsilon模型涡粘计算
    Args:
        k (m²/s²): 湍动能
        epsilon (m²/s³): 耗散率
        L (m): 特征长度
    Returns:
        mu_t (Pa·s): 湍流粘度
    """
    return C_mu * L * k**2 / epsilon


# ===================== 运动学 ========================
def linear_motion(v0, a, t):
    """匀变速直线运动计算
    Args:
        v0 (m/s): 初速度
        a (m/s²): 加速度
        t (s): 时间
    Returns:
        (位移, 末速度): (s, v)
    """
    s = v0 * t + 0.5 * a * t ** 2
    v = v0 + a * t
    return s, v


def projectile_motion(v0, theta, t, drag=0):
    """抛体运动轨迹计算(含空气阻力)
    Args:
        v0 (m/s): 初速度
        theta (deg): 抛射角
        t (s): 时间数组
        drag: 阻力系数 (0为无阻力)
    Returns:
        (x, y) 轨迹坐标数组
    """
    theta_rad = np.deg2rad(theta)
    vx = v0 * np.cos(theta_rad)
    vy = v0 * np.sin(theta_rad)

    def deriv(y, t):
        x, y, vx, vy = y
        dxdt = vx
        dydt = vy
        dvxdt = -drag * vx
        dvydt = -CONSTANTS['g'] - drag * vy
        return [dxdt, dydt, dvxdt, dvydt]

    sol = odeint(deriv, [0, 0, vx, vy], t)
    return sol[:, 0], sol[:, 1]


# ===================== 动力学 ========================
def newtons_second(mass, a=None, F=None):
    """牛顿第二定律计算器
    Args:
        输入两个参数，自动计算第三个
    Returns:
        缺失的物理量 (F, a 或 m)
    """
    inputs = [mass, a, F]
    if sum(x is None for x in inputs) != 1:
        raise ValueError("必须且只能有一个未知量")

    if F is None: return mass * a
    if a is None: return F / mass
    if mass is None: return F / a


def circular_motion_force(m, v, r):
    """匀速圆周运动向心力
    Args:
        v (m/s): 线速度
        r (m): 半径
    Returns:
        F (N): 向心力大小
    """
    return m * v ** 2 / r


# ===================== 能量与动量 =====================
def kinetic_energy(m, v):
    """动能计算 (支持相对论修正)
    Args:
        v (m/s): 速度标量值
    Returns:
        KE (J): 动能
    """
    gamma = 1 / np.sqrt(1 - (v ** 2 / CONSTANTS['c'] ** 2))
    return (gamma - 1) * m * CONSTANTS['c'] ** 2


def momentum_conservation(m1, v1, m2, v2, inelastic=False):
    """动量守恒碰撞计算
    Args:
        inelastic: 是否为完全非弹性碰撞
    Returns:
        碰撞后速度 (v1', v2') 或共同速度
    """
    p_total = m1 * v1 + m2 * v2
    if inelastic:
        v_final = p_total / (m1 + m2)
        return v_final
    else:
        # 假设弹性碰撞，需解方程组
        A = np.array([[m1, m2], [1, -1]])
        b = np.array([p_total, v1 - v2])
        return np.linalg.solve(A, b)


# ===================== 刚体力学 ======================
def moment_of_inertia(mass, geometry, **params):
    """常见刚体转动惯量计算
    geometry: 'sphere', 'cylinder', 'rod'等
    params: 几何参数如半径、长度等
    """
    if geometry == 'solid_sphere':
        r = params['r']
        return 0.4 * mass * r ** 2
    elif geometry == 'cylinder':
        r, h = params['r'], params['h']
        return 0.5 * mass * r ** 2
    elif geometry == 'rod_center':
        L = params['L']
        return (1 / 12) * mass * L ** 2
    else:
        raise ValueError("不支持的几何形状")


def torque(r, F, theta):
    """力矩计算
    Args:
        r (m): 位矢
        F (N): 力矢量
        theta (deg): 力与位矢夹角
    Returns:
        τ (N·m): 力矩大小
    """
    return np.linalg.norm(r) * np.linalg.norm(F) * np.sin(np.deg2rad(theta))


# ===================== 振动与波 ======================
def simple_harmonic(t, A, omega, phi=0):
    """简谐运动位移计算
    Args:
        A (m): 振幅
        omega (rad/s): 角频率
        phi (rad): 初相位
    Returns:
        x(t): 位移函数
    """
    return A * np.cos(omega * t + phi)


def pendulum_period(L, theta_max=5):
    """单摆周期计算 (小角度近似)
    Args:
        theta_max (deg): 最大摆角 (超过5度需用椭圆积分)
    """
    if theta_max > 5:
        raise NotImplementedError("大角度请使用数值解法")
    return 2 * np.pi * np.sqrt(L / CONSTANTS['g'])


# =================== 几何光学 ===================
def snells_law(n1, n2, theta1_deg):
    """斯涅尔定律计算折射角
    Args:
        theta1_deg: 入射角 (度数)
    Returns:
        theta2_deg: 折射角 (度数), 若全反射返回NaN
    """
    theta1 = np.deg2rad(theta1_deg)
    sin_theta2 = (n1 / n2) * np.sin(theta1)
    if np.abs(sin_theta2) > 1:
        return np.nan
    return np.rad2deg(np.arcsin(sin_theta2))


def lens_makers_formula(n, R1, R2):
    """透镜制造公式计算焦距
    Args:
        R1, R2: 曲率半径 (凸面为正)
    Returns:
        f: 焦距 [m]
    """
    return 1 / ((n - 1) * (1/R1 - 1/R2))


def thin_lens_equation(f, do):
    """薄透镜成像公式计算像距
    Args:
        do: 物距 [m]
    Returns:
        di: 像距 [m]
    """
    return 1 / (1/f - 1/do)


# =================== 波动光学 ===================
def young_interference_intensity(x, d, L, wavelength, I0=1):
    """杨氏双缝干涉强度分布
    Args:
        x: 观测屏位置数组 [m]
        d: 缝间距 [m]
        L: 缝到屏距离 [m]
    Returns:
        I: 光强分布
    """
    beta = np.pi * d * x / (wavelength * L)
    return 4 * I0 * (np.cos(beta))**2


def fraunhofer_diffraction(a, wavelength, theta):
    """夫琅禾费单缝衍射强度
    Args:
        a: 缝宽 [m]
        theta: 衍射角 [rad]
    Returns:
        相对强度
    """
    alpha = np.pi * a * np.sin(theta) / wavelength
    return (np.sin(alpha) / alpha)**2


def rayleigh_criterion(lambda_, D, n=1):
    """瑞利判据 (最小分辨角)
    Args:
        D: 光学孔径 [m]
        n: 介质折射率
    Returns:
        theta_min: 最小分辨角 [radians]
    """
    return 1.22 * lambda_ / (n * D)


# =================== 偏振光学 ===================
def brewsters_angle(n1, n2):
    """布鲁斯特角计算"""
    return np.rad2deg(np.arctan(n2 / n1))


def malus_law(I0, theta_deg):
    """马吕斯定律"""
    return I0 * (np.cos(np.deg2rad(theta_deg)))**2


# =================== 量子光学 ===================
def planck_law(lambda_, T):
    """普朗克黑体辐射公式
    Returns:
        Spectral radiance [W·sr⁻¹·m⁻³]
    """
    hc = CONSTANTS['h'] * CONSTANTS['c']
    k = 1.380649e-23  # 玻尔兹曼常数
    return (2 * np.pi * hc**2) / (lambda_**5 * (np.exp(hc/(lambda_*k*T)) - 1))


def compton_shift(theta_deg, lambda_initial):
    """康普顿散射波长偏移"""
    theta = np.deg2rad(theta_deg)
    delta_lambda = (CONSTANTS['h'] / (9.1093837015e-31 * CONSTANTS['c'])
                    ) * (1 - np.cos(theta))
    return lambda_initial + delta_lambda


def plot_interference(cls, d=1e-3, L=1, wavelength=500e-9):
    """绘制双缝干涉图样"""
    x = np.linspace(-0.01, 0.01, 1000)
    I = cls.young_interference_intensity(x, d, L, wavelength)
    plt.plot(x, I)
    plt.title("Young's Double-Slit Interference")
    plt.xlabel('Position (m)')
    plt.ylabel('Relative Intensity')


if __name__ == "__main__":
    # 计算地球逃逸速度 (使用NumPy数组支持批量计算)
    earth_mass = 5.97237e24  # kg
    earth_radius = 6.3781e6  # m
    print(f"地球逃逸速度: {escape_velocity(earth_mass, earth_radius):.2f} m/s")

    # 批量计算不同质量黑洞的史瓦西半径
    blackhole_masses = np.array([10, 1e6, 4e6]) * CONSTANTS['M_sun']
    sch_radii = schwarzschild_radius(blackhole_masses)
    print(f"史瓦西半径 (km): {sch_radii / 1e3}")

    # 计算不同距离的哈勃流速度
    distances = np.logspace(20, 25, 5)  # 10^20 到 10^25 米
    velocities = hubble_flow_velocity(distances)
    print(f"哈勃速度 (km/s): {velocities / 1e3}")

    # 计算不同质量恒星的寿命
    star_masses = np.array([0.5, 1.0, 20.0]) * CONSTANTS['M_sun']
    lifetimes = 1e10 * np.power(star_masses / CONSTANTS['M_sun'], -2.5)
    print(f"恒星寿命 (亿年): {lifetimes / 1e8}")

    # 计算两个1C电荷在1m距离的作用力
    F = coulomb_force(1, 1, [0, 0, 0], [1, 0, 0])
    print(f"库仑力: {F} N (理论值: [8.987e9, 0, 0] N)")

    # 计算长直导线磁场
    I = 1.0  # 1A电流
    dl = np.array([0, 0, 1e-3])  # 1mm导线段
    B = biot_savart(I, dl, [0.1, 0, 0])  # 10cm外观测点
    print(f"磁场: {B} T (理论值: [0, 2e-6, 0] T)")

    # 计算趋肤深度 (铜导线，1MHz)
    delta = skin_depth(1e6, 5.96e7)  # 铜电导率5.96e7 S/m
    print(f"趋肤深度: {delta * 1e3:.2f} mm")

    # 场强变换 (0.5c速度下的场变换)
    E = [0, 100, 0]  # 100 V/m y方向电场
    B = [0, 0, 1e-4]  # 0.1 mT z方向磁场
    E_new, B_new = lorentz_transform_fields(E, B, 0.5 * CONSTANTS['c'])
    print(f"变换后电场: {E_new} V/m")

    # 计算水管雷诺数
    water = FLUID_PROPS['water']
    Re = reynolds_number(water['rho'], 2.0, 0.1, water['mu'])
    print(f"水管雷诺数: {Re:.1e} ({'湍流' if Re > 4000 else '层流'})")

    # 计算飞机巡航马赫数
    Ma = mach_number(250, 216.65)  # 250m/s在-56.5°C高度
    print(f"巡航马赫数: {Ma:.2f}")

    # 计算油滴沉降速度
    oil_drop = stokes_law(None, 1e-4, FLUID_PROPS['oil']['mu'], 900, 1.225)
    print(f"油滴沉降速度: {oil_drop * 1e3:.2f} mm/s")

    # 计算收缩喷管马赫数 (A/A*=1.5)
    Ma = isentropic_flow(1.5, None)
    print(f"等熵流马赫数: {Ma:.2f}")

    # 计算5秒后自由落体位移
    print("自由落体5秒位移:", linear_motion(0, CONSTANTS['g'], 5)[0], "m")

    # 计算抛体运动轨迹
    t = np.linspace(0, 3, 30)
    x, y = projectile_motion(50, 45, t)
    print("抛射最高点:", np.max(y), "m")

    # 弹性碰撞后速度
    v1, v2 = momentum_conservation(2, 3, 4, -1)
    print(f"碰撞后速度: v1'={v1:.2f} m/s, v2'={v2:.2f} m/s")

    # 计算地球自转动能 (假设为实心球)
    earth_mass = 5.97e24  # kg
    earth_radius = 6.37e6  # m
    I = moment_of_inertia(earth_mass, 'solid_sphere', r=earth_radius)
    omega = 2 * np.pi / (24 * 3600)  # rad/s
    print(f"地球自转动能: {0.5 * I * omega ** 2:.2e} J")

    # 几何光学示例
    print("水中到空气临界角:", snells_law(1.33, 1.0, 90))  # 应返回NaN（全反射）

    # 波动光学示例
    x = np.linspace(-0.005, 0.005, 500)
    I = young_interference_intensity(x, 1e-3, 1, 500e-9)
    plt.figure()
    plt.plot(x, I)

    # 量子光学示例
    wavelengths = np.linspace(100e-9, 3000e-9, 200)
    T = 5778  # 太阳表面温度
    B = planck_law(wavelengths, T)
    plt.figure()
    plt.plot(wavelengths * 1e9, B)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Spectral Radiance')

    # 偏振示例
    print("玻璃(n=1.5)的布鲁斯特角:", brewsters_angle(1.0, 1.5))

