import numpy as np
from scipy.optimize import newton
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.time import Time
from astropy import coordinates as coord
from astropy import units as u


class OrbitalDynamics:
    def __init__(self, mu=3.986004418e5, **kwargs):
        self.mu = mu  # 地球引力常数 (km^3/s^2)
        self.earth_radius = 6378.137  # 地球赤道半径 (km)
        self.omega_earth = 7.292115e-5  # 地球自转角速度 (rad/s)

        # 天体引力常数 (km^3/s^2)
        self.mu_sun = 1.32712440018e11
        self.mu_moon = 4.9048695e3

        # 大气参数
        self.rho0 = 1.225e-9  # 海平面密度 (kg/km^3)
        self.H0 = 8.5  # 大气标高 (km)

        # 初始化参数
        if 'a' in kwargs:
            self._init_from_elements(kwargs)
        elif 'r' in kwargs and 'v' in kwargs:
            self.r = np.array(kwargs['r'])
            self.v = np.array(kwargs['v'])
            self._state_to_elements()

        # 摄动配置
        self.perturbations = {
            'J2': kwargs.get('J2', False),
            'drag': kwargs.get('drag', False),
            'third_body': kwargs.get('third_body', False)
        }

        # 卫星物理参数
        self.Cd = kwargs.get('Cd', 2.2)  # 阻力系数
        self.A = kwargs.get('A', 10.0)  # 横截面积 (m²)
        self.mass = kwargs.get('mass', 1000)  # 质量 (kg)

        # 时间系统
        self.epoch = kwargs.get('epoch', '2023-01-01 00:00:00')

        self.thrust = np.zeros(3)  # 当前推力矢量 (N)
        self.isp = kwargs.get('isp', 300)  # 发动机比冲 (s)
        self.dry_mass = self.mass  # 干质量 (kg)
        self.propellant = kwargs.get('propellant', 500)  # 推进剂质量 (kg)
        self.maneuver_history = []  # 新增机动记录

    def _elements_to_state(self):
        """将轨道要素转换为位置速度向量（二体问题）"""
        if self.e < 1e-6:
            self.nu = 0  # 统一圆轨道初始位置

        # 计算轨道平面参数
        p = self.a * (1 - self.e ** 2)  # 半通径
        r = p / (1 + self.e * np.cos(self.nu))

        # 轨道平面坐标
        x = r * np.cos(self.nu)
        y = r * np.sin(self.nu)
        z = 0

        # 速度分量
        v_r = np.sqrt(self.mu / p) * self.e * np.sin(self.nu)
        v_n = np.sqrt(self.mu / p) * (1 + self.e * np.cos(self.nu))

        # 旋转到赤道坐标系
        R = self._rotation_matrix(self.raan, self.argp, self.i)
        self.r = R @ np.array([x, y, z])
        self.v = R @ np.array([v_r, v_n, 0])

    def _rotation_matrix(self, raan, argp, i):
        """生成旋转矩阵：轨道平面 -> 惯性系"""
        Rz_raan = np.array([
            [np.cos(raan), -np.sin(raan), 0],
            [np.sin(raan), np.cos(raan), 0],
            [0, 0, 1]
        ])

        Rx_i = np.array([
            [1, 0, 0],
            [0, np.cos(i), -np.sin(i)],
            [0, np.sin(i), np.cos(i)]
        ])

        Rz_argp = np.array([
            [np.cos(argp), -np.sin(argp), 0],
            [np.sin(argp), np.cos(argp), 0],
            [0, 0, 1]
        ])

        return Rz_raan @ Rx_i @ Rz_argp

    def _init_from_elements(self, kwargs):
        """通过轨道要素初始化"""
        self.a = kwargs['a']
        self.e = kwargs['e']
        self.i = np.radians(kwargs.get('i', 0))
        self.raan = np.radians(kwargs.get('raan', 0))
        self.argp = np.radians(kwargs.get('argp', 0))
        self.nu = np.radians(kwargs.get('nu', 0))
        self._elements_to_state()

    def _state_to_elements(self):
        """将状态向量转换为轨道要素"""
        r = self.r
        v = self.v
        h = np.cross(r, v)  # 角动量矢量
        n = np.cross([0, 0, 1], h)  # 节点矢量
        e_vec = ((np.linalg.norm(v) ** 2 - self.mu / np.linalg.norm(r)) * r - np.dot(r, v) * v)
        e_vec /= self.mu
        e = np.linalg.norm(e_vec)  # 偏心率

        # 能量和半长轴
        energy = np.linalg.norm(v) ** 2 / 2 - self.mu / np.linalg.norm(r)
        self.a = -self.mu / (2 * energy) if energy < 0 else 1e9

        # 轨道倾角
        self.i = np.arccos(h[2] / np.linalg.norm(h))

        # 升交点赤经
        if np.linalg.norm(n) > 1e-6:
            self.raan = np.arctan2(n[1], n[0])
        else:
            self.raan = 0.0  # 赤道轨道

        # 近地点幅角
        if e > 1e-6:
            self.argp = np.arctan2(np.dot(e_vec, np.cross(n, h)), np.dot(e_vec, n))
        else:
            self.argp = 0.0  # 圆轨道

        # 真近点角
        if np.linalg.norm(r) > 1e-6:
            self.nu = np.arctan2(np.dot(r, np.cross(h, e_vec)), np.dot(r, e_vec))

        self.e = e

    def _j2_perturbation(self, J2=1.0826e-3):
        """计算J2项摄动加速度"""
        r = np.linalg.norm(self.r)
        x, y, z = self.r
        k = 3 / 2 * J2 * self.mu * self.earth_radius ** 2 / r ** 5
        return np.array([
            k * x / r * (5 * z ** 2 / r ** 2 - 1),
            k * y / r * (5 * z ** 2 / r ** 2 - 1),
            k * z / r * (5 * z ** 2 / r ** 2 - 3)
        ])

    def _get_third_body_positions(self, t):
        """获取太阳和月球的位置（简化模型）"""
        # 使用astropy计算天体位置
        time = Time(self.epoch) + t * u.second
        sun = coord.get_sun(time)
        moon = coord.get_body('moon', time)

        # 转换为地球中心惯性系 (GCRS)
        sun_pos = sun.gcrs.cartesian.xyz.to(u.km).value
        moon_pos = moon.gcrs.cartesian.xyz.to(u.km).value

        return sun_pos, moon_pos

    def _third_body_perturbation(self, t):
        """第三体引力摄动"""
        sun_pos, moon_pos = self._get_third_body_positions(t)
        a_sun = np.zeros(3)
        a_moon = np.zeros(3)

        # 太阳摄动
        r_sun = sun_pos - self.r
        a_sun = self.mu_sun * (r_sun / np.linalg.norm(r_sun) ** 3 -
                               sun_pos / np.linalg.norm(sun_pos) ** 3)

        # 月球摄动
        r_moon = moon_pos - self.r
        a_moon = self.mu_moon * (r_moon / np.linalg.norm(r_moon) ** 3 -
                                 moon_pos / np.linalg.norm(moon_pos) ** 3)

        return a_sun + a_moon

    def _drag_perturbation(self):
        """大气阻力摄动"""
        # 计算高度
        altitude = np.linalg.norm(self.r) - self.earth_radius

        # 大气密度模型（指数模型）
        rho = self.rho0 * np.exp(-altitude / self.H0)

        # 转换到地固系计算相对速度
        omega = np.array([0, 0, self.omega_earth])
        v_rel = self.v - np.cross(omega, self.r)

        # 阻力加速度
        B = self.Cd * self.A / (2 * self.mass)  # 弹道系数
        a_drag = -0.5 * B * rho * np.linalg.norm(v_rel) * v_rel

        return a_drag

    def apply_impulse(self, dv, direction='body'):
        """
        应用脉冲机动 (瞬时速度变化)
        参数：
            dv : 速度变化量 (km/s)
            direction :
                'body' - 本体坐标系 (沿速度方向)
                'inertial' - 惯性系固定方向
        """
        if self.propellant <= 0:
            print("Error: 推进剂耗尽")
            return

        # 计算所需推进剂消耗
        g0 = 9.80665e-3  # km/s²
        delta_m = self.mass * (1 - np.exp(-dv / (self.isp * g0)))

        if delta_m > self.propellant:
            print("Warning: 推进剂不足，部分执行")
            delta_m = self.propellant

        # 更新质量和速度
        self.propellant -= delta_m
        self.mass = self.dry_mass + self.propellant

        # 计算推力方向
        if direction == 'body':
            # 沿当前速度方向
            v_dir = self.v / np.linalg.norm(self.v)
            self.v += dv * v_dir
        elif direction == 'inertial':
            # 固定惯性系方向
            self.v += np.array(dv)

        print(f"执行脉冲机动 Δv={dv * 1e3:.2f} m/s, 消耗推进剂 {delta_m:.2f} kg")

    def set_thrust(self, thrust_vector, duration=0):
        """
        设置有限推力机动
        参数：
            thrust_vector : 推力矢量 (N) [惯性系方向]
            duration : 持续时间 (s)，0表示持续到推进剂耗尽
        """
        self.thrust = np.array(thrust_vector)
        self.thrust_duration = duration
        self.thrust_start_time = None

    def _thrust_acceleration(self):
        """计算推力产生的加速度"""
        if np.linalg.norm(self.thrust) == 0:
            return np.zeros(3)

        # 推进剂质量流率
        g0 = 9.80665  # m/s²
        m_dot = np.linalg.norm(self.thrust) / (self.isp * g0)  # kg/s

        # 计算当前时间步消耗的推进剂
        dt = self.dt  # 从propagate方法传入的时间步长
        delta_m = m_dot * dt

        if delta_m > self.propellant:
            delta_m = self.propellant
            self.thrust = np.zeros(3)

        self.propellant -= delta_m
        self.mass = self.dry_mass + self.propellant

        # 计算加速度 (km/s²)
        a = self.thrust / self.mass * 1e-3  # 转换为 km/s²
        return a

    def hohmann_transfer(self, target_altitude):
        """
        计算霍曼转移所需Δv并自动执行
        参数：
            target_altitude : 目标圆轨道高度 (km)
        """
        # 当前轨道参数
        r1 = self.a  # 初始圆轨道半长轴
        r2 = self.earth_radius + target_altitude

        # 计算霍曼转移所需Δv
        dv1 = np.sqrt(self.mu / r1) * (np.sqrt(2 * r2 / (r1 + r2)) - 1)
        dv2 = np.sqrt(self.mu / r2) * (1 - np.sqrt(2 * r1 / (r1 + r2)))
        total_dv = dv1 + dv2

        print(f"霍曼转移需求Δv: {dv1 * 1e3:.2f} + {dv2 * 1e3:.2f} = {total_dv * 1e3:.2f} m/s")

        # 执行第一次机动（在近地点）
        self.apply_impulse(dv1, direction='body')

        # 等待转移到远地点
        # （实际需根据轨道周期计算转移时间）

        # 执行第二次机动
        self.apply_impulse(dv2, direction='body')

    def coplanar_adjustment(self, new_altitude, at_perigee=True):
        """
        共面轨道调整（改变圆轨道高度）
        参数：
            new_altitude : 新轨道高度 (km)
            at_perigee : 是否在近地点执行（True=效率最优，False=在当前位置执行）
        """
        r1 = np.linalg.norm(self.r)
        r2 = self.earth_radius + new_altitude

        if at_perigee:
            # 在近地点执行最优调整
            v1 = np.sqrt(self.mu * (2 / r1 - 1 / self.a))
            v2 = np.sqrt(self.mu / r2)
            dv = abs(v2 - v1)
        else:
            # 当前位置调整（效率较低）
            v_current = np.linalg.norm(self.v)
            v_needed = np.sqrt(2 * self.mu * (1 / r1 - 1 / (r1 + r2)))
            dv = abs(v_needed - v_current)

        self.apply_impulse(dv, direction='body')
        self.maneuver_history.append(('Coplanar', dv))

    def plane_change(self, new_i, new_raan):
        """
        轨道面变更（调整倾角/升交点赤经）
        参数：
            new_i : 新倾角 (deg)
            new_raan : 新升交点赤经 (deg)
        """
        v = np.linalg.norm(self.v)
        delta_i = np.radians(new_i) - self.i
        delta_raan = np.radians(new_raan) - self.raan

        # 计算所需Δv
        dv = 2 * v * np.sin(0.5 * np.sqrt(delta_i ** 2 + (np.sin(self.i) * delta_raan) ** 2))

        # 在轨道交点执行（假设当前在交点）
        self.apply_impulse(dv, direction='normal')
        self.maneuver_history.append(('Plane Change', dv))

    def bielliptic_transfer(self, intermediate_altitude):
        """
        双椭圆转移（适用于特定高度比的高效转移）
        参数：
            intermediate_altitude : 中间轨道远地点高度 (km)
        """
        r1 = self.a
        r2 = self.earth_radius + intermediate_altitude
        r3 = self.earth_radius  # 假设最终降低到低轨道（可根据需要修改）

        # 第一次加速
        dv1 = np.sqrt(2 * self.mu * (1 / r1 - 1 / (r1 + r2))) - np.sqrt(self.mu / r1)
        self.apply_impulse(dv1, direction='body')

        # 转移到远地点后的第二次加速
        # （需等待到达远地点，此处简化处理）
        dv2 = np.sqrt(2 * self.mu * (1 / r2 - 1 / (r2 + r3))) - np.sqrt(2 * self.mu * (1 / r2 - 1 / (r1 + r2)))
        self.apply_impulse(dv2, direction='body')

        # 第三次减速
        # （需等待到达目标点，此处简化处理）
        dv3 = np.sqrt(self.mu / r3) - np.sqrt(2 * self.mu * (1 / r3 - 1 / (r2 + r3)))
        self.apply_impulse(dv3, direction='body')

        self.maneuver_history.extend([('Bielliptic1', dv1), ('Bielliptic2', dv2), ('Bielliptic3', dv3)])

    def phase_adjustment(self, target_mean_anomaly, time_limit=None):
        """
        相位调整（改变轨道周期以调整相对位置）
        参数：
            target_mean_anomaly : 目标平近点角差 (deg)
            time_limit : 允许的调整时间 (s)
        """
        delta_theta = np.radians(target_mean_anomaly)
        current_T = 2 * np.pi * np.sqrt(self.a ** 3 / self.mu)

        if time_limit:
            delta_n = delta_theta / time_limit  # 需要改变的角速度
            delta_a = (self.mu / ((2 * np.pi / (current_T + delta_n)) ** 2)) ** (1 / 3) - self.a
        else:
            delta_T = delta_theta / (2 * np.pi) * current_T
            delta_a = delta_T / (3 * current_T) * self.a

        # 计算所需的半长轴变化
        dv = (self.mu / (2 * self.a)) * delta_a / self.a
        self.apply_impulse(dv, direction='body')
        self.maneuver_history.append(('Phase Adjust', dv))

    def sun_sync_maintenance(self):
        """太阳同步轨道保持（抵消J2摄动影响）"""
        # 计算轨道进动率
        n = np.sqrt(self.mu / self.a ** 3)
        precession_rate = (-3 * n * self.earth_radius ** 2 * self.J2 * (3 * np.cos(self.i) ** 2 - 1)) / (
                    4 * self.a ** 2 * (1 - self.e ** 2) ** 2)

        # 计算需要的倾角调整
        delta_i = precession_rate * 365.25 * 86400  # 年累积量
        self.plane_change(np.degrees(self.i) + delta_i, self.raan)

    def apply_maneuver_sequence(self, sequence):
        """
        执行机动序列
        参数：
            sequence : 机动指令列表，例如：
                [('hohmann', 35786), ('plane', 28, 15), ...]
        """
        for cmd in sequence:
            if cmd[0] == 'hohmann':
                self.hohmann_transfer(cmd[1])
            elif cmd[0] == 'plane':
                self.plane_change(cmd[1], cmd[2])
            # ...其他指令扩展...

    def _get_thrust_direction(self, frame):
        """获取推力方向参考系"""
        if frame == 'velocity':
            return self.v / np.linalg.norm(self.v)
        elif frame == 'radial':
            return self.r / np.linalg.norm(self.r)
        elif frame == 'normal':
            h = np.cross(self.r, self.v)
            return h / np.linalg.norm(h)

    def propagate(self, dt, steps, save_interval=10):
        """带摄动的轨道传播"""
        self.dt = dt / steps  # 记录时间步长
        t = np.linspace(0, dt, steps)
        self.states = []
        self.elements_history = []

        for i, ti in enumerate(t):
            # 基础二体加速度
            a = -self.mu * self.r / np.linalg.norm(self.r) ** 3

            # 推力加速度
            a_thrust = self._thrust_acceleration()
            a += a_thrust

            # 摄动加速度
            if self.perturbations['J2']:
                a += self._j2_perturbation()
            if self.perturbations['drag']:
                a += self._drag_perturbation()
            if self.perturbations['third_body']:
                a += self._third_body_perturbation(ti)

            # 改进积分方法（二阶龙格-库塔）
            k1_v = a * (dt / steps)
            k1_r = self.v * (dt / steps)

            v_temp = self.v + 0.5 * k1_v
            r_temp = self.r + 0.5 * k1_r

            a_temp = -self.mu * r_temp / np.linalg.norm(r_temp) ** 3
            # 此处应重新计算摄动，为简化暂时省略

            k2_v = a_temp * (dt / steps)
            k2_r = v_temp * (dt / steps)

            self.v += k2_v
            self.r += k2_r

            # 记录状态和轨道要素
            self.states.append(self.r.copy())
            if i % save_interval == 0:
                self._state_to_elements()
                self.elements_history.append({
                    'a': self.a,
                    'e': self.e,
                    'i': np.degrees(self.i),
                    'raan': np.degrees(self.raan),
                    'argp': np.degrees(self.argp),
                    'nu': np.degrees(self.nu)
                })

        return np.array(self.states)

    def plot_3d(self, states=None):
        """三维轨道可视化"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        if states is None and hasattr(self, 'states'):
            states = self.states

        x, y, z = [], [], []
        for each in states:
            x.append(each[0])
            y.append(each[1])
            z.append(each[2])

        ax.plot(x, y, z, label='Orbit')
        ax.scatter(0, 0, 0, c='r', marker='o', label='Earth')
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        plt.legend()
        plt.show()

    def plot_parameters(self):
        """绘制轨道参数随时间变化"""
        if not self.elements_history:
            print("No elements history available")
            return

        time = np.arange(len(self.elements_history)) * (self.dt / 24) * 1

        fig, axs = plt.subplots(3, 2, figsize=(15, 10))

        params = ['a', 'e', 'i', 'raan', 'argp', 'nu']
        units = ['km', '', 'deg', 'deg', 'deg', 'deg']

        for i, (param, unit) in enumerate(zip(params, units)):
            ax = axs[i // 2, i % 2]
            values = [entry[param] for entry in self.elements_history]
            ax.plot(time, values)
            ax.set_title(f'{param.upper()} vs Time')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'{param} ({unit})')
            ax.grid(True)

        plt.tight_layout()
        plt.show()


# 示例用法
if __name__ == "__main__":
    # 初始化低地球轨道
    orb = OrbitalDynamics(
        a=6778, e=0.0, i=45, raan=30, argp=0, nu=0,
        mass=2000, propellant=500,
        isp=300, J2=True
    )

    # 执行机动序列
    sequence = [
        ('coplanar', 800),  # 提升到800km高度
        ('plane', 28, 15),  # 调整到倾角28°, RAAN 15°
        ('bielliptic', 10000),  # 双椭圆转移到10000km中间轨道
        ('phase', 90, 86400)  # 24小时内调整90°相位
    ]
    orb.apply_maneuver_sequence(sequence)

    # 可视化轨道演化
    states = orb.propagate(3600 * 24 * 7, steps=10000)
    orb.plot_3d()
    orb.plot_parameters()

    # 打印机动记录
    print("\n机动历史记录：")
    for name, dv in orb.maneuver_history:
        print(f"{name}: {dv * 1e3:.2f} m/s")

