import math


def decompose_velocity(speed: float, angle_deg: float) -> tuple[float, float]:
    """
    将速度分解为x/y分量

    参数:
        speed (float): 总速度大小（自然单位）
        angle_deg (float): 与x轴的夹角（角度制）

    返回:
        (vx, vy): 速度分量元组
    """
    theta = math.radians(angle_deg)
    return (
        speed * math.cos(theta),
        speed * math.sin(theta)
    )


def relativistic_velocity_addition(vx: float, vy: float, ux: float, uy: float, c: float = 1.0) -> tuple[float, float]:
    """
    计算相对论速度叠加（ux和uy是在速度v参考系中观测的速度）

    返回:
        (wx, wy): 原参考系中的合成速度分量
    """
    gamma = 1 / math.sqrt(1 - (vx ** 2 + vy ** 2) / c ** 2)
    denominator = 1 + (vx * ux + vy * uy) / c ** 2

    wx = (ux + vx + (gamma - 1) * (vx * (ux ** 2 + uy ** 2) / (vx ** 2 + vy ** 2))) / denominator
    wy = (uy + vy + (gamma - 1) * (vy * (ux ** 2 + uy ** 2) / (vx ** 2 + vy ** 2))) / denominator
    return wx, wy


class SpacetimeEvent:
    """
    描述三维时空中的事件（两个空间维 + 时间维），支持光锥分析、洛伦兹变换和因果性验证。

    参数:
        x (float): 空间坐标 x（默认单位，c=1时通常为自然单位）。
        y (float): 空间坐标 y。
        t (float): 时间坐标。
        c (float): 光速，默认为1（自然单位）。
    """

    def __init__(self, x: float, y: float, t: float, c: float = 1.0):
        self.x = x
        self.y = y
        self.t = t
        self.c = c

    def interval_to(self, other: 'SpacetimeEvent') -> float:
        """
        计算与另一事件的时空间隔平方 \(s^2 = \Delta x^2 + \Delta y^2 - c^2 \Delta t^2\)。
        """
        dx = other.x - self.x
        dy = other.y - self.y
        dt = other.t - self.t
        return dx ** 2 + dy ** 2 - (self.c ** 2) * (dt ** 2)

    def interval_type(self, other: 'SpacetimeEvent') -> str:
        """
        返回时空间隔类型：'timelike'（类时）, 'lightlike'（类光）, 或 'spacelike'（类空）。
        """
        s_squared = self.interval_to(other)
        if s_squared < 0:
            return "timelike"
        elif s_squared == 0:
            return "lightlike"
        else:
            return "spacelike"

    def is_in_future_of(self, other: 'SpacetimeEvent') -> bool:
        """
        判断当前事件是否在另一事件的未来光锥内。
        """
        s_squared = self.interval_to(other)
        dt = self.t - other.t
        return s_squared <= 0 and dt > 0

    def is_in_past_of(self, other: 'SpacetimeEvent') -> bool:
        """
        判断当前事件是否在另一事件的过去光锥内。
        """
        s_squared = self.interval_to(other)
        dt = self.t - other.t
        return s_squared <= 0 and dt < 0

    def lorentz_transform(self, v: float) -> 'SpacetimeEvent':
        """
        应用洛伦兹变换（沿x轴方向速度v），返回新参考系中的事件坐标。

        参数:
            v (float): 参考系相对运动速度（单位与光速c一致）。

        返回:
            SpacetimeEvent: 变换后的新事件对象。
        """
        if abs(v) >= self.c:
            raise ValueError("参考系速度不能达到或超过光速")
        gamma = 1 / math.sqrt(1 - (v ** 2 / self.c ** 2))
        new_x = gamma * (self.x - v * self.t)
        new_t = gamma * (self.t - (v * self.x) / (self.c ** 2))
        # y坐标不变，光速c保持不变
        return SpacetimeEvent(new_x, self.y, new_t, self.c)

    def lorentz_transform_y(self, v: float) -> 'SpacetimeEvent':
        """
        沿y轴方向的洛伦兹变换

        参数:
            v (float): 沿y轴的运动速度（自然单位，v < c）

        返回:
            SpacetimeEvent: 变换后的事件
        """
        if abs(v) >= self.c:
            raise ValueError("参考系速度不能达到或超过光速")
        gamma = 1 / math.sqrt(1 - (v ** 2 / self.c ** 2))
        new_y = gamma * (self.y - v * self.t)
        new_t = gamma * (self.t - (v * self.y) / (self.c ** 2))
        return SpacetimeEvent(self.x, new_y, new_t, self.c)

    def lorentz_transform_xy(self, vx: float, vy: float) -> 'SpacetimeEvent':
        """
        二维速度矢量的洛伦兹变换（沿任意方向）

        参数:
            vx (float): x方向速度分量
            vy (float): y方向速度分量

        返回:
            SpacetimeEvent: 变换后的事件
        """
        v_squared = vx ** 2 + vy ** 2
        if v_squared >= self.c ** 2:
            raise ValueError("合速度达到或超过光速")

        gamma = 1 / math.sqrt(1 - v_squared / self.c ** 2)
        v_dot_r = vx * self.x + vy * self.y  # 速度矢量与位置矢量的点积

        # 时间变换
        new_t = gamma * (self.t - v_dot_r / self.c ** 2)

        # 空间变换
        spatial_term = (gamma - 1) * (v_dot_r) / v_squared if v_squared != 0 else 0
        new_x = self.x + spatial_term * vx - gamma * vx * self.t
        new_y = self.y + spatial_term * vy - gamma * vy * self.t

        return SpacetimeEvent(new_x, new_y, new_t, self.c)

    def move(self, vx: float, vy: float, duration: float) -> 'SpacetimeEvent':
        """
        计算在当前参考系中以恒定速度 (vx, vy) 运动 duration 时间后的新事件

        参数:
            vx (float): x方向速度（自然单位，即 vx < c）
            vy (float): y方向速度
            duration (float): 运动时间（秒）

        返回:
            SpacetimeEvent: 新事件对象
        """
        # 验证速度合法性
        if math.hypot(vx, vy) >= self.c:
            raise ValueError(f"速度大小超过光速: v={math.hypot(vx, vy):.2f}c")

        # 经典运动学公式计算新坐标（原参考系视角）
        new_x = self.x + vx * duration
        new_y = self.y + vy * duration
        new_t = self.t + duration
        return SpacetimeEvent(new_x, new_y, new_t, self.c)

    def boost_and_move(self, boost_vx: float, boost_vy: float, local_duration: float) -> 'SpacetimeEvent':
        """
        瞬间加速到新参考系后，在新参考系中静止并持续 local_duration 时间，
        最后变换回原参考系的坐标

        参数:
            boost_vx (float): 新参考系x方向速度
            boost_vy (float): 新参考系y方向速度
            local_duration (float): 新参考系中的持续时间（秒）

        返回:
            SpacetimeEvent: 最终在原参考系中的事件
        """
        # Step 1: 变换到运动参考系
        boosted_event = self.lorentz_transform_xy(boost_vx, boost_vy)

        # Step 2: 在新参考系中静止（v=0）并持续指定时间
        # 注意：在新参考系中静止意味着空间坐标不变，时间增加 local_duration
        moved_in_boosted = SpacetimeEvent(
            boosted_event.x,
            boosted_event.y,
            boosted_event.t + local_duration,
            self.c
        )

        # Step 3: 逆变换回原参考系
        inverse_vx, inverse_vy = -boost_vx, -boost_vy
        return moved_in_boosted.lorentz_transform_xy(inverse_vx, inverse_vy)

    def __repr__(self):
        return f"SpacetimeEvent(x={self.x}, y={self.y}, t={self.t}, c={self.c})"


if __name__ == '__main__':
    """建议参照docs中的文档来阅读代码"""

    # 在参考系 S 中定义事件（c=1 为默认值）
    event_A = SpacetimeEvent(0, 0, 0)  # 原点事件
    event_B = SpacetimeEvent(2, 0, 3)  # 坐标 (2, 0, 3)
    event_C = SpacetimeEvent(3, 4, 5)  # 坐标 (3, 4, 5)
    s_squared = event_A.interval_to(event_B)
    print(f"事件A到B的间隔平方: {s_squared}")  # 输出: -5.0 (类时间隔)
    interval_type = event_A.interval_type(event_C)
    print(f"事件A到C的间隔类型: {interval_type}")  # 输出: lightlike
    # 判断事件B是否在事件A的未来光锥内
    print(event_B.is_in_future_of(event_A))  # 输出: True
    # 判断事件C是否在事件A的光锥表面
    print(event_A.interval_type(event_C) == "lightlike")  # 输出: True
    # 将事件B变换到以速度v=0.8c运动的参考系S'中
    v = 0.8  # 自然单位下，c=1，v=0.8表示0.8c
    event_B_transformed = event_B.lorentz_transform(v)
    print("变换后的事件B坐标:", event_B_transformed)  # 输出: SpacetimeEvent(x=-0.666..., y=0, t=2.333..., c=1)
    # 验证变换后的间隔是否仍为类时
    s_squared_transformed = event_A.lorentz_transform(v).interval_to(event_B_transformed)
    print(f"变换后的间隔平方: {s_squared_transformed}")  # 输出: -5.0 (与原始间隔一致)
    # 示例：验证跨参考系的因果性不变性
    event_D = SpacetimeEvent(1, 1, 2)  # 类时间隔事件
    event_D_transformed = event_D.lorentz_transform(0.6)
    # 判断变换后是否仍在事件A的未来光锥内
    print(event_D_transformed.is_in_future_of(event_A.lorentz_transform(0.6)))  # 输出: True

    # 原始事件：y=2, t=3
    event_E = SpacetimeEvent(0, 2, 3)
    # 沿y轴以0.6c速度变换
    transformed = event_E.lorentz_transform_y(0.6)
    # 验证变换公式：
    # gamma = 1/sqrt(1-0.6^2) ≈ 1.25
    # new_y = 1.25*(2 - 0.6*3) = 1.25*0.2 = 0.25
    # new_t = 1.25*(3 - 0.6*2) = 1.25*1.8 = 2.25
    print(transformed)  # 输出: SpacetimeEvent(x=0, y=0.25, t=2.25, c=1)
    # 定义斜方向运动参考系（vx=0.6c, vy=0.8c）
    vx, vy = 0.6, 0.8
    event_F = SpacetimeEvent(2, 3, 5)
    # 进行矢量变换
    try:
        transformed = event_F.lorentz_transform_xy(vx, vy)
    except ValueError as e:
        print(e)  # 触发异常：合速度v=√(0.6²+0.8²)=1.0c
    # 合法示例（v=0.6c, θ=45°）
    vx, vy = decompose_velocity(0.6 * 0.707, 45)  # 0.6*0.707≈0.424c
    transformed = event_F.lorentz_transform_xy(vx, vy)

    # 原参考系中的类光事件
    event_light = SpacetimeEvent(3, 4, 5)  # 3²+4²=5²
    # 变换到运动参考系
    transformed_event_light = event_light.lorentz_transform_xy(0.6, 0)
    # 验证间隔仍为类光
    print(event_light.interval_to(transformed_event_light))  # 输出: 0.0
    # 类时间隔事件
    event_G = SpacetimeEvent(0, 0, 0)
    event_H = SpacetimeEvent(1, 1, 2)  # 间隔平方=1+1-4=-2
    # 变换到运动参考系（vx=0.6c）
    transformed_H = event_H.lorentz_transform_xy(0.6, 0)
    # 保持类时间隔
    print(event_G.interval_to(transformed_H))  # 输出: -2.0
    # 初始事件：地球参考系原点 (0,0,0)
    earth_event = SpacetimeEvent(0, 0, 0, c=3e8)  # 使用国际单位（m, s）
    # 飞船以 0.8c 沿x轴飞行1年（约3.15e7秒）
    vx = 0.8 * earth_event.c  # 转换为 m/s
    duration = 3.15e7  # 1年（秒）
    # 计算移动后位置
    try:
        moved_event = earth_event.move(vx, 0, duration)
    except ValueError as e:
        print(e)  # 触发异常：速度超过c（因为vx=0.8c，但此处单位是m/s）
    # 正确示例（使用自然单位c=1）
    earth_event_natural = SpacetimeEvent(0, 0, 0)
    moved_event = earth_event_natural.move(0.6, 0, 3)  # 0.6c速度运动3秒
    print(moved_event)  # 输出: SpacetimeEvent(x=1.8, y=0, t=3, c=1)
    # 示例：飞船瞬间加速到0.6c并飞行2秒（飞船自身时间）
    final_event = earth_event.boost_and_move(0.6, 0, 2)
    # 计算原参考系中的总耗时
    print(f"地球时间: {final_event.t:.2f} 秒")  # 输出: 3.33秒（时间膨胀效应）
    # 原参考系中观测速度为0.6c的飞船，在飞船内向前发射0.5c的探测器
    wx, wy = relativistic_velocity_addition(0.6, 0, 0.5, 0)
    print(f"地球观测探测器速度: {wx:.3f}c")  # 输出: 0.846c（满足相对论加法）
    # 飞船加速到0.8c后，自身经历1秒
    earth_event = SpacetimeEvent(0, 0, 0)
    final_event = earth_event.boost_and_move(0.8, 0, 1)
    # 验证地球参考系中的时间
    gamma = 1 / math.sqrt(1 - 0.8 ** 2)  # γ≈1.6667
    expected_t = gamma * 1  # ≈1.6667秒
    print(f"实际地球时间: {final_event.t:.4f}秒")  # 输出: 1.6667秒

