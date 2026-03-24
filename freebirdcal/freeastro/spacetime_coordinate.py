import math


class SpacetimeCoordinateSystem:
    def __init__(self):
        """
        设置初始原点为 (0,0,0,0)，使用字典存储点的绝对坐标，`next_id` 用于生成唯一ID。
        """
        self.origin = (0.0, 0.0, 0.0, 0.0)  # (x, y, z, t) in absolute coordinates
        self.points = {}  # {id: (x_abs, y_abs, z_abs, t_abs)}
        self.next_id = 1

    def add_point(self, x_rel, y_rel, z_rel, t_rel):
        """
        将用户提供的相对坐标转换为绝对坐标后存储。
        """
        x_abs = self.origin[0] + x_rel
        y_abs = self.origin[1] + y_rel
        z_abs = self.origin[2] + z_rel
        t_abs = self.origin[3] + t_rel
        new_id = self.next_id
        self.points[new_id] = (x_abs, y_abs, z_abs, t_abs)
        self.next_id += 1
        return new_id

    def remove_point(self, point_id):
        """
        根据ID删除对应点。
        """
        if point_id in self.points:
            del self.points[point_id]

    def get_coordinate(self, point_id):
        """
        返回相对于当前原点的坐标。
        """
        abs_coords = self.points.get(point_id)
        if abs_coords is None:
            return None
        x_rel = abs_coords[0] - self.origin[0]
        y_rel = abs_coords[1] - self.origin[1]
        z_rel = abs_coords[2] - self.origin[2]
        t_rel = abs_coords[3] - self.origin[3]
        return (x_rel, y_rel, z_rel, t_rel)

    def change_origin(self, new_origin_id):
        """
        将指定点的绝对坐标设为新的原点。
        """
        new_origin_abs = self.points.get(new_origin_id)
        if new_origin_abs is not None:
            self.origin = new_origin_abs

    def calculate_space_distance(self, id1, id2):
        """
        使用绝对坐标计算三维欧氏距离。
        """
        p1 = self.points.get(id1)
        p2 = self.points.get(id2)
        if p1 is None or p2 is None:
            return None
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dz = p1[2] - p2[2]
        return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def calculate_time_distance(self, id1, id2):
        """
        使用绝对时间值的差的绝对值。
        """
        p1 = self.points.get(id1)
        p2 = self.points.get(id2)
        if p1 is None or p2 is None:
            return None
        return abs(p1[3] - p2[3])

    def move_point(self, point_id, vx, vy, vz, time):
        """
        根据速度和时间计算新绝对坐标，不影响原有点。
        """
        abs_coords = self.points.get(point_id)
        if abs_coords is None:
            return None
        x_new = abs_coords[0] + vx * time
        y_new = abs_coords[1] + vy * time
        z_new = abs_coords[2] + vz * time
        t_new = abs_coords[3] + time
        return (x_new, y_new, z_new, t_new)


def decompose_velocity(speed, azimuth, elevation):
    """
    根据速度大小、方位角和仰角计算三维速度分量，使用三角函数分解速度。
    """
    if speed == 0:
        return (0.0, 0.0, 0.0)
    cos_elev = math.cos(elevation)
    sin_elev = math.sin(elevation)
    cos_azim = math.cos(azimuth)
    sin_azim = math.sin(azimuth)
    vx = speed * cos_elev * cos_azim
    vy = speed * cos_elev * sin_azim
    vz = speed * sin_elev
    return (vx, vy, vz)


def compute_velocity_angles(vx, vy, vz):
    """
    根据三维速度分量计算速度大小、方位角和仰角，处理特殊情况如零速度或纯垂直运动。
    """
    speed_sq = vx ** 2 + vy ** 2 + vz ** 2
    if speed_sq == 0:
        return (0.0, 0.0, 0.0)
    speed = math.sqrt(speed_sq)

    if vx == 0 and vy == 0:
        azimuth = 0.0
    else:
        azimuth = math.atan2(vy, vx)

    xy_projection = math.sqrt(vx ** 2 + vy ** 2)
    if xy_projection == 0:
        elevation = math.copysign(math.pi / 2, vz)
    else:
        elevation = math.atan2(vz, xy_projection)

    return (speed, azimuth, elevation)


if __name__ == "__main__":
    # 初始化坐标系
    system = SpacetimeCoordinateSystem()

    # 添加两个坐标点（相对原点）
    point1 = system.add_point(x_rel=2, y_rel=3, z_rel=1, t_rel=0)
    point2 = system.add_point(x_rel=5, y_rel=7, z_rel=2, t_rel=4)

    # 计算空间距离（应输出 5.0）
    distance = system.calculate_space_distance(point1, point2)
    print(f"空间距离: {distance}")

    # 计算时间距离（应输出 4.0）
    time_diff = system.calculate_time_distance(point1, point2)
    print(f"时间距离: {time_diff}")

    # 分解速度为三维分量（30度方位角，45度仰角，速率10）
    vx, vy, vz = decompose_velocity(
        speed=10,
        azimuth=math.radians(30),
        elevation=math.radians(45)
    )
    print(f"速度分量: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}")

    # 模拟点1以速度(1,2,3)移动5单位时间
    new_coords = system.move_point(point1, vx=1, vy=2, vz=3, time=5)
    print(f"移动后坐标: {new_coords}")

    # 将点2设为新原点
    system.change_origin(new_origin_id=point2)

    # 获取点1在新原点下的相对坐标（原绝对坐标差）
    rel_coords = system.get_coordinate(point1)
    print(f"新原点下点1的坐标: {rel_coords}")

