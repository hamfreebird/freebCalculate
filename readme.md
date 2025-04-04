# **freebCalculate库**
#### **develop by freebird**

---

## **依赖**
本库依赖以下第三方库：
```python
numpy astropy scipy matplotlib mpmath scikit-image geopandas shapely
opencv-contrib-python openvino
```

---

## **功能**

| 名称                      | 用途       | 依赖                                                    |
|-------------------------|----------|-------------------------------------------------------|
| astro_simulator.py      | 星图生成     | numpy astropy scipy                                   |
| spacetime_event.py      | 光锥计算     | 无                                                     |
| formula_cal.py          | 常用公式计算   | numpy scipy mpmath matplotlib                         |
| equation_solver.py      | 求解方程     | 无                                                     |
| number_operations.py    | 数字计算     | 无                                                     |
| spacetime_coordinate.py | 四维坐标系    | 无                                                     |
| contour_map.py          | 等高线地形图生成 | numpy scipy scikit-image geopandas shapely matplotlib |
| video_interpolator.py   | 视频插帧     | numpy opencv-contrib-python openvino                  |
| npc_manager.py          | 游戏NPC管理  | 无                                                     |

---

## **示例**
#### **astro_simulator.py 用于星图生成**
```python
from astro_simulator import AstronomicalSimulator

# 初始化模拟器
sim = AstronomicalSimulator(
    image_size=2048,  # 图像尺寸
    pixel_scale=0.2,  # 0.2角秒/像素
    zeropoint=25.0,   # 星等零点
    gain=2.0          # 相机增益
)

# 生成恒星参数
stars = sim.generate_stars(num_stars=500, min_mag=18, max_mag=24)

# 生成PSF核（Moffat分布）
psf = sim.generate_psf(fwhm=3.0, profile='moffat')

# 生成图像（包含背景噪声）
image = sim.generate_image(stars, psf, sky_brightness=21.0)

# 生成星表
catalog = sim.generate_catalog(stars)

# 保存为FITS
sim.save_to_fits(image, catalog, 'observation.fits')
```
#### **spacetime_event.py 用于光锥计算**
```python
from spacetime_event import SpacetimeEvent, relativistic_velocity_addition

# 光子沿x轴运动（c=1自然单位）
photon_start = SpacetimeEvent(0, 0, 0)
photon_end = photon_start.move(1, 0, 5)  # 以光速运动5秒
print(photon_end)  # 输出: SpacetimeEvent(x=5, y=0, t=5, c=1)
print(photon_start.interval_type(photon_end))  # 输出: lightlike（类光间隔）

# 飞船加速到0.8c后，自身经历1秒
earth_event = SpacetimeEvent(0, 0, 0)
final_event = earth_event.boost_and_move(0.8, 0, 1)
print(f"实际地球时间: {final_event.t:.4f}秒")  # 输出: 1.6667秒

# 原参考系速度0.9c，叠加0.9c同方向速度
wx, wy = relativistic_velocity_addition(0.9, 0, 0.9, 0)
print(f"合成速度: {wx:.5f}c")  # 输出: 0.99448c（仍小于c）
```

#### **formula_cal.py 用于常用公式计算**
```python
import numpy as np
import formula_cal as fc

# 计算地球逃逸速度 (使用NumPy数组支持批量计算)
earth_mass = 5.97237e24  # kg
earth_radius = 6.3781e6  # m
print(f"地球逃逸速度: {fc.escape_velocity(earth_mass, earth_radius):.2f} m/s")

# 计算长直导线磁场
I = 1.0  # 1A电流
dl = np.array([0, 0, 1e-3])  # 1mm导线段
B = fc.biot_savart(I, dl, [0.1, 0, 0])  # 10cm外观测点
print(f"磁场: {B} T (理论值: [0, 2e-6, 0] T)")

# 计算抛体运动轨迹
t = np.linspace(0, 3, 30)
x, y = fc.projectile_motion(50, 45, t)
print("抛射最高点:", np.max(y), "m")

# 几何光学示例
print("水中到空气临界角:", fc.snells_law(1.33, 1.0, 90))  # 应返回NaN（全反射）
```

#### **equation_solver.py 用于求解方程**
```python
from equation_solver import EquationSolver
solver = EquationSolver()

# 一元一次方程
print("一元一次方程解:", solver.solve_linear_1v(2, -4))  # 2x -4 = 0 → x=2

# 一元二次方程
print("一元二次方程解:", solver.solve_quadratic_1v(1, -3, 2))  # x²-3x+2=0 → (2,1)

# 二元一次方程组
print("二元一次方程组解:",
      solver.solve_linear_2v([[3, 2], [2, -1]], [7, 4]))  # 3x+2y=7, 2x-y=4 → (3,2)

# 三元一次方程组
print("三元一次方程组解:",
      solver.solve_linear_3v([[2, 1, 1], [1, 3, 2], [1, 0, 1]], [4, 5, 1]))  # 解为(1,1,1)

# 二元二次方程组（线性+二次）
print("二元二次方程组解:",
      solver.solve_quadratic_2v((1, 1, 3), (1, 1, 0, 0, 0, -9)))  # x+y=3, x²+y²=9 → (3,0)和(0,3)
```

#### **number_operations.py 用于数字计算**
```python
from number_operations import NumberOperations

# 分数和基本运算
num1 = NumberOperations("2/3 + 1/6")
print(num1.value)  # (0.8333333333333333+0j)
# 希腊字母支持
num2 = NumberOperations("π/2")
print(num2.value)  # (1.5707963267948966+0j)
# 复数运算
num3 = NumberOperations("sqrt(-4) + 3^2")  # 2j + 9
print(num3.power(0.5))  # 开平方计算结果
# 因数分解
num4 = NumberOperations("12")
print(num4.factorize())  # [2, 2, 3]
```

#### **spacetime_coordinate.py 用于四维坐标系**
```python
from spacetime_coordinate import SpacetimeCoordinateSystem

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

# 模拟点1以速度(1,2,3)移动5单位时间
new_coords = system.move_point(point1, vx=1, vy=2, vz=3, time=5)
print(f"移动后坐标: {new_coords}")
```

#### **contour_map.py 用于等高线地形图生成**
```python
from contour_map import VirtualContourMapGenerator

# 初始化生成器
generator = VirtualContourMapGenerator(
    width=300,
    height=300,
    resolution=30.0,
    elevation_range=(100, 1500),
    contour_interval=100,
    noise_scale=4.0
)

# 生成数据
generator.generate_elevation()
generator.generate_contours()

# 可视化并保存
generator.plot_contours(save_path="contour_map.png")
generator.save_to_shapefile("output/contour_map.shp")
```

#### **video_interpolator.py 用于视频插帧**
```python
from video_interpolator import VideoInterpolator

interpolator = VideoInterpolator(
        input_path='input.mp4',
        output_path='output.mp4',
        interp_factor=2,  # 每两帧之间插入2帧
        method='optical_flow',
        use_gpu=True
    )

interpolator.process()
print("视频插帧处理完成")
```

#### **npc_manager.py 用于游戏NPC管理**
```python
from npc_manager import BaseNPC

npc = BaseNPC(
    identifier=1002,
    name="XXX",
    nickname="xxx",
    age=28,
    position=Position(120, 45, 2023),
    faction="xxxx",
    image_path=None,
    quotes=[
        "xxxx,xxxxx!",
        "xxxxxxx~"
    ]
)

npc.move_to(130, 50)  # 空间移动
npc.time_travel(2025)  # 时间跳跃
print(npc.speak())  # 随机语录输出
print(npc.get_info())  # 查看完整信息
```

---

## **使用**
- 详细文档在docs目录里，注意formula_cal.py和number_operations.py没有对应文档。
- number_operations.py不成熟，计算复杂表达式时可能出错。
- molecular_generator.py用于生成化合物，但问题较多，不成熟，故不建议使用。

---

#### **freebird fly in the sky~**

