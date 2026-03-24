#!/usr/bin/env python3
"""
高级地形生成器
支持构造抬升、河流侵蚀、山坡扩散、微观细节，并输出高度图(RAW)和纹理权重图(Splatmap)。
"""

import os
import tempfile
import warnings

import matplotlib.pyplot as plt
import numpy as np
from noise import pnoise2, snoise2
from PIL import Image
from scipy.spatial import cKDTree

# 检查 Landlab 可用性
try:
    from landlab import RasterModelGrid
    from landlab.components import (
        DepressionFinderAndRouter,
        FlowAccumulator,
        LinearDiffuser,
        StreamPowerEroder,
    )

    LANDLAB_AVAILABLE = True
except ImportError:
    LANDLAB_AVAILABLE = False
    warnings.warn("Landlab 未安装，将无法进行侵蚀模拟。请安装: pip install landlab")

# 检查 ElevationAnalyzer 可用性
try:
    from .elevation_analyzer import ElevationAnalyzer

    ELEVATION_ANALYZER_AVAILABLE = True
except ImportError:
    ELEVATION_ANALYZER_AVAILABLE = False
    warnings.warn(
        "ElevationAnalyzer 依赖未安装，将无法生成 Shapefile。请安装: pip install freebirdcal[geospatial]"
    )


class TerrainGenerator:
    """
    地形生成器类：模拟构造-侵蚀过程，生成高度图和纹理权重图。
    """

    def __init__(self, shape=(513, 513), dx=10.0, seed=42):
        """
        初始化参数
        :param shape: 地形网格尺寸 (nx, ny)，推荐 2^n+1
        :param dx: 网格间距（米）
        :param seed: 随机种子
        """
        self.shape = shape
        self.nx, self.ny = shape
        self.dx = dx
        self.seed = seed
        self.initial_z = None  # 初始地形
        self.uplift_field = None  # 构造抬升场
        self.eroded_z = None  # 侵蚀后地形
        self.final_z = None  # 最终地形（含微观细节）
        self.rivers_mask = None  # 河流掩膜
        self.splatmap = None  # 纹理权重图

    def generate_tectonic_field(self):
        """
        生成空间变化的构造抬升速率场（Voronoi + 噪声）
        """
        nx, ny = self.nx, self.ny
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)

        np.random.seed(self.seed)
        n_points = 12  # 构造板块数量
        points = np.random.rand(n_points, 2)
        tree = cKDTree(points)
        _, region_idx = tree.query(np.c_[X.ravel(), Y.ravel()])
        region_idx = region_idx.reshape((nx, ny))

        uplift_rates = np.random.uniform(0.0005, 0.005, n_points)
        uplift_field = uplift_rates[region_idx]

        # 叠加分形噪声使过渡更自然
        for i in range(nx):
            for j in range(ny):
                val = pnoise2(
                    i / nx * 2.0,
                    j / ny * 2.0,
                    octaves=2,
                    persistence=0.5,
                    base=self.seed,
                )
                uplift_field[i, j] += 0.0005 * (val + 1) / 2
        self.uplift_field = uplift_field
        return self.uplift_field

    def initial_topography(self, scale=300, octaves=7):
        """
        使用 Simplex 噪声生成初始地形（海拔范围 0~5000 米）
        """
        nx, ny = self.nx, self.ny
        z = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                x = i / nx * scale
                y = j / ny * scale
                val = 0.0
                amp = 1.0
                freq = 1.0
                for _ in range(octaves):
                    val += amp * snoise2(x * freq, y * freq, base=self.seed)
                    amp *= 0.5
                    freq *= 2.0
                z[i, j] = (val + 1) / 2  # 范围 0~1
        self.initial_z = z * 5000  # 映射到 0~5000 米
        return self.initial_z

    def run_erosion(self, total_time=300000, dt=100, output_interval=500):
        """
        运行 Landlab 侵蚀模拟（需要 Landlab 已安装）
        """
        if not LANDLAB_AVAILABLE:
            raise RuntimeError("Landlab 未安装，无法进行侵蚀模拟。")

        if self.initial_z is None:
            raise ValueError("请先调用 initial_topography() 生成初始地形。")
        if self.uplift_field is None:
            self.generate_tectonic_field()

        nx, ny = self.nx, self.ny
        dx = self.dx
        grid = RasterModelGrid((ny, nx), xy_spacing=dx)

        # 初始化高程场（注意 Landlab 的索引顺序）
        z_init = self.initial_z.T.flatten()
        grid.add_field("topographic__elevation", z_init, at="node")

        # 组件初始化
        fa = FlowAccumulator(grid, flow_director="D8")
        sp = StreamPowerEroder(grid, K_sp=0.0001, m_sp=0.5, n_sp=1.0)
        diffuser = LinearDiffuser(grid, linear_diffusivity=0.005)
        dep_finder = DepressionFinderAndRouter(grid)

        n_steps = int(total_time // dt)
        for step in range(n_steps):
            # 构造抬升（空间变化）
            uplift_array = self.uplift_field.T.flatten() * dt
            grid.at_node["topographic__elevation"] += uplift_array

            # 水文计算
            fa.run_one_step()
            if step % 500 == 0:
                dep_finder.map_depressions()

            # 河流侵蚀
            sp.run_one_step(dt)

            # 山坡扩散
            diffuser.run_one_step(dt)

            if step % output_interval == 0:
                print(
                    f"步骤 {step}/{n_steps}, 最大高程: {grid.at_node['topographic__elevation'].max():.2f} m"
                )

        self.eroded_z = grid.at_node["topographic__elevation"].reshape((nx, ny)).T
        return self.eroded_z

    def add_micro_details(self, amplitude=30, scale=80):
        """
        叠加微观细节（小尺度分形噪声）
        """
        if self.eroded_z is None:
            raise ValueError("请先运行 run_erosion() 生成侵蚀后地形。")

        z = self.eroded_z
        nx, ny = self.nx, self.ny
        micro = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                micro[i, j] = pnoise2(
                    i / scale,
                    j / scale,
                    octaves=4,
                    persistence=0.5,
                    lacunarity=2.0,
                    base=self.seed + 100,
                )
        micro = (micro - micro.min()) / (micro.max() - micro.min()) * 2 - 1
        z_detail = z + amplitude * micro
        self.final_z = np.clip(z_detail, 0, None)
        return self.final_z

    def extract_rivers(self, threshold_m2=500000):
        """
        使用 Landlab 的流量累积提取河流掩膜
        """
        if self.final_z is None:
            raise ValueError("请先完成地形生成（最终高程尚未生成）。")
        if not LANDLAB_AVAILABLE:
            warnings.warn("Landlab 未安装，无法精确提取河流，使用高程阈值替代。")
            # 备用：低洼区域作为河流（粗略）
            from scipy.ndimage import gaussian_filter, minimum_filter

            z_smooth = gaussian_filter(self.final_z, sigma=5.0)
            local_min = z_smooth == minimum_filter(z_smooth, size=10)
            self.rivers_mask = local_min
            return self.rivers_mask

        nx, ny = self.nx, self.ny
        dx = self.dx
        grid = RasterModelGrid((ny, nx), xy_spacing=dx)
        grid.add_field("topographic__elevation", self.final_z.T.flatten(), at="node")
        fa = FlowAccumulator(grid, flow_director="D8")
        fa.run_one_step()
        area = grid.at_node["drainage_area"].reshape((nx, ny)).T  # 单位 m²
        self.rivers_mask = area > threshold_m2
        return self.rivers_mask

    def compute_slope(self, z=None):
        """
        计算地形坡度（度）
        """
        if z is None:
            if self.final_z is None:
                raise ValueError("未提供地形数据且 final_z 为空。")
            z = self.final_z
        dzdx, dzdy = np.gradient(z, self.dx, self.dx)
        slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
        slope_deg = np.degrees(slope_rad)
        return slope_deg

    def generate_splatmap(self, rivers_mask=None):
        """
        根据高程、坡度、河流掩膜生成 splatmap（RGBA 四通道权重图）
        """
        if self.final_z is None:
            raise ValueError("最终地形未生成，无法创建 splatmap。")

        z = self.final_z
        if rivers_mask is None:
            rivers_mask = self.rivers_mask

        # 计算坡度
        slope = self.compute_slope(z)

        h, w = self.nx, self.ny
        splat = np.zeros((h, w, 4), dtype=np.float32)

        # 纹理分配规则（可修改）
        for i in range(h):
            for j in range(w):
                e = z[i, j]
                s = slope[i, j]

                if e < 1500 and s < 30:
                    splat[i, j, 0] = 1.0  # 草地
                elif 1500 <= e < 3500 and s < 45:
                    splat[i, j, 1] = 1.0  # 岩石
                elif e >= 3500:
                    splat[i, j, 2] = 1.0  # 雪地
                else:
                    splat[i, j, 1] = 1.0  # 默认岩石

                # 河流覆盖
                if rivers_mask is not None and rivers_mask[i, j]:
                    splat[i, j, :] = 0.0
                    splat[i, j, 3] = 1.0  # 河流/沙地

        # 平滑过渡
        try:
            from scipy.ndimage import gaussian_filter

            for ch in range(4):
                splat[:, :, ch] = gaussian_filter(splat[:, :, ch], sigma=2.0)
        except ImportError:
            warnings.warn("scipy.ndimage 未安装，跳过平滑处理。")

        # 归一化权重
        row_sums = splat.sum(axis=2, keepdims=True)
        splat = np.divide(
            splat, row_sums, out=np.zeros_like(splat), where=row_sums != 0
        )

        self.splatmap = splat
        return self.splatmap

    def save_raw(self, filename, z=None):
        """
        保存 16-bit RAW 高度图（Unity 导入用）
        """
        if z is None:
            if self.final_z is None:
                raise ValueError("未提供地形数据且 final_z 为空。")
            z = self.final_z
        z_norm = (z - z.min()) / (z.max() - z.min())
        z_uint16 = (z_norm * 65535).astype(np.uint16)
        z_uint16.tofile(filename)

    def save_splatmap_png(self, filename):
        """
        保存 splatmap 为 RGBA PNG 图像
        """
        if self.splatmap is None:
            raise ValueError("splatmap 尚未生成，请先调用 generate_splatmap()。")
        splat_uint8 = (self.splatmap * 255).astype(np.uint8)
        img = Image.fromarray(splat_uint8, mode="RGBA")
        img.save(filename)

    def save_to_shapefile(
        self,
        output_path,
        interval=10,
        crs="EPSG:4326",
        pixel_size=None,
        top_left_x=0,
        top_left_y=0,
        smooth_sigma=0,
        z=None,
    ):
        """
        将地形数据保存为等高线 Shapefile 文件

        利用 ElevationAnalyzer 模块将地形数据转换为 GeoTIFF 格式，
        然后生成等高线并保存为 Shapefile。

        :param output_path: 输出的 .shp 文件路径
        :param interval: 等高线间隔（米）
        :param crs: 坐标参考系统，默认为 WGS84 (EPSG:4326)
        :param pixel_size: 像素尺寸（米），默认为 self.dx
        :param top_left_x: 左上角 X 坐标（米）
        :param top_left_y: 左上角 Y 坐标（米）
        :param smooth_sigma: 等高线平滑系数（高斯平滑标准差）
        :param z: 地形数据数组，若为 None 则使用 self.final_z
        :raises ImportError: 当 ElevationAnalyzer 依赖不可用时抛出
        :raises ValueError: 当地形数据未生成时抛出
        """
        if not ELEVATION_ANALYZER_AVAILABLE:
            raise ImportError(
                "ElevationAnalyzer 依赖未安装，无法生成 Shapefile。"
                "请安装: pip install freebirdcal[geospatial]"
            )

        if z is None:
            if self.final_z is None:
                raise ValueError("地形数据未生成，请先运行地形生成流程。")
            z = self.final_z

        if pixel_size is None:
            pixel_size = self.dx

        # 创建临时目录和文件
        with tempfile.TemporaryDirectory() as temp_dir:
            # 保存为原始二进制文件
            raw_path = os.path.join(temp_dir, "terrain.raw")
            z.astype(np.float32).tofile(raw_path)

            # 创建临时 GeoTIFF 文件
            tif_path = os.path.join(temp_dir, "terrain.tif")

            # 使用 ElevationAnalyzer 的静态方法转换为 GeoTIFF
            ElevationAnalyzer.raw_to_geotiff(
                raw_path=raw_path,
                output_path=tif_path,
                width=self.shape[1],  # 注意：宽度是列数 (ny)
                height=self.shape[0],  # 高度是行数 (nx)
                dtype=np.float32,
                crs=crs,
                pixel_size=pixel_size,
                top_left_x=top_left_x,
                top_left_y=top_left_y,
            )

            # 使用 ElevationAnalyzer 生成等高线并保存为 Shapefile
            with ElevationAnalyzer(tif_path) as ea:
                ea.export_contours(
                    output_path=output_path,
                    interval=interval,
                    smooth_sigma=smooth_sigma,
                )

        print(f"Shapefile 已保存至: {output_path}")

    def visualize(self, save_plots=False, prefix="terrain"):
        """
        生成可视化图表（可选保存）
        """
        if self.initial_z is None:
            raise ValueError("初始地形未生成。")
        if self.final_z is None:
            raise ValueError("最终地形未生成。")

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        # 初始地形
        im1 = axes[0, 0].imshow(self.initial_z, cmap="terrain")
        axes[0, 0].set_title("Initial Topography")
        plt.colorbar(im1, ax=axes[0, 0], label="Elevation (m)")
        # 侵蚀后（如果有）
        if self.eroded_z is not None:
            im2 = axes[0, 1].imshow(self.eroded_z, cmap="terrain")
            axes[0, 1].set_title("After Erosion")
            plt.colorbar(im2, ax=axes[0, 1], label="Elevation (m)")
        # 最终地形
        im3 = axes[0, 2].imshow(self.final_z, cmap="terrain")
        axes[0, 2].set_title("Final (with details)")
        plt.colorbar(im3, ax=axes[0, 2], label="Elevation (m)")
        # 坡度图
        slope = self.compute_slope()
        im4 = axes[1, 0].imshow(slope, cmap="viridis", vmax=60)
        axes[1, 0].set_title("Slope (deg)")
        plt.colorbar(im4, ax=axes[1, 0])
        # 河流掩膜
        if self.rivers_mask is not None:
            im5 = axes[1, 1].imshow(
                self.rivers_mask, cmap="Blues", interpolation="nearest"
            )
            axes[1, 1].set_title("River Network")
            plt.colorbar(im5, ax=axes[1, 1])
        # Splatmap 简示（只显示RGB）
        if self.splatmap is not None:
            rgb = self.splatmap[:, :, :3]
            im6 = axes[1, 2].imshow(rgb)
            axes[1, 2].set_title("Splatmap (RGB)")
            plt.colorbar(im6, ax=axes[1, 2])

        plt.tight_layout()
        if save_plots:
            plt.savefig(f"{prefix}_visualization.png", dpi=150)
            plt.show()
        else:
            plt.show()

    def run_full_pipeline(
        self,
        total_time=300000,
        dt=100,
        micro_amplitude=30,
        micro_scale=80,
        river_threshold=500000,
        save_plots=True,
        out_prefix="terrain",
        save_shapefile=False,
        shapefile_interval=10,
        shapefile_crs="EPSG:4326",
        shapefile_pixel_size=None,
        shapefile_top_left_x=0,
        shapefile_top_left_y=0,
        shapefile_smooth_sigma=0,
    ):
        """
        执行完整的地形生成流程

        :param total_time: 总模拟时间（年）
        :param dt: 时间步长（年）
        :param micro_amplitude: 微观细节振幅（米）
        :param micro_scale: 微观细节尺度
        :param river_threshold: 河流阈值（平方米）
        :param save_plots: 是否保存可视化图片
        :param out_prefix: 输出文件名前缀
        :param save_shapefile: 是否保存为 Shapefile 格式
        :param shapefile_interval: 等高线间隔（米）
        :param shapefile_crs: 坐标参考系统
        :param shapefile_pixel_size: 像素尺寸（米），默认为 self.dx
        :param shapefile_top_left_x: 左上角 X 坐标（米）
        :param shapefile_top_left_y: 左上角 Y 坐标（米）
        :param shapefile_smooth_sigma: 等高线平滑系数
        """
        print("=" * 60)
        print("高级地形生成器 - 构造抬升 + 河流侵蚀 + 微观细节 + Splatmap")
        print("=" * 60)

        # 1. 生成构造抬升场
        print("\n[1/6] 生成构造抬升场...")
        self.generate_tectonic_field()

        # 2. 生成初始地形
        print("[2/6] 生成初始地形（Simplex 噪声）...")
        self.initial_topography()

        # 3. 运行侵蚀模拟
        print("[3/6] 运行水力侵蚀模拟（可能需要几分钟）...")
        if LANDLAB_AVAILABLE:
            self.run_erosion(total_time=total_time, dt=dt, output_interval=50)
        else:
            warnings.warn("Landlab 不可用，跳过侵蚀模拟，直接使用初始地形。")
            self.eroded_z = self.initial_z.copy()

        # 4. 添加微观细节
        print("[4/6] 添加微观细节...")
        self.add_micro_details(amplitude=micro_amplitude, scale=micro_scale)

        # 5. 提取河流网络
        print("[5/6] 提取河流网络...")
        self.extract_rivers(threshold_m2=river_threshold)

        # 6. 生成 splatmap
        print("[6/6] 生成 splatmap 纹理权重图...")
        self.generate_splatmap()

        # 保存文件
        print("\n保存文件...")
        self.save_raw(f"{out_prefix}_heightmap.raw")
        self.save_splatmap_png(f"{out_prefix}_splatmap.png")
        print(f"  - 高度图 RAW: {out_prefix}_heightmap.raw")
        print(f"  - Splatmap PNG: {out_prefix}_splatmap.png")

        # 保存 Shapefile（如果启用）
        if save_shapefile:
            try:
                shapefile_output = f"{out_prefix}_contours.shp"
                if shapefile_pixel_size is None:
                    shapefile_pixel_size = self.dx

                print(f"  - Shapefile: {shapefile_output}")
                self.save_to_shapefile(
                    output_path=shapefile_output,
                    interval=shapefile_interval,
                    crs=shapefile_crs,
                    pixel_size=shapefile_pixel_size,
                    top_left_x=shapefile_top_left_x,
                    top_left_y=shapefile_top_left_y,
                    smooth_sigma=shapefile_smooth_sigma,
                )
            except ImportError as e:
                warnings.warn(f"无法保存 Shapefile: {e}")
            except Exception as e:
                warnings.warn(f"保存 Shapefile 时出错: {e}")

        # 可视化
        if save_plots:
            self.visualize(save_plots=True, prefix=out_prefix)

        print("\n" + "=" * 60)
        print("生成完成！")
        print("=" * 60)


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 创建生成器实例
    generator = TerrainGenerator(shape=(513, 513), dx=10.0, seed=42)

    # 运行完整流程
    generator.run_full_pipeline(
        total_time=100000,  # 总模拟时间（年）
        dt=10,  # 时间步长（年）
        micro_amplitude=5,  # 微观细节振幅（米）
        micro_scale=20,  # 微观细节尺度
        river_threshold=10000,  # 河流阈值（平方米）
        save_plots=True,  # 保存可视化图片
        out_prefix="terrain",  # 输出文件名前缀
    )
