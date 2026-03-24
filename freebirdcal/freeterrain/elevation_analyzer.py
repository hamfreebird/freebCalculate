import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib import contour
from rasterio import features
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.ndimage import gaussian_filter, sobel
from shapely.geometry import LineString, MultiLineString, Polygon


class ElevationAnalyzer:
    """
    高程数据分析类：读取栅格高程数据，生成等高线、坡度/坡向图，保存为 Shapefile 等。
    """

    def __init__(self, raster_path):
        """
        初始化，读取栅格数据。

        :param raster_path: 高程栅格文件路径（支持格式：GeoTIFF 等，由 rasterio 决定）
        """
        self.raster_path = raster_path
        self.src = rasterio.open(raster_path)
        self.data = self.src.read(1).astype(np.float32)  # 读取第一波段
        self.nodata = self.src.nodata
        if self.nodata is not None:
            # 将 NoData 值设为 NaN
            self.data[self.data == self.nodata] = np.nan
        self.transform = self.src.transform
        self.crs = self.src.crs
        self.width = self.src.width
        self.height = self.src.height
        self.bounds = self.src.bounds

    def close(self):
        """关闭栅格文件句柄"""
        self.src.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def generate_contours(self, interval=10, levels=None, smooth_sigma=0):
        """
        生成等高线，返回 GeoDataFrame。

        :param interval: 等高线间隔（当 levels 未指定时使用）
        :param levels: 指定等高线值列表，若提供则忽略 interval
        :param smooth_sigma: 高斯平滑标准差（0 表示不平滑），可减少锯齿
        :return: GeoDataFrame (geometry: LineString 或 MultiLineString)
        """
        if levels is None:
            # 计算有效高程范围（排除 NaN）
            valid = self.data[~np.isnan(self.data)]
            if len(valid) == 0:
                raise ValueError("栅格中没有有效数据")
            min_z = np.floor(np.nanmin(self.data))
            max_z = np.ceil(np.nanmax(self.data))
            levels = np.arange(min_z, max_z + interval, interval)

        # 可选平滑
        arr = self.data.copy()
        if smooth_sigma > 0:
            arr = gaussian_filter(arr, sigma=smooth_sigma, mode="constant", cval=np.nan)
            # 保持 NaN 区域
            arr[np.isnan(self.data)] = np.nan

        # 生成等高线路径
        # 创建 x, y 坐标网格（像素中心坐标）
        x = np.arange(self.width) * self.transform.a + self.transform.c
        y = np.arange(self.height) * self.transform.e + self.transform.f
        X, Y = np.meshgrid(x, y)

        # 使用 matplotlib 的 contour 函数获取路径
        fig, ax = plt.subplots()
        cs = ax.contour(X, Y, arr, levels=levels)
        plt.close(fig)

        # 将路径转换为 Shapely 几何对象
        geometries = []
        levels_used = []
        # cs.allsegs 是一个列表的列表，其中 allsegs[i] 是第 i 个层级的线段列表
        for i, level in enumerate(cs.levels):
            segments = cs.allsegs[i]
            if segments:
                for segment in segments:
                    if len(segment) > 1:
                        geom = LineString(segment)
                        if geom.is_valid and not geom.is_empty:
                            geometries.append(geom)
                            levels_used.append(level)

        # 构建 GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {"level": levels_used, "geometry": geometries}, crs=self.crs
        )
        return gdf

    def calculate_slope(self, unit="degree"):
        """
        计算坡度（单位：度或百分比）。

        :param unit: 'degree' 或 'percent'
        :return: 坡度数组（与原始数据同形状）
        """
        # 计算 x, y 方向的梯度（米/像素）
        dx, dy = np.gradient(self.data, self.transform.a, -self.transform.e)
        # 坡度 = arctan(sqrt(dx^2 + dy^2))
        slope = np.arctan(np.sqrt(dx**2 + dy**2))
        if unit == "degree":
            slope = np.degrees(slope)
        elif unit == "percent":
            slope = np.tan(slope) * 100
        else:
            raise ValueError("unit 必须是 'degree' 或 'percent'")
        # 将 NaN 区域设为 NaN
        slope[np.isnan(self.data)] = np.nan
        return slope

    def calculate_aspect(self):
        """
        计算坡向（度，0-360，从北顺时针）。

        :return: 坡向数组
        """
        dx, dy = np.gradient(self.data, self.transform.a, -self.transform.e)
        aspect = np.arctan2(dy, dx)  # 方位角（东为0，逆时针）
        # 转换为北为0，顺时针
        aspect = (np.pi / 2) - aspect
        aspect[aspect < 0] += 2 * np.pi
        aspect = np.degrees(aspect)
        aspect[np.isnan(self.data)] = np.nan
        return aspect

    def hillshade(self, azimuth=315, altitude=45, z_factor=1):
        """
        计算山体阴影。

        :param azimuth: 光源方位角（度，从北顺时针）
        :param altitude: 光源高度角（度，从地平线起算）
        :param z_factor: 垂直夸大因子（用于突出地形）
        :return: 山体阴影数组（0-255）
        """
        slope = self.calculate_slope(unit="degree")
        aspect = self.calculate_aspect()
        # 转换角度为弧度
        azimuth_rad = np.radians(360 - azimuth + 90)  # 调整至数学坐标系
        altitude_rad = np.radians(altitude)

        slope_rad = np.radians(slope)
        aspect_rad = np.radians(aspect)

        # 计算光照强度
        illum = np.cos(altitude_rad) * np.cos(slope_rad) + np.sin(
            altitude_rad
        ) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)
        illum = np.clip(illum, 0, 1) * 255
        illum[np.isnan(slope)] = np.nan
        return illum.astype(np.uint8)

    def save_shapefile(self, gdf, output_path):
        """
        将 GeoDataFrame 保存为 Shapefile。

        :param gdf: GeoDataFrame
        :param output_path: 输出 .shp 文件路径（自动创建目录）
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        gdf.to_file(output_path, driver="ESRI Shapefile")

    def export_contours(self, output_path, interval=10, levels=None, smooth_sigma=0):
        """
        生成等高线并直接保存为 Shapefile。

        :param output_path: 输出 .shp 文件路径
        :param interval: 等高线间隔
        :param levels: 自定义等高线值列表
        :param smooth_sigma: 平滑系数
        """
        gdf = self.generate_contours(
            interval=interval, levels=levels, smooth_sigma=smooth_sigma
        )
        self.save_shapefile(gdf, output_path)

    def plot_contours(
        self, interval=10, levels=None, smooth_sigma=0, ax=None, **kwargs
    ):
        """
        绘制等高线图。

        :param interval: 等高线间隔
        :param levels: 自定义等高线值列表
        :param smooth_sigma: 平滑系数
        :param ax: matplotlib 轴对象（若为 None 则新建）
        :param kwargs: 传递给 matplotlib.contour 的其他参数
        :return: matplotlib 轴对象
        """
        if ax is None:
            fig, ax = plt.subplots()

        # 生成等高线数据（不转换为矢量，直接绘图）
        arr = self.data.copy()
        if smooth_sigma > 0:
            arr = gaussian_filter(arr, sigma=smooth_sigma, mode="constant", cval=np.nan)
            arr[np.isnan(self.data)] = np.nan

        x = np.arange(self.width) * self.transform.a + self.transform.c
        y = np.arange(self.height) * self.transform.e + self.transform.f
        X, Y = np.meshgrid(x, y)

        if levels is None:
            valid = arr[~np.isnan(arr)]
            min_z = np.floor(np.nanmin(arr))
            max_z = np.ceil(np.nanmax(arr))
            levels = np.arange(min_z, max_z + interval, interval)

        cs = ax.contour(X, Y, arr, levels=levels, **kwargs)
        ax.set_aspect("equal")
        ax.set_title("Contour Map")
        plt.colorbar(cs, ax=ax, label="Elevation (m)")
        return ax

    def get_statistics(self):
        """
        获取高程统计信息。

        :return: 包含最小值、最大值、均值、标准差、有效像素数的字典
        """
        valid = self.data[~np.isnan(self.data)]
        if len(valid) == 0:
            return {
                "min": np.nan,
                "max": np.nan,
                "mean": np.nan,
                "std": np.nan,
                "count": 0,
            }
        return {
            "min": np.min(valid),
            "max": np.max(valid),
            "mean": np.mean(valid),
            "std": np.std(valid),
            "count": len(valid),
        }

    def reproject(self, dst_crs, resolution=None, resampling=Resampling.bilinear):
        """
        将数据重投影到新的坐标系（返回新的 ElevationAnalyzer 对象，不修改原对象）。

        :param dst_crs: 目标坐标系（如 'EPSG:4326' 或 rasterio.crs.CRS 对象）
        :param resolution: 目标分辨率（米或度，取决于坐标系），若不提供则自动计算
        :param resampling: 重采样方法
        :return: 新的 ElevationAnalyzer 实例
        """
        # 计算目标变换和尺寸
        transform, width, height = calculate_default_transform(
            self.src.crs,
            dst_crs,
            self.width,
            self.height,
            *self.bounds,
            resolution=resolution,
        )
        # 创建目标数组
        dst_data = np.zeros((1, height, width), dtype=np.float32)
        reproject(
            source=self.data[np.newaxis, :, :],
            destination=dst_data,
            src_transform=self.transform,
            src_crs=self.src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=resampling,
        )
        # 写入临时文件（或者直接在内存中构造，这里简单写临时文件）
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name
        with rasterio.open(
            tmp_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=dst_data.dtype,
            crs=dst_crs,
            transform=transform,
            nodata=self.nodata,
        ) as dst:
            dst.write(dst_data)
        # 返回新对象
        return ElevationAnalyzer(tmp_path)

    @staticmethod
    def raw_to_geotiff(
        raw_path,
        output_path,
        width,
        height,
        dtype,
        crs=None,
        transform=None,
        nodata=None,
        band=1,
        pixel_size=None,
        top_left_x=None,
        top_left_y=None,
    ):
        """
        将原始二进制高程文件转换为 GeoTIFF 格式。

        :param raw_path: 输入的原始二进制文件路径
        :param output_path: 输出的 GeoTIFF 文件路径
        :param width: 像素宽度（列数）
        :param height: 像素高度（行数）
        :param dtype: numpy 数据类型，例如 'float32', 'int16' 等
        :param crs: 坐标参考系（如 'EPSG:4326' 或 rasterio.crs.CRS 对象），可选
        :param transform: 仿射变换矩阵（rasterio.transform.Affine），若提供则忽略 pixel_size/top_left_x/y
        :param nodata: 无数据值，可选
        :param band: 输出波段数，默认 1
        :param pixel_size: 像素尺寸（米或度），当 transform 未提供时使用，需与 top_left_x/y 配合
        :param top_left_x: 左上角 x 坐标，当 transform 未提供时使用
        :param top_left_y: 左上角 y 坐标，当 transform 未提供时使用
        """
        # 检查参数一致性
        if transform is None:
            if pixel_size is None or top_left_x is None or top_left_y is None:
                raise ValueError(
                    "必须提供 transform 或者同时提供 pixel_size, top_left_x, top_left_y"
                )
            transform = from_origin(top_left_x, top_left_y, pixel_size, pixel_size)

        # 读取原始数据
        data = np.fromfile(raw_path, dtype=dtype).reshape((height, width))
        # 如果需要多波段，可扩展，这里保持单波段
        if band == 1:
            data = data[np.newaxis, ...]
        else:
            # 若需要多波段，假设 raw 文件按波段顺序排列（需调整）
            raise NotImplementedError("多波段转换暂未实现")

        # 创建输出目录
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # 写入 GeoTIFF
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=band,
            dtype=data.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(data)

    @staticmethod
    def geotiff_to_raw(geotiff_path, output_path, dtype=None):
        """
        将 GeoTIFF 导出为原始二进制文件（仅数据，无头信息）。

        :param geotiff_path: 输入的 GeoTIFF 文件路径
        :param output_path: 输出的原始二进制文件路径
        :param dtype: 输出数据类型，若为 None 则使用原始数据类型
        """
        with rasterio.open(geotiff_path) as src:
            data = src.read(1)  # 读取第一波段
            if dtype is not None:
                data = data.astype(dtype)
            # 写入二进制文件
            data.tofile(output_path)


# 使用示例
if __name__ == "__main__":
    # 请替换为实际的高程栅格文件路径
    demo_file = "dem.tif"
    if os.path.exists(demo_file):
        with ElevationAnalyzer(demo_file) as ea:
            # 统计信息
            stats = ea.get_statistics()
            print("Statistics:", stats)

            # 生成等高线并保存为 Shapefile
            ea.export_contours("output/contours.shp", interval=20)

            # 计算坡度
            slope = ea.calculate_slope()
            print("Slope computed, shape:", slope.shape)

            # 绘制等高线图
            ea.plot_contours(interval=20, colors="black")
            plt.show()
    else:
        print("示例文件不存在，请指定有效的高程栅格文件。")
