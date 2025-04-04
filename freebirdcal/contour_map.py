import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import find_contours
import geopandas as gpd
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import os
import logging
from warnings import warn

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VirtualContourMapGenerator:
    """
    生成虚拟等高线地形图的类（带错误处理）

    参数：
    - width (int): 地图宽度（像素数）
    - height (int): 地图高度（像素数）
    - resolution (float): 每个像素的分辨率（单位：米）
    - elevation_range (tuple): 高程范围（最小值，最大值），单位：米
    - contour_interval (float): 等高线间距，单位：米
    - noise_scale (float): 噪声平滑系数（控制地形复杂度）
    """

    def __init__(
            self,
            width=500,
            height=500,
            resolution=30.0,
            elevation_range=(0, 1000),
            contour_interval=50,
            noise_scale=5.0
    ):
        try:
            # 参数验证
            if not (isinstance(width, int) and width > 0):
                raise ValueError("width必须是正整数")
            if not (isinstance(height, int) and height > 0):
                raise ValueError("height必须是正整数")
            if resolution <= 0:
                raise ValueError("resolution必须是正数")
            if elevation_range[0] >= elevation_range[1]:
                raise ValueError("高程范围最小值必须小于最大值")
            if contour_interval <= 0:
                raise ValueError("等高线间距必须是正数")
            if noise_scale <= 0:
                raise ValueError("噪声平滑系数必须是正数")

            self.width = width
            self.height = height
            self.resolution = resolution
            self.elevation_range = elevation_range
            self.contour_interval = contour_interval
            self.noise_scale = noise_scale
            self.elevation_grid = None
            self.gdf_contours = None

        except ValueError as e:
            logger.error(f"参数初始化失败: {str(e)}")
            raise

    def generate_elevation(self):
        """生成随机高程数据并平滑处理"""
        try:
            # 生成随机噪声
            noise = np.random.rand(self.height, self.width)

            # 高斯滤波
            self.elevation_grid = gaussian_filter(noise, sigma=self.noise_scale)

            # 防止全零或全相同值的情况
            if np.all(self.elevation_grid == self.elevation_grid[0, 0]):
                warn("生成的高程数据缺少变化，请调整噪声参数", UserWarning)
                self.elevation_grid = np.zeros_like(self.elevation_grid)
                return

            # 归一化到高程范围
            min_val = self.elevation_grid.min()
            max_val = self.elevation_grid.max()

            # 防止除以零
            if max_val == min_val:
                self.elevation_grid = np.full_like(
                    self.elevation_grid,
                    self.elevation_range[0]
                )
                warn("高程数据无变化，使用最低高程值", UserWarning)
            else:
                self.elevation_grid = (
                        (self.elevation_grid - min_val) / (max_val - min_val) *
                        (self.elevation_range[1] - self.elevation_range[0]) +
                        self.elevation_range[0]
                )

        except Exception as e:
            logger.error(f"高程数据生成失败: {str(e)}")
            raise

    def generate_contours(self):
        """从高程数据中提取等高线"""
        if self.elevation_grid is None:
            raise RuntimeError("请先调用 generate_elevation() 生成高程数据")

        try:
            levels = np.arange(
                self.elevation_range[0],
                self.elevation_range[1] + self.contour_interval,
                self.contour_interval
            )

            contours = []
            for level in levels:
                try:
                    raw_contours = find_contours(self.elevation_grid, level)
                except ValueError as e:
                    logger.warning(f"在层级 {level} 处提取等高线失败: {str(e)}")
                    continue

                for contour in raw_contours:
                    if len(contour) < 2:
                        continue

                    # 坐标转换（考虑地理坐标系原点在左下角）
                    coordinates = [
                        (x * self.resolution, (self.height - y) * self.resolution)
                        for y, x in contour
                    ]
                    try:
                        line = LineString(coordinates)
                        if not line.is_valid:
                            logger.warning(f"无效几何在层级 {level}，已跳过")
                            continue
                        contours.append({"elevation": level, "geometry": line})
                    except Exception as e:
                        logger.warning(f"几何创建失败: {str(e)}")
                        continue

            if not contours:
                raise RuntimeError("未生成任何等高线，请调整参数")

            self.gdf_contours = gpd.GeoDataFrame(
                contours,
                crs="EPSG:3857"
            )

        except Exception as e:
            logger.error(f"等高线生成失败: {str(e)}")
            raise

    def plot_contours(self, save_path=None):
        """绘制等高线地图"""
        if self.gdf_contours is None:
            raise RuntimeError("请先调用 generate_contours() 生成数据")

        try:
            fig, ax = plt.subplots(figsize=(10, 10))
            self.gdf_contours.plot(
                ax=ax,
                column="elevation",
                cmap="viridis",
                legend=True,
                linewidth=1.0
            )
            ax.set_title("Virtual Contour Map")
            ax.set_xlabel("X (meters)")
            ax.set_ylabel("Y (meters)")

            if save_path:
                try:
                    plt.savefig(save_path, dpi=300)
                    logger.info(f"地图已保存至 {save_path}")
                except Exception as e:
                    logger.error(f"保存图片失败: {str(e)}")
                    raise
            plt.show()

        except Exception as e:
            logger.error(f"绘图失败: {str(e)}")
            raise
        finally:
            plt.close()

    def save_to_shapefile(self, path):
        """保存等高线为Shapefile"""
        if self.gdf_contours is None:
            raise RuntimeError("请先调用 generate_contours() 生成数据")

        try:
            # 检查目录是否存在
            dir_path = os.path.dirname(path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logger.info(f"创建目录: {dir_path}")

            self.gdf_contours.to_file(path, driver="ESRI Shapefile")
            logger.info(f"Shapefile 已保存至 {path}")

        except PermissionError:
            logger.error("文件写入权限被拒绝，请检查路径")
            raise
        except Exception as e:
            logger.error(f"保存Shapefile失败: {str(e)}")
            raise


# 示例用法
if __name__ == "__main__":
    try:
        generator = VirtualContourMapGenerator(
            width=300,
            height=300,
            resolution=30.0,
            elevation_range=(100, 1500),
            contour_interval=100,
            noise_scale=4.0
        )

        generator.generate_elevation()
        generator.generate_contours()
        generator.plot_contours(save_path="contour_map.png")
        generator.save_to_shapefile("output/contour_map.shp")

    except Exception as e:
        logger.critical(f"程序运行失败: {str(e)}")

