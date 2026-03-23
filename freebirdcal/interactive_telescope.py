"""
交互式天文望远镜模拟器

该模块提供交互式天文望远镜模拟功能，包括半球星图生成、望远镜光学效应模拟、
大气折射校正和交互式可视化。

主要特性：
1. 半球星图模拟：生成覆盖整个可见天空的模拟星图
2. 交互式控制：实时调整望远镜指向、视场、CMOS参数
3. 光学效应模拟：大气折射、镜头畸变、色差、彗差等
4. 实时可视化：使用matplotlib进行交互式显示
"""

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button, Slider

from .astro_simulator import AstronomicalSimulator

logger = logging.getLogger(__name__)

# 物理常数
R_EARTH = 6371.0  # 地球半径 (km)
REFRACTIVE_INDEX_SEA_LEVEL = 1.00029  # 海平面空气折射率


@dataclass
class TelescopeParameters:
    """望远镜参数容器类"""

    # 基本参数
    aperture_diameter: float = 0.1  # 口径直径 (m)
    focal_length: float = 1.0  # 焦距 (m)
    obstruction_ratio: float = 0.3  # 中心遮挡比

    # CMOS/CCD参数
    sensor_width: int = 1024  # 传感器宽度 (像素)
    sensor_height: int = 1024  # 传感器高度 (像素)
    pixel_size: float = 5.6e-6  # 像素尺寸 (m)
    read_noise: float = 5.0  # 读出噪声 (e-)
    gain: float = 2.0  # 相机增益 (e-/ADU)
    dark_current: float = 0.1  # 暗电流 (e-/像素/秒)
    quantum_efficiency: float = 0.8  # 量子效率

    # 光学参数
    distortion_k1: float = -0.1  # 径向畸变系数 k1
    distortion_k2: float = 0.01  # 径向畸变系数 k2
    chromatic_aberration: float = 0.02  # 色差系数
    coma_coefficient: float = 0.05  # 彗差系数
    astigmatism_coefficient: float = 0.03  # 像散系数

    # 大气参数
    seeing_fwhm: float = 2.0  # 视宁度 FWHM (角秒)
    atmospheric_extinction: float = 0.2  # 大气消光 (星等/空气质量)
    refractive_index: float = REFRACTIVE_INDEX_SEA_LEVEL  # 空气折射率

    # 观测参数
    exposure_time: float = 30.0  # 曝光时间 (秒)
    filter_band: str = "V"  # 滤光片波段

    def __post_init__(self):
        """参数验证"""
        if self.aperture_diameter <= 0:
            raise ValueError("Aperture diameter must be positive")
        if self.focal_length <= 0:
            raise ValueError("Focal length must be positive")
        if not 0 <= self.obstruction_ratio < 1:
            raise ValueError("Obstruction ratio must be in [0, 1) range")
        if self.pixel_size <= 0:
            raise ValueError("Pixel size must be positive")
        if self.exposure_time <= 0:
            raise ValueError("Exposure time must be positive")


@dataclass
class ObservationParameters:
    """观测参数容器类"""

    # 观测者位置
    latitude: float = 40.0  # 纬度 (度)
    longitude: float = 116.0  # 经度 (度)
    altitude: float = 0.0  # 海拔 (m)

    # 观测时间
    observation_time: Optional[Time] = None  # 观测时间

    # 望远镜指向
    azimuth: float = 180.0  # 方位角 (度，北=0，东=90)
    altitude_angle: float = 45.0  # 高度角 (度，地平=0，天顶=90)

    # 视场
    fov_width: float = 2.0  # 视场宽度 (度)
    fov_height: float = 2.0  # 视场高度 (度)

    # 星图参数
    star_density: float = 15.0  # 恒星密度 (每平方度)
    min_magnitude: float = -1.0  # 最亮星等 (如天狼星)
    max_magnitude: float = 12.0  # 最暗星等 (小型望远镜可见)

    # 限制参数 (None表示无限制)
    max_stars: Optional[int] = 10000  # 最大恒星数量限制
    max_flux_per_star: Optional[float] = 1e9  # 单星最大通量限制
    max_total_flux: Optional[float] = 1e9  # 总通量限制

    def __post_init__(self):
        """参数验证"""
        if not -90 <= self.latitude <= 90:
            raise ValueError("Latitude must be in [-90, 90] range")
        if not -180 <= self.longitude <= 180:
            raise ValueError("Longitude must be in [-180, 180] range")
        if not 0 <= self.altitude_angle <= 90:
            raise ValueError("Altitude angle must be in [0, 90] range")
        if self.fov_width <= 0 or self.fov_height <= 0:
            raise ValueError("Field of view must be positive")
        if self.star_density <= 0:
            raise ValueError("Star density must be positive")

        # 验证限制参数
        if self.max_stars is not None and self.max_stars <= 0:
            raise ValueError("max_stars must be a positive integer or None")
        if self.max_flux_per_star is not None and self.max_flux_per_star <= 0:
            raise ValueError("max_flux_per_star must be a positive float or None")
        if self.max_total_flux is not None and self.max_total_flux <= 0:
            raise ValueError("max_total_flux must be a positive float or None")

        # 设置默认观测时间（当前时间）
        if self.observation_time is None:
            self.observation_time = Time.now()


class InteractiveTelescopeSimulator:
    """
    交互式天文望远镜模拟器

    模拟完整的半球星图，并提供交互式控制功能，包括：
    1. 半球星图生成（模拟可见天空）
    2. 望远镜光学效应模拟
    3. 大气折射和消光
    4. 交互式参数调整和可视化

    参数：
    telescope_params : TelescopeParameters
        望远镜参数
    observation_params : ObservationParameters
        观测参数
    """

    def __init__(
        self,
        telescope_params: Optional[TelescopeParameters] = None,
        observation_params: Optional[ObservationParameters] = None,
    ):
        """初始化模拟器"""
        self.telescope_params = telescope_params or TelescopeParameters()
        self.observation_params = observation_params or ObservationParameters()

        # 初始化天文模拟器（用于生成恒星图像）
        self.astro_simulator = AstronomicalSimulator(
            image_size=max(
                self.telescope_params.sensor_width, self.telescope_params.sensor_height
            ),
            pixel_scale=self._calculate_pixel_scale(),
            zeropoint=25.0,
            gain=self.telescope_params.gain,  # 使用望远镜参数中的增益
            exposure_time=self.telescope_params.exposure_time,
        )

        # 半球星图数据
        self.hemisphere_stars: Optional[Dict[str, np.ndarray]] = None
        self.visible_stars: Optional[Dict[str, np.ndarray]] = None

        # 可视化组件
        self._colorbar = None

        # 生成半球星图
        self.generate_hemisphere_stars()

        # 更新可见恒星
        self.update_visible_stars()

        logger.info("Interactive telescope simulator initialized")

    def _calculate_pixel_scale(self) -> float:
        """计算像素比例（角秒/像素）"""
        # 像素比例 = (像素尺寸 / 焦距) * 206265 (角秒/弧度)
        pixel_scale = (
            self.telescope_params.pixel_size / self.telescope_params.focal_length
        ) * 206265.0
        return pixel_scale

    def generate_hemisphere_stars(self) -> None:
        """
        生成半球星图

        在可见半球（地平线以上）随机生成恒星，使用幂律分布模拟真实星等分布。
        """
        logger.info("Generating hemisphere star map...")

        # 计算半球上的恒星数量（每平方度密度）
        # 半球立体角：2π 球面度 = 2π * (180/π)^2 = 2 * 180^2 / π ≈ 20626.5 平方度
        hemisphere_area = 2 * (180.0 / np.pi) * 180.0  # 精确半球立体角（平方度）
        num_stars = int(self.observation_params.star_density * hemisphere_area)

        # 限制最大恒星数量以避免内存问题
        max_stars = self.observation_params.max_stars
        if max_stars is not None and num_stars > max_stars:
            logger.warning(f"Too many stars ({num_stars}), limiting to {max_stars}")
            num_stars = max_stars

        # 在半球上均匀分布恒星（使用球坐标）
        # 方位角：0-360度，高度角：0-90度
        azimuth = np.random.uniform(0, 360, num_stars)

        # 高度角分布：需要确保在球面上均匀分布
        # 对于球面均匀分布，高度角的正弦值应均匀分布
        sin_altitude = np.random.uniform(0, 1, num_stars)
        altitude_angle = np.degrees(np.arcsin(sin_altitude))

        # 生成星等（幂律分布）
        # dN/dm ∝ 10^(0.33*m) 近似
        magnitude_slope = 0.33
        min_mag = self.observation_params.min_magnitude
        max_mag = self.observation_params.max_magnitude

        # 使用逆变换采样生成幂律分布的星等
        u_rand = np.random.uniform(0, 1, num_stars)
        if abs(magnitude_slope) < 1e-10:
            magnitude = min_mag + u_rand * (max_mag - min_mag)
        else:
            c = 10 ** (magnitude_slope * min_mag)
            d = 10 ** (magnitude_slope * max_mag)
            magnitude = (1.0 / magnitude_slope) * np.log10(c + u_rand * (d - c))

        # 计算通量
        zeropoint = self.astro_simulator.zeropoint
        flux = 10 ** (-0.4 * (magnitude - zeropoint))

        # 限制通量值，避免数值溢出
        max_flux = self.observation_params.max_flux_per_star
        if max_flux is not None and np.any(flux > max_flux):
            logger.warning(f"Flux value too large, limiting to {max_flux:.1e}")
            flux = np.clip(flux, 0, max_flux)

        self.hemisphere_stars = {
            "azimuth": azimuth,
            "altitude": altitude_angle,
            "magnitude": magnitude,
            "flux": flux,
            "ra": np.zeros(num_stars),  # 暂留位置，可用于转换
            "dec": np.zeros(num_stars),
        }

        logger.info(f"Hemisphere star map generated: {num_stars} stars")

    def atmospheric_refraction(
        self, altitude_angle: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        计算大气折射修正

        参数：
        altitude_angle : float or ndarray
            真实高度角（度）

        返回：
        corrected_altitude : float or ndarray
            修正后的视高度角（度）
        """
        # 处理标量和数组输入
        altitude_angle_array = np.asarray(altitude_angle)
        altitude_rad = np.radians(altitude_angle_array)

        # 初始化修正后的高度角
        corrected_altitude_array = altitude_angle_array.copy()

        # 仅对大于0的高度角应用折射修正
        mask = altitude_angle_array > 0
        if np.any(mask):
            # Bennett公式（精度约0.1弧分）
            refraction_arcmin = 1.0 / np.tan(
                altitude_rad[mask] + 7.31 / (altitude_angle_array[mask] + 4.4)
            )
            refraction_deg = refraction_arcmin / 60.0

            # 考虑海拔修正
            altitude_factor = np.exp(-self.observation_params.altitude / 8500.0)
            refraction_deg *= altitude_factor

            corrected_altitude_array[mask] = altitude_angle_array[mask] + refraction_deg

        # 返回与输入相同的类型
        if isinstance(altitude_angle, np.ndarray):
            return corrected_altitude_array
        else:
            return float(corrected_altitude_array)

    def apply_optical_distortion(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用光学畸变（径向畸变）

        参数：
        x, y : ndarray
            归一化坐标（-1到1）

        返回：
        x_distorted, y_distorted : ndarray
            畸变后的坐标
        """
        # 计算径向距离
        r = np.sqrt(x**2 + y**2)

        # 径向畸变模型：r_distorted = r * (1 + k1*r^2 + k2*r^4)
        k1 = self.telescope_params.distortion_k1
        k2 = self.telescope_params.distortion_k2

        distortion_factor = 1.0 + k1 * r**2 + k2 * r**4

        # 应用畸变
        x_distorted = x * distortion_factor
        y_distorted = y * distortion_factor

        return x_distorted, y_distorted

    def apply_chromatic_aberration(
        self, x: np.ndarray, y: np.ndarray, wavelength_factor: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用色差效应

        参数：
        x, y : ndarray
            坐标
        wavelength_factor : float
            波长因子（1.0为中心波长）

        返回：
        x_aberrated, y_aberrated : ndarray
            有色差的坐标
        """
        # 色差导致不同波长的焦距不同
        # 简单模型：偏移量与到中心的距离成正比
        aberration_coeff = self.telescope_params.chromatic_aberration

        # 计算到中心的距离
        r = np.sqrt(x**2 + y**2)

        # 偏移量
        offset = aberration_coeff * r * (wavelength_factor - 1.0)

        # 径向偏移
        x_aberrated = x * (1.0 + offset)
        y_aberrated = y * (1.0 + offset)

        return x_aberrated, y_aberrated

    def apply_coma(
        self, x: np.ndarray, y: np.ndarray, direction_angle: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用彗差效应

        参数：
        x, y : ndarray
            坐标
        direction_angle : float
            彗差方向角（度）

        返回：
        x_comatic, y_comatic : ndarray
            有彗差的坐标
        """
        coma_coeff = self.telescope_params.coma_coefficient

        # 计算径向距离
        r = np.sqrt(x**2 + y**2)

        # 彗差方向
        theta = np.radians(direction_angle)

        # 彗差模型：偏移量与r^2成正比，方向固定
        offset_x = coma_coeff * r**2 * np.cos(theta)
        offset_y = coma_coeff * r**2 * np.sin(theta)

        x_comatic = x + offset_x
        y_comatic = y + offset_y

        return x_comatic, y_comatic

    def update_visible_stars(self) -> None:
        """更新当前视场内的可见恒星"""
        if self.hemisphere_stars is None:
            return

        azimuth = self.hemisphere_stars["azimuth"]
        altitude = self.hemisphere_stars["altitude"]

        # 获取当前望远镜指向
        center_az = self.observation_params.azimuth
        center_alt = self.observation_params.altitude_angle

        # 计算视场范围
        fov_width = self.observation_params.fov_width
        fov_height = self.observation_params.fov_height

        # 选择在视场内的恒星
        # 注意：需要处理方位角的周期性（0-360度）
        az_diff = np.abs(azimuth - center_az)
        az_diff = np.minimum(az_diff, 360 - az_diff)

        alt_diff = np.abs(altitude - center_alt)

        # 筛选条件
        in_fov = (az_diff <= fov_width / 2) & (alt_diff <= fov_height / 2)

        # 应用大气折射修正
        corrected_altitude = altitude.copy()
        if np.any(in_fov):
            corrected_altitude[in_fov] = self.atmospheric_refraction(altitude[in_fov])

        # 存储可见恒星
        self.visible_stars = {
            "azimuth": azimuth[in_fov],
            "altitude": corrected_altitude[in_fov],
            "magnitude": self.hemisphere_stars["magnitude"][in_fov],
            "flux": self.hemisphere_stars["flux"][in_fov],
        }

        logger.info(f"Visible stars updated: {np.sum(in_fov)} stars in field of view")

    def project_to_sensor(
        self, azimuth: np.ndarray, altitude: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将天空坐标投影到传感器平面

        参数：
        azimuth, altitude : ndarray
            方位角和高度角（度）

        返回：
        x, y : ndarray
            传感器平面坐标（像素）
        """
        # 获取望远镜指向中心
        center_az = self.observation_params.azimuth
        center_alt = self.observation_params.altitude_angle

        # 计算相对于中心的角距离
        # 简化处理：使用正投影（小视场近似）
        delta_az = azimuth - center_az
        delta_alt = altitude - center_alt

        # 处理方位角周期性
        delta_az = np.where(delta_az > 180, delta_az - 360, delta_az)
        delta_az = np.where(delta_az < -180, delta_az + 360, delta_az)

        # 将角度转换为弧度
        delta_az_rad = np.radians(delta_az)
        delta_alt_rad = np.radians(delta_alt)

        # 使用正投影：x = f * tan(delta_az), y = f * tan(delta_alt)
        # 对于小角度，tan(θ) ≈ θ
        focal_length = self.telescope_params.focal_length
        x_mm = focal_length * np.tan(delta_az_rad)
        y_mm = focal_length * np.tan(delta_alt_rad)

        # 转换为像素坐标
        pixel_size = self.telescope_params.pixel_size
        x_pixels = x_mm / (pixel_size * 1000)  # mm到米转换
        y_pixels = y_mm / (pixel_size * 1000)

        # 中心偏移到传感器中心
        sensor_width = self.telescope_params.sensor_width
        sensor_height = self.telescope_params.sensor_height

        x_pixels = x_pixels + sensor_width / 2
        y_pixels = y_pixels + sensor_height / 2

        return x_pixels, y_pixels

    def simulate_observation(self) -> Dict[str, Any]:
        """
        模拟单次观测

        返回：
        observation : dict
            包含观测结果的字典：
            - 'image': 模拟图像
            - 'stars': 星点位置和参数
            - 'parameters': 使用的参数
        """
        logger.info("Starting observation simulation...")

        # 更新可见恒星
        self.update_visible_stars()

        if self.visible_stars is None or len(self.visible_stars["azimuth"]) == 0:
            logger.warning("No stars in field of view")
            return {
                "image": np.zeros(
                    (
                        self.telescope_params.sensor_height,
                        self.telescope_params.sensor_width,
                    )
                ),
                "stars": {},
                "parameters": {
                    "telescope": self.telescope_params,
                    "observation": self.observation_params,
                },
            }

        # 投影到传感器平面
        x_pixels, y_pixels = self.project_to_sensor(
            self.visible_stars["azimuth"], self.visible_stars["altitude"]
        )

        # 应用光学效应
        # 1. 归一化坐标（用于畸变应用）
        sensor_width = self.telescope_params.sensor_width
        sensor_height = self.telescope_params.sensor_height

        x_norm = (x_pixels - sensor_width / 2) / (sensor_width / 2)
        y_norm = (y_pixels - sensor_height / 2) / (sensor_height / 2)

        # 2. 应用径向畸变
        x_distorted, y_distorted = self.apply_optical_distortion(x_norm, y_norm)

        # 3. 应用色差（简单中心波长）
        x_chromatic, y_chromatic = self.apply_chromatic_aberration(
            x_distorted, y_distorted, wavelength_factor=1.0
        )

        # 4. 应用彗差
        x_comatic, y_comatic = self.apply_coma(
            x_chromatic, y_chromatic, direction_angle=45.0
        )

        # 转换回像素坐标
        x_final = x_comatic * (sensor_width / 2) + sensor_width / 2
        y_final = y_comatic * (sensor_height / 2) + sensor_height / 2

        # 准备恒星参数用于图像生成
        # 计算总通量并限制最大值
        flux_total = self.visible_stars["flux"] * self.telescope_params.exposure_time
        max_flux_total = self.observation_params.max_total_flux
        if max_flux_total is not None and np.any(flux_total > max_flux_total):
            logger.warning(f"Total flux too large, limiting to {max_flux_total:.1e}")
            flux_total = np.clip(flux_total, 0, max_flux_total)

        stars_for_image = {
            "x": x_final,
            "y": y_final,
            "flux": self.visible_stars["flux"],
            "flux_total": flux_total,
            "magnitude": self.visible_stars["magnitude"],
        }

        # 生成PSF（考虑视宁度）
        seeing_fwhm = self.telescope_params.seeing_fwhm
        pixel_scale = self._calculate_pixel_scale()
        fwhm_pixels = seeing_fwhm / pixel_scale

        psf_kernel = self.astro_simulator.generate_psf(
            fwhm=fwhm_pixels, profile="moffat", beta=3.5
        )

        # 生成图像
        # 使用简化背景（根据高度角变化）
        altitude_mean = np.mean(self.visible_stars["altitude"])
        sky_brightness = 22.0 - 2.0 * (1.0 - altitude_mean / 90.0)  # 简单模型

        image = self.astro_simulator.generate_image(
            stars=stars_for_image,
            psf_kernel=psf_kernel,
            sky_brightness=sky_brightness,
            read_noise=self.telescope_params.read_noise,
            dark_current=self.telescope_params.dark_current,
            include_cosmic_rays=True,
            cosmic_ray_rate=0.0001,
        )

        # 应用大气消光
        airmass = 1.0 / np.sin(np.radians(altitude_mean)) if altitude_mean > 0 else 10.0
        extinction_factor = 10 ** (
            -0.4 * self.telescope_params.atmospheric_extinction * airmass
        )
        image = image * extinction_factor

        logger.info("Observation simulation completed")

        return {
            "image": image,
            "stars": {
                "x_pixels": x_final,
                "y_pixels": y_final,
                "magnitude": self.visible_stars["magnitude"],
                "azimuth": self.visible_stars["azimuth"],
                "altitude": self.visible_stars["altitude"],
            },
            "parameters": {
                "telescope": self.telescope_params,
                "observation": self.observation_params,
            },
        }

    def create_interactive_interface(self) -> None:
        """
        创建交互式界面

        使用matplotlib创建滑块和控制按钮，用于实时调整参数。
        """
        # 设置matplotlib字体以确保正确显示英文
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [
            "Arial",
            "DejaVu Sans",
            "Liberation Sans",
            "Bitstream Vera Sans",
            "sans-serif",
        ]
        plt.rcParams["axes.unicode_minus"] = False

        # 创建图形和子图
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(
            "Interactive Astronomical Telescope Simulator",
            fontsize=16,
            fontweight="bold",
        )

        # 创建子图布局，为底部控件留出空间
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3, bottom=0.18)

        # 星图显示（左上）
        ax_starmap = fig.add_subplot(gs[0:2, 0:2], projection="polar")
        ax_starmap.set_title("Hemisphere Star Map", fontsize=12)
        ax_starmap.set_theta_zero_location("N")
        ax_starmap.set_theta_direction(-1)
        ax_starmap.set_ylim(0, 90)
        ax_starmap.set_yticks([0, 30, 60, 90])
        ax_starmap.set_yticklabels(["Horizon", "30°", "60°", "Zenith"])

        # 望远镜视场显示（右上）
        ax_telescope = fig.add_subplot(gs[0:2, 2])
        ax_telescope.set_title("Telescope Field of View", fontsize=12)
        ax_telescope.set_xlabel("X (pixels)")
        ax_telescope.set_ylabel("Y (pixels)")
        ax_telescope.grid(True, alpha=0.3)

        # 图像显示（中下）
        ax_image = fig.add_subplot(gs[2, :])
        ax_image.set_title("Simulated Image", fontsize=12)
        ax_image.set_xlabel("X (pixels)")
        ax_image.set_ylabel("Y (pixels)")

        # 创建滑块区域（增加高度以提高可见性）
        slider_ax1 = fig.add_axes((0.15, 0.11, 0.25, 0.05))
        slider_ax2 = fig.add_axes((0.15, 0.07, 0.25, 0.05))
        slider_ax3 = fig.add_axes((0.55, 0.11, 0.25, 0.05))
        slider_ax4 = fig.add_axes((0.55, 0.07, 0.25, 0.05))

        # 设置滑块背景色以提高可见性
        slider_ax1.set_facecolor("lightgray")
        slider_ax2.set_facecolor("lightgray")
        slider_ax3.set_facecolor("lightgray")
        slider_ax4.set_facecolor("lightgray")

        # 为滑块axes添加边框以提高可见性
        for ax in [slider_ax1, slider_ax2, slider_ax3, slider_ax4]:
            ax.spines["top"].set_visible(True)
            ax.spines["bottom"].set_visible(True)
            ax.spines["left"].set_visible(True)
            ax.spines["right"].set_visible(True)
            ax.spines["top"].set_color("black")
            ax.spines["bottom"].set_color("black")
            ax.spines["left"].set_color("black")
            ax.spines["right"].set_color("black")
            ax.spines["top"].set_linewidth(1.5)
            ax.spines["bottom"].set_linewidth(1.5)
            ax.spines["left"].set_linewidth(1.5)
            ax.spines["right"].set_linewidth(1.5)

        # 创建滑块
        slider_azimuth = Slider(
            slider_ax1,
            "Azimuth",
            0,
            360,
            valinit=self.observation_params.azimuth,
            valstep=1,
            facecolor="blue",
            track_color="lightgray",
            valfmt="%0.0f°",
        )
        # 设置滑块字体和颜色
        slider_azimuth.valtext.set_fontsize(10)
        slider_azimuth.valtext.set_fontweight("bold")
        slider_azimuth.valtext.set_color("black")
        slider_azimuth.label.set_fontsize(11)
        slider_azimuth.label.set_fontweight("bold")
        slider_azimuth.label.set_color("darkblue")

        slider_altitude = Slider(
            slider_ax2,
            "Altitude",
            0,
            90,
            valinit=self.observation_params.altitude_angle,
            valstep=1,
            facecolor="blue",
            track_color="lightgray",
            valfmt="%0.0f°",
        )
        # 设置滑块字体和颜色
        slider_altitude.valtext.set_fontsize(10)
        slider_altitude.valtext.set_fontweight("bold")
        slider_altitude.valtext.set_color("black")
        slider_altitude.label.set_fontsize(11)
        slider_altitude.label.set_fontweight("bold")
        slider_altitude.label.set_color("darkblue")

        slider_fov = Slider(
            slider_ax3,
            "Field of View",
            0.1,
            10,
            valinit=self.observation_params.fov_width,
            valstep=0.1,
            facecolor="green",
            track_color="lightgray",
            valfmt="%0.1f°",
        )
        # 设置滑块字体和颜色
        slider_fov.valtext.set_fontsize(10)
        slider_fov.valtext.set_fontweight("bold")
        slider_fov.valtext.set_color("black")
        slider_fov.label.set_fontsize(11)
        slider_fov.label.set_fontweight("bold")
        slider_fov.label.set_color("darkgreen")

        slider_exposure = Slider(
            slider_ax4,
            "Exposure Time",
            1,
            300,
            valinit=self.telescope_params.exposure_time,
            valstep=1,
            facecolor="red",
            track_color="lightgray",
            valfmt="%0.0f s",
        )
        # 设置滑块字体和颜色
        slider_exposure.valtext.set_fontsize(10)
        slider_exposure.valtext.set_fontweight("bold")
        slider_exposure.valtext.set_color("black")
        slider_exposure.label.set_fontsize(11)
        slider_exposure.label.set_fontweight("bold")
        slider_exposure.label.set_color("darkred")

        # 创建按钮
        button_ax = fig.add_axes((0.85, 0.14, 0.1, 0.05))
        button_ax.set_facecolor("lightblue")
        # 为按钮axes添加边框
        button_ax.spines["top"].set_visible(True)
        button_ax.spines["bottom"].set_visible(True)
        button_ax.spines["left"].set_visible(True)
        button_ax.spines["right"].set_visible(True)
        button_ax.spines["top"].set_color("darkblue")
        button_ax.spines["bottom"].set_color("darkblue")
        button_ax.spines["left"].set_color("darkblue")
        button_ax.spines["right"].set_color("darkblue")
        button_ax.spines["top"].set_linewidth(2)
        button_ax.spines["bottom"].set_linewidth(2)
        button_ax.spines["left"].set_linewidth(2)
        button_ax.spines["right"].set_linewidth(2)
        update_button = Button(
            button_ax,
            "Update",
            color="lightblue",
            hovercolor="lightgreen",
        )
        # 设置按钮标签字体和颜色
        update_button.label.set_fontsize(12)
        update_button.label.set_fontweight("bold")
        update_button.label.set_color("darkblue")

        # 初始绘图
        self._update_plots(ax_starmap, ax_telescope, ax_image)

        # 调整图形布局，确保所有控件可见
        fig.subplots_adjust(bottom=0.25)

        # 更新函数
        def update(val):
            # 更新参数
            self.observation_params.azimuth = slider_azimuth.val
            self.observation_params.altitude_angle = slider_altitude.val
            self.observation_params.fov_width = slider_fov.val
            self.observation_params.fov_height = slider_fov.val
            self.telescope_params.exposure_time = slider_exposure.val

            # 更新显示
            self._update_plots(ax_starmap, ax_telescope, ax_image)
            fig.canvas.draw_idle()

        # 连接事件
        slider_azimuth.on_changed(update)
        slider_altitude.on_changed(update)
        slider_fov.on_changed(update)
        slider_exposure.on_changed(update)
        update_button.on_clicked(lambda event: update(None))

        plt.show()

    def _update_plots(
        self, ax_starmap: plt.Axes, ax_telescope: plt.Axes, ax_image: plt.Axes
    ) -> None:
        """
        更新所有绘图

        参数：
        ax_starmap, ax_telescope, ax_image : matplotlib.Axes
            各个子图的坐标轴
        """
        # 清除之前的绘图
        ax_starmap.clear()
        ax_telescope.clear()
        ax_image.clear()

        # 设置星图子图
        ax_starmap.set_title("Hemisphere Star Map", fontsize=12)
        ax_starmap.set_theta_zero_location("N")
        ax_starmap.set_theta_direction(-1)
        ax_starmap.set_ylim(0, 90)
        ax_starmap.set_yticks([0, 30, 60, 90])
        ax_starmap.set_yticklabels(["Horizon", "30°", "60°", "Zenith"])

        # 绘制半球星图
        if self.hemisphere_stars is not None:
            azimuth_rad = np.radians(self.hemisphere_stars["azimuth"])
            altitude = self.hemisphere_stars["altitude"]
            magnitude = self.hemisphere_stars["magnitude"]

            # 根据星等设置点的大小和颜色
            mag_norm = (magnitude - np.min(magnitude)) / (
                np.max(magnitude) - np.min(magnitude)
            )
            sizes = 50 * (1.0 - mag_norm) + 5
            colors = plt.cm.viridis(mag_norm)

            ax_starmap.scatter(
                azimuth_rad,
                altitude,
                s=sizes,
                c=colors,
                alpha=0.7,
                edgecolors="white",
                linewidth=0.5,
            )

        # 绘制望远镜指向和视场
        center_az_rad = np.radians(self.observation_params.azimuth)
        center_alt = self.observation_params.altitude_angle
        fov = self.observation_params.fov_width

        # 绘制指向点
        ax_starmap.plot(
            center_az_rad, center_alt, "ro", markersize=10, label="Telescope Pointing"
        )

        # 绘制视场范围（近似为圆）
        theta = np.linspace(0, 2 * np.pi, 100)
        r = np.ones_like(theta) * fov / 2

        # 转换为极坐标
        az_circle = center_az_rad + r * np.cos(theta) / np.cos(np.radians(center_alt))
        alt_circle = center_alt + r * np.sin(theta)

        ax_starmap.plot(az_circle, alt_circle, "r--", alpha=0.7, linewidth=1.5)
        ax_starmap.legend(loc="upper right", fontsize=8)

        # 设置望远镜视场子图
        ax_telescope.set_title("Telescope Field of View", fontsize=12)
        ax_telescope.set_xlabel("X (pixels)")
        ax_telescope.set_ylabel("Y (pixels)")
        ax_telescope.grid(True, alpha=0.3)
        ax_telescope.set_xlim(0, self.telescope_params.sensor_width)
        ax_telescope.set_ylim(0, self.telescope_params.sensor_height)

        # 绘制传感器边界
        sensor_rect = Rectangle(
            (0, 0),
            self.telescope_params.sensor_width,
            self.telescope_params.sensor_height,
            fill=False,
            edgecolor="blue",
            linewidth=2,
            alpha=0.5,
        )
        ax_telescope.add_patch(sensor_rect)

        # 绘制可见恒星
        if self.visible_stars is not None and len(self.visible_stars["azimuth"]) > 0:
            # 投影到传感器平面
            x_pixels, y_pixels = self.project_to_sensor(
                self.visible_stars["azimuth"], self.visible_stars["altitude"]
            )

            magnitude = self.visible_stars["magnitude"]
            mag_norm = (magnitude - np.min(magnitude)) / (
                np.max(magnitude) - np.min(magnitude)
            )
            sizes = 100 * (1.0 - mag_norm) + 10
            colors = plt.cm.plasma(mag_norm)

            ax_telescope.scatter(
                x_pixels,
                y_pixels,
                s=sizes,
                c=colors,
                alpha=0.8,
                edgecolors="white",
                linewidth=0.5,
            )

        # 设置图像子图
        ax_image.set_title("Simulated Image", fontsize=12)
        ax_image.set_xlabel("X (pixels)")
        ax_image.set_ylabel("Y (pixels)")

        # 生成并显示模拟图像
        observation = self.simulate_observation()
        image = observation["image"]

        if image is not None and image.size > 0:
            im = ax_image.imshow(
                image,
                cmap="gray",
                origin="lower",
                vmax=np.percentile(image, 99.5)
                if np.percentile(image, 99.5) > 0
                else 1.0,
            )
            # 删除已存在的颜色条
            if self._colorbar is not None:
                try:
                    self._colorbar.remove()
                except:
                    pass
                self._colorbar = None

            # 删除ax_image所在figure的其他颜色条（如果有）
            fig = ax_image.get_figure()
            if fig is not None:
                # 检查所有axes，找到并删除颜色条axes
                for ax in fig.get_axes():
                    if hasattr(ax, "colorbar") and ax.colorbar is not None:
                        try:
                            ax.colorbar.remove()
                        except:
                            pass
                    # 检查ax是否包含颜色条（通常颜色条有自己的axes）
                    if hasattr(ax, "name") and "colorbar" in str(ax.name).lower():
                        try:
                            ax.remove()
                        except:
                            pass

            # 创建新的颜色条
            self._colorbar = plt.colorbar(im, ax=ax_image, fraction=0.046, pad=0.04)

    def save_observation(self, filename: str) -> None:
        """
        保存观测结果到FITS文件

        参数：
        filename : str
            输出文件名
        """
        observation = self.simulate_observation()

        # 创建星表
        if observation["stars"]:
            from astropy.io import fits
            from astropy.table import Table

            catalog_data = {
                "ID": np.arange(1, len(observation["stars"]["x_pixels"]) + 1),
                "X_PIXEL": observation["stars"]["x_pixels"],
                "Y_PIXEL": observation["stars"]["y_pixels"],
                "MAG": observation["stars"]["magnitude"],
                "AZIMUTH": observation["stars"]["azimuth"],
                "ALTITUDE": observation["stars"]["altitude"],
            }

            catalog = Table(catalog_data)

            # 使用天文模拟器保存FITS
            self.astro_simulator.save_to_fits(
                image=observation["image"],
                catalog=catalog,
                filename=filename,
                overwrite=True,
            )

            logger.info(f"Observation saved: {filename}")
        else:
            logger.warning("No catalog data, cannot save FITS file")


# 演示函数
def demo_interactive_telescope() -> None:
    """演示交互式望远镜模拟器"""
    print("=" * 60)
    print("Interactive Astronomical Telescope Simulator Demo")
    print("=" * 60)

    # 创建望远镜参数
    telescope_params = TelescopeParameters(
        aperture_diameter=0.2,  # 20厘米口径
        focal_length=2.0,  # 2 meter focal length
        sensor_width=2048,
        sensor_height=2048,
        pixel_size=3.8e-6,  # 3.8 micron pixels
        exposure_time=60.0,  # 60 second exposure
        seeing_fwhm=1.5,  # 1.5 arcsec seeing
    )

    # 创建观测参数
    observation_params = ObservationParameters(
        latitude=31.2,  # Latitude near Shanghai
        longitude=121.5,  # Longitude near Shanghai
        azimuth=180.0,  # Pointing south
        altitude_angle=45.0,  # 45 degree altitude
        fov_width=1.0,  # 1 degree field of view
        star_density=30.0,  # 30 stars per square degree (more realistic density)
        min_magnitude=-1.0,
        max_magnitude=12.0,
    )

    # 创建模拟器
    simulator = InteractiveTelescopeSimulator(
        telescope_params=telescope_params, observation_params=observation_params
    )

    print("\nSimulator Parameters:")
    print(f"  Telescope aperture: {telescope_params.aperture_diameter} m")
    print(f"  Telescope focal length: {telescope_params.focal_length} m")
    print(
        f"  Sensor size: {telescope_params.sensor_width}×{telescope_params.sensor_height} pixels"
    )
    print(f"  Pixel scale: {simulator._calculate_pixel_scale():.2f} arcsec/pixel")
    print(
        f"  Observation location: Latitude={observation_params.latitude}°, Longitude={observation_params.longitude}°"
    )
    print(
        f"  Telescope pointing: Azimuth={observation_params.azimuth}°, Altitude={observation_params.altitude_angle}°"
    )
    print(
        f"  Field of view: {observation_params.fov_width}°×{observation_params.fov_height}°"
    )

    print("\nGenerating simulated observation...")
    observation = simulator.simulate_observation()

    if observation["stars"]:
        print(f"\nObservation Results:")
        print(f"  Number of stars detected: {len(observation['stars']['magnitude'])}")
        print(f"  Image dimensions: {observation['image'].shape}")
        print(
            f"  Image brightness range: [{observation['image'].min():.2f}, {observation['image'].max():.2f}]"
        )
        print(f"  Average magnitude: {np.mean(observation['stars']['magnitude']):.2f}")
    else:
        print("Warning: No stars detected in the field of view")

    print("\nLaunching interactive interface...")
    print("Instructions:")
    print("  1. Use sliders to adjust telescope pointing and parameters")
    print("  2. Click 'Update' button to manually refresh")
    print("  3. Close the window to end the demo")

    simulator.create_interactive_interface()


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 运行演示
    demo_interactive_telescope()
