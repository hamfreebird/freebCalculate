import datetime
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from astropy.convolution import convolve, convolve_fft
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


class AstronomicalSimulator:
    """
    天文模拟数据生成器

    该模拟器生成真实的天文观测图像，包括恒星、背景天空和各种仪器效应。

    参数：
    image_size : int, 可选 (默认=1024)
        生成图像的边长（像素），必须为正整数
    pixel_scale : float, 可选 (默认=0.2)
        像素比例（角秒/像素），必须为正数
    zeropoint : float, 可选 (默认=25.0)
        星等零点（m = -2.5*log10(flux) + zeropoint），AB星等系统
    gain : float, 可选 (默认=2.0)
        相机增益（e-/ADU），必须为正数
    exposure_time : float, 可选 (默认=1.0)
        曝光时间（秒），必须为正数
    ra_center : float, 可选 (默认=180.0)
        中心赤经（度）
    dec_center : float, 可选 (默认=0.0)
        中心赤纬（度）
    wcs_projection : str, 可选 (默认='TAN')
        WCS投影类型，支持'TAN'（正切投影）, 'SIN'（正弦投影）, 'CAR'（笛卡尔投影）

    异常：
    ValueError: 当输入参数无效时
    """

    def __init__(
        self,
        image_size: int = 1024,
        pixel_scale: float = 0.2,
        zeropoint: float = 25.0,
        gain: float = 2.0,
        exposure_time: float = 1.0,
        ra_center: float = 180.0,
        dec_center: float = 0.0,
        wcs_projection: str = "TAN",
    ):
        # 输入验证
        if image_size <= 0:
            raise ValueError(f"image_size必须是正整数，当前值: {image_size}")
        if pixel_scale <= 0:
            raise ValueError(f"pixel_scale必须是正数，当前值: {pixel_scale}")
        if gain <= 0:
            raise ValueError(f"gain必须是正数，当前值: {gain}")
        if exposure_time <= 0:
            raise ValueError(f"exposure_time必须是正数，当前值: {exposure_time}")
        if wcs_projection not in ["TAN", "SIN", "CAR"]:
            raise ValueError(
                f"不支持的投影类型: {wcs_projection}，支持: 'TAN', 'SIN', 'CAR'"
            )

        self.image_size: int = image_size
        self.pixel_scale: float = pixel_scale
        self.zeropoint: float = zeropoint
        self.gain: float = gain
        self.exposure_time: float = exposure_time
        self.ra_center: float = ra_center
        self.dec_center: float = dec_center
        self.wcs_projection: str = wcs_projection

        # 初始化WCS（世界坐标系）
        self.wcs: WCS = self._create_wcs()

        logger.info(
            "天文模拟器初始化: 图像尺寸=%d×%d, "
            "像素比例=%.2f″/像素, 零点=%.1f, "
            "增益=%.1f e-/ADU, 曝光时间=%.1f秒",
            image_size,
            image_size,
            pixel_scale,
            zeropoint,
            gain,
            exposure_time,
        )

    def _create_wcs(self) -> WCS:
        """
        创建WCS（世界坐标系）

        返回：
        wcs : WCS对象
            配置好的世界坐标系
        """
        w = WCS(naxis=2)
        w.wcs.crpix = [self.image_size / 2, self.image_size / 2]
        w.wcs.crval = [self.ra_center, self.dec_center]
        w.wcs.cdelt = np.array([-self.pixel_scale / 3600, self.pixel_scale / 3600])
        w.wcs.ctype = [f"RA---{self.wcs_projection}", f"DEC--{self.wcs_projection}"]
        w.wcs.radesys = "ICRS"  # 国际天球参考系
        w.wcs.equinox = 2000.0  # J2000.0历元

        return w

    def generate_psf(
        self,
        fwhm: float = 2.5,
        profile: str = "gaussian",
        beta: float = 4.0,
        ellipticity: float = 0.0,
        position_angle: float = 0.0,
    ) -> np.ndarray:
        """
        生成点扩散函数（PSF）核

        参数：
        fwhm : float, 可选 (默认=2.5)
            全宽半最大值（像素），必须为正数
        profile : str, 可选 ('gaussian'或'moffat')
            PSF类型
        beta : float, 可选 (默认=4.0)
            Moffat分布的beta参数（仅当profile='moffat'时使用），必须>1
        ellipticity : float, 可选 (默认=0.0)
            椭圆率，0为圆形，1为极端椭圆
        position_angle : float, 可选 (默认=0.0)
            椭圆方向角（度）

        返回：
        kernel : ndarray
            归一化的PSF核

        异常：
        ValueError: 当输入参数无效时
        """
        # 输入验证
        if fwhm <= 0:
            raise ValueError(f"fwhm必须是正数，当前值: {fwhm}")
        if profile not in ["gaussian", "moffat"]:
            raise ValueError(
                f"不支持的profile类型: {profile}，支持: 'gaussian', 'moffat'"
            )
        if profile == "moffat" and beta <= 1:
            raise ValueError(f"beta参数必须大于1，当前值: {beta}")
        if ellipticity < 0 or ellipticity >= 1:
            raise ValueError(f"ellipticity必须在[0, 1)范围内，当前值: {ellipticity}")

        # 计算核大小（确保足够大以包含PSF）
        if profile == "moffat":
            # Moffat分布需要更大的核
            size = int(fwhm * 12)
        else:  # 高斯
            size = int(fwhm * 3.5)

        # 确保核大小为奇数，以便有明确的中心
        if size % 2 == 0:
            size += 1

        half_size = size // 2
        y, x = np.mgrid[-half_size : half_size + 1, -half_size : half_size + 1]

        # 应用椭圆变换
        if ellipticity > 0:
            # 将椭圆率和位置角转换为变换矩阵
            theta = np.radians(position_angle)
            a = 1.0  # 长轴
            b = 1.0 - ellipticity  # 短轴
            # 旋转和缩放
            x_rot = x * np.cos(theta) + y * np.sin(theta)
            y_rot = -x * np.sin(theta) + y * np.cos(theta)
            r2 = (x_rot / a) ** 2 + (y_rot / b) ** 2
        else:
            r2 = x**2 + y**2

        if profile == "moffat":
            # 根据FWHM计算alpha参数
            # FWHM = 2 * alpha * sqrt(2^(1/beta) - 1)
            alpha = fwhm / (2 * np.sqrt(2 ** (1.0 / beta) - 1))
            kernel = (beta - 1) / (np.pi * alpha**2) * (1 + r2 / alpha**2) ** (-beta)
            logger.debug(f"生成Moffat PSF: FWHM={fwhm}, alpha={alpha:.3f}, beta={beta}")
        else:  # 高斯
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
            kernel = np.exp(-r2 / (2 * sigma**2))
            logger.debug(f"生成高斯PSF: FWHM={fwhm}, sigma={sigma:.3f}")

        # 归一化确保总通量为1
        kernel = kernel / kernel.sum()

        # 验证归一化
        if not np.isclose(kernel.sum(), 1.0, rtol=1e-10):
            warnings.warn(f"PSF核归一化不精确: sum={kernel.sum()}")
            kernel = kernel / kernel.sum()

        logger.info(f"生成{profile} PSF核: 尺寸={size}×{size}, FWHM={fwhm}像素")
        return kernel

    def generate_stars(
        self,
        num_stars: int = 100,
        min_mag: float = 18,
        max_mag: float = 24,
        distribution: str = "uniform",
        magnitude_law: str = "uniform",
        magnitude_slope: float = 0.33,
    ) -> Dict[str, np.ndarray]:
        """
        生成恒星参数

        参数：
        num_stars : int, 可选 (默认=100)
            恒星数量，必须为正整数
        min_mag : float, 可选 (默认=18)
            最小星等
        max_mag : float, 可选 (默认=24)
            最大星等，必须≥min_mag
        distribution : str, 可选 (默认='uniform')
            空间分布类型：'uniform'（均匀）, 'clustered'（聚类）
        magnitude_law : str, 可选 (默认='uniform')
            星等分布类型：'uniform'（均匀）, 'powerlaw'（幂律分布）
        magnitude_slope : float, 可选 (默认=0.33)
            幂律分布斜率（dN/dm ∝ 10^(slope*m)）

        返回：
        stars : dict
            包含位置、流量、星等的字典，键为：
            'x', 'y' (像素坐标), 'mag' (星等), 'flux' (总通量，e-/s)

        异常：
        ValueError: 当输入参数无效时
        """
        # 输入验证
        if num_stars <= 0:
            raise ValueError(f"num_stars必须是正整数，当前值: {num_stars}")
        if max_mag < min_mag:
            raise ValueError(
                f"max_mag必须≥min_mag，当前值: min_mag={min_mag}, max_mag={max_mag}"
            )
        if distribution not in ["uniform", "clustered"]:
            raise ValueError(
                f"不支持的分布类型: {distribution}，支持: 'uniform', 'clustered'"
            )
        if magnitude_law not in ["uniform", "powerlaw"]:
            raise ValueError(
                f"不支持的星等分布: {magnitude_law}，支持: 'uniform', 'powerlaw'"
            )

        # 生成位置
        if distribution == "uniform":
            # 均匀分布
            x = np.random.uniform(0, self.image_size, num_stars)
            y = np.random.uniform(0, self.image_size, num_stars)
        else:  # clustered
            # 生成几个聚类中心
            num_clusters = max(1, num_stars // 50)
            cluster_centers_x = np.random.uniform(
                0.2 * self.image_size, 0.8 * self.image_size, num_clusters
            )
            cluster_centers_y = np.random.uniform(
                0.2 * self.image_size, 0.8 * self.image_size, num_clusters
            )

            # 为每个恒星分配聚类中心
            cluster_assignments = np.random.choice(num_clusters, num_stars)
            x = np.zeros(num_stars)
            y = np.zeros(num_stars)

            for i in range(num_clusters):
                mask = cluster_assignments == i
                n = mask.sum()
                if n > 0:
                    # 围绕聚类中心的正态分布
                    x[mask] = np.random.normal(
                        cluster_centers_x[i], 0.05 * self.image_size, n
                    )
                    y[mask] = np.random.normal(
                        cluster_centers_y[i], 0.05 * self.image_size, n
                    )

            # 限制在图像范围内
            x = np.clip(x, 0, self.image_size)
            y = np.clip(y, 0, self.image_size)

        # 生成星等
        if magnitude_law == "uniform":
            # 均匀分布
            mag = np.random.uniform(min_mag, max_mag, num_stars)
        else:  # powerlaw
            # 幂律分布：dN/dm ∝ 10^(slope*m)
            # 使用逆变换采样
            u = np.random.uniform(0, 1, num_stars)
            if abs(magnitude_slope) < 1e-10:
                # 斜率为0时退化为均匀分布
                mag = min_mag + u * (max_mag - min_mag)
            else:
                c = 10 ** (magnitude_slope * min_mag)
                d = 10 ** (magnitude_slope * max_mag)
                mag = (1.0 / magnitude_slope) * np.log10(c + u * (d - c))

        # 计算通量（每秒电子数）
        # flux = 10^(-0.4*(mag - zeropoint)) [e-/s]
        flux = 10 ** (-0.4 * (mag - self.zeropoint))

        # 考虑曝光时间
        flux_total = flux * self.exposure_time  # 总电子数

        # 检查通量是否过大（可能导致数值问题）
        max_flux = flux_total.max()
        if max_flux > 1e10:
            warnings.warn(f"检测到极大通量值: {max_flux:.2e} e-，可能导致数值问题")

        logger.info(
            "生成%d颗恒星: 空间分布=%s, 星等分布=%s, 星等范围=%.1f-%.1f",
            num_stars,
            distribution,
            magnitude_law,
            min_mag,
            max_mag,
        )

        return {"x": x, "y": y, "mag": mag, "flux": flux, "flux_total": flux_total}

    def generate_image(
        self,
        stars: Dict[str, np.ndarray],
        psf_kernel: np.ndarray,
        sky_brightness: float = 21.0,
        read_noise: float = 5.0,
        dark_current: float = 0.1,
        include_cosmic_rays: bool = False,
        cosmic_ray_rate: float = 0.001,
    ) -> np.ndarray:
        """
        生成科学图像

        参数：
        stars : dict
            恒星参数，来自generate_stars()
        psf_kernel : ndarray
            PSF卷积核，来自generate_psf()
        sky_brightness : float
            天光背景星等（mag/arcsec²）
        read_noise : float
            读出噪声（e- RMS）
        dark_current : float
            暗电流（e-/像素/秒）
        include_cosmic_rays : bool
            是否包含宇宙射线
        cosmic_ray_rate : float
            宇宙射线率（事件/像素）

        返回：
        image : ndarray
            生成的图像数据（ADU）

        异常：
        ValueError: 当输入参数无效时
        """
        # 输入验证
        if read_noise < 0:
            raise ValueError(f"read_noise必须为非负数，当前值: {read_noise}")
        if dark_current < 0:
            raise ValueError(f"dark_current必须为非负数，当前值: {dark_current}")
        if cosmic_ray_rate < 0 or cosmic_ray_rate > 1:
            raise ValueError(
                f"cosmic_ray_rate必须在[0, 1]范围内，当前值: {cosmic_ray_rate}"
            )

        logger.info("开始生成图像...")

        # 1. 计算天光背景通量（电子/像素）
        # 星等每平方角秒 -> 每像素通量
        sky_flux_per_pix = 10 ** (-0.4 * (sky_brightness - self.zeropoint)) * (
            self.pixel_scale**2
        )
        sky_electrons = sky_flux_per_pix * self.exposure_time  # 总电子数/像素

        # 2. 计算暗电流（电子/像素）
        dark_electrons = dark_current * self.exposure_time

        # 3. 创建空图像（期望电子数）
        expected_electrons = np.full(
            (self.image_size, self.image_size), sky_electrons + dark_electrons
        )

        logger.debug(
            f"背景: 天光={sky_electrons:.2f} e-/像素, 暗电流={dark_electrons:.2f} e-/像素"
        )

        # 4. 添加恒星（使用向量化方法提高性能）
        # 创建星点图像（在整数像素位置）
        star_image = np.zeros((self.image_size, self.image_size))

        # 将恒星通量添加到最近的整数像素位置
        x_int = np.round(stars["x"]).astype(int)
        y_int = np.round(stars["y"]).astype(int)

        # 确保坐标在范围内
        mask = (
            (x_int >= 0)
            & (x_int < self.image_size)
            & (y_int >= 0)
            & (y_int < self.image_size)
        )
        x_int = x_int[mask]
        y_int = y_int[mask]
        flux_total_masked = stars["flux_total"][mask]

        # 使用bincount累加通量（更高效）
        if len(x_int) > 0:
            # 线性化索引
            indices = x_int * self.image_size + y_int
            # 使用bincount累加通量
            flat_star_image = np.bincount(
                indices,
                weights=flux_total_masked,
                minlength=self.image_size * self.image_size,
            )
            star_image = flat_star_image.reshape(self.image_size, self.image_size)

        logger.debug(f"添加{len(x_int)}颗恒星到星点图像")

        # 5. 对星点图像进行PSF卷积
        if star_image.sum() > 0:
            # 使用FFT卷积提高大核性能
            if psf_kernel.shape[0] > 32:
                logger.debug("使用FFT卷积（大核）")
                star_image_conv = convolve_fft(
                    star_image, psf_kernel, boundary="fill", fill_value=0
                )
            else:
                logger.debug("使用直接卷积（小核）")
                star_image_conv = convolve(
                    star_image, psf_kernel, boundary="fill", fill_value=0
                )

            # 验证通量守恒（应近似）
            input_flux = star_image.sum()
            output_flux = star_image_conv.sum()
            if input_flux > 0 and not np.isclose(input_flux, output_flux, rtol=0.01):
                logger.warning(
                    "PSF卷积后通量不守恒: 输入=%.2f, 输出=%.2f, 相对误差=%.2f%%",
                    input_flux,
                    output_flux,
                    (output_flux - input_flux) / input_flux * 100,
                )

            expected_electrons += star_image_conv

        # 6. 添加宇宙射线（可选）
        if include_cosmic_rays:
            cosmic_mask = (
                np.random.random((self.image_size, self.image_size)) < cosmic_ray_rate
            )
            n_cosmic = cosmic_mask.sum()
            if n_cosmic > 0:
                # 宇宙射线通量（典型值：100-10000 e-）
                cosmic_flux = np.random.lognormal(mean=7.0, sigma=1.0, size=n_cosmic)
                expected_electrons[cosmic_mask] += cosmic_flux
                logger.debug(f"添加{n_cosmic}个宇宙射线事件")

        # 7. 施加噪声模型
        # Poisson噪声（光子/电子噪声）
        noisy_electrons = np.random.poisson(expected_electrons).astype(float)

        # 读出噪声（高斯噪声）
        if read_noise > 0:
            read_noise_array = np.random.normal(0, read_noise, expected_electrons.shape)
            noisy_electrons += read_noise_array

        # 转换为ADU（模拟数）
        image = noisy_electrons / self.gain

        # 8. 添加坏像素/热像素（可选，此处简化）
        # 可以在这里添加固定图案噪声

        logger.info(
            "图像生成完成: 总通量=%.2e e-, 平均背景=%.2f e-/像素",
            expected_electrons.sum(),
            sky_electrons,
        )

        return image

    def generate_catalog(
        self, stars: Dict[str, np.ndarray], include_errors: bool = True
    ) -> Table:
        """
        生成星表表格

        参数：
        stars : dict
            恒星参数，来自generate_stars()
        include_errors : bool
            是否包含测量误差估计

        返回：
        catalog : astropy.Table
            包含恒星参数的表格
        """
        # 转换像素坐标到天文坐标
        ra, dec = self.wcs.all_pix2world(stars["x"], stars["y"], 0)

        # 创建基本表格
        catalog_data = {
            "ID": np.arange(1, len(stars["x"]) + 1),
            "RA": ra,
            "DEC": dec,
            "MAG": stars["mag"],
            "FLUX": stars["flux"],  # e-/s
            "FLUX_TOTAL": stars["flux_total"],  # 总电子数
            "X_PIXEL": stars["x"],
            "Y_PIXEL": stars["y"],
        }

        # 添加测量误差估计（如果请求）
        if include_errors:
            # 简化的误差估计：主要来自光子噪声
            snr = np.sqrt(stars["flux_total"])  # 信噪比（简化）
            mag_error = 1.0857 / snr  # 星等误差 ≈ 2.5*log10(1 + 1/SNR) ≈ 1.0857/SNR
            flux_error = stars["flux_total"] / snr  # 通量误差

            catalog_data["MAG_ERR"] = mag_error
            catalog_data["FLUX_ERR"] = flux_error
            catalog_data["SNR"] = snr

        catalog = Table(catalog_data)

        # 添加元数据
        catalog.meta["SIMULATR"] = "AstronomicalSimulator"
        catalog.meta["ZEROPNT"] = self.zeropoint
        catalog.meta["PIXSCALE"] = self.pixel_scale
        catalog.meta["EXPTIME"] = self.exposure_time

        logger.info("生成星表: %d个源", len(catalog))
        return catalog

    def save_to_fits(
        self,
        image: np.ndarray,
        catalog: Table,
        filename: str,
        overwrite: bool = False,
        compression: bool = True,
    ) -> None:
        """
        保存为FITS文件

        参数：
        image : ndarray
            图像数据
        catalog : Table
            星表数据
        filename : str
            输出文件名
        overwrite : bool
            是否覆盖已存在文件
        compression : bool
            是否使用FITS压缩

        异常：
        IOError: 当文件操作失败时
        """
        from pathlib import Path

        # 检查目录是否存在
        output_path = Path(filename)
        output_dir = output_path.parent
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"创建目录: {output_dir}")
            except Exception as e:
                raise IOError(f"无法创建目录 {output_dir}: {e}")

        # 检查文件是否已存在
        if not overwrite and output_path.exists():
            raise FileExistsError(f"文件已存在: {filename}。使用overwrite=True覆盖。")

        try:
            # 主HDU（图像）
            primary_hdu = fits.PrimaryHDU(data=image)
            primary_hdu.header.update(self.wcs.to_header())

            # 添加标准FITS关键字
            primary_hdu.header["BUNIT"] = ("ADU", "Data unit")
            primary_hdu.header["GAIN"] = (self.gain, "e-/ADU")
            primary_hdu.header["ZEROPNT"] = (self.zeropoint, "Magnitude zero point")
            primary_hdu.header["EXPTIME"] = (self.exposure_time, "Exposure time [s]")
            primary_hdu.header["PIXSCALE"] = (
                self.pixel_scale,
                "Pixel scale [arcsec/pix]",
            )
            primary_hdu.header["IMGSIZE"] = (self.image_size, "Image size [pixels]")
            primary_hdu.header["RADESYS"] = ("ICRS", "Reference system")
            primary_hdu.header["EQUINOX"] = (2000.0, "Equinox of coordinate system")

            # 添加创建信息
            primary_hdu.header["CREATOR"] = ("AstronomicalSimulator", "Software")
            primary_hdu.header["DATE"] = (
                datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "File creation date",
            )
            primary_hdu.header["COMMENT"] = "Simulated astronomical image"

            # 表格HDU
            table_hdu = fits.BinTableHDU(catalog)
            table_hdu.header["EXTNAME"] = "CATALOG"
            table_hdu.header["EXTVER"] = 1

            # 构建HDU列表
            hdulist = fits.HDUList([primary_hdu, table_hdu])

            # 保存文件
            if compression:
                # 使用图像压缩
                hdulist.writeto(filename, overwrite=overwrite, output_verify="fix")
                logger.info(f"保存压缩FITS文件: {filename}")
            else:
                hdulist.writeto(filename, overwrite=overwrite)
                logger.info(f"保存FITS文件: {filename}")

        except Exception as e:
            raise IOError(f"保存FITS文件失败: {e}")

    def validate_simulation(
        self, stars: Dict[str, np.ndarray], image: np.ndarray
    ) -> Dict[str, Any]:
        """
        验证模拟结果

        参数：
        stars : dict
            输入的恒星参数
        image : ndarray
            生成的图像

        返回：
        validation : dict
            验证结果，包含各种统计量
        """
        validation = {}

        # 1. 检查恒星数量
        validation["num_stars_input"] = len(stars["x"])

        # 2. 计算总通量
        total_flux_input = stars["flux_total"].sum()
        validation["total_flux_input"] = total_flux_input

        # 3. 图像统计
        validation["image_mean"] = image.mean()
        validation["image_std"] = image.std()
        validation["image_min"] = image.min()
        validation["image_max"] = image.max()

        # 4. 估计背景水平（使用稳健估计）
        from scipy import stats

        image_flat = image.flatten()
        # 使用中位数和MAD估计背景
        background_median = np.median(image_flat)
        mad = stats.median_abs_deviation(image_flat)
        validation["background_median"] = background_median
        validation["background_mad"] = mad

        # 5. 信噪比估计
        if total_flux_input > 0:
            # 近似信噪比
            snr_approx = np.sqrt(total_flux_input)
            validation["snr_approx"] = snr_approx

        logger.info(
            "模拟验证: %d颗恒星, 输入通量=%.2e e-, 图像均值=%.2f ADU",
            validation["num_stars_input"],
            total_flux_input,
            validation["image_mean"],
        )

        return validation


# 兼容性函数（保持向后兼容）
def create_simple_simulator():
    """创建简单模拟器（向后兼容）"""
    return AstronomicalSimulator()


class FITSReader:
    """
    FITS文件读取器和分析器

    用于读取、分析和可视化天文FITS文件，包括图像数据和星表。

    参数：
    filename : str
        FITS文件路径

    异常：
    FileNotFoundError: 当文件不存在时
    IOError: 当读取FITS文件失败时
    """

    def __init__(self, filename: str):
        """初始化FITS读取器"""
        from pathlib import Path

        self.filename = filename
        self.file_path = Path(filename)

        # 验证文件存在
        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {filename}")

        # 读取FITS文件
        self.hdul = None
        self.image_hdu = None
        self.catalog_hdu = None
        self.image_data = None
        self.catalog_data = None
        self.wcs = None

        self._load_fits()

        logger.info(f"FITS文件加载成功: {filename}")

    def _load_fits(self):
        """加载FITS文件内容"""
        try:
            self.hdul = fits.open(self.filename)

            # 查找图像HDU和星表HDU
            for i, hdu in enumerate(self.hdul):
                if hasattr(hdu, "data") and hdu.data is not None:
                    if hdu.data.ndim == 2:  # 2D图像
                        self.image_hdu = hdu
                        self.image_data = hdu.data
                        # 尝试创建WCS
                        if hasattr(hdu, "header"):
                            try:
                                self.wcs = WCS(hdu.header)
                            except Exception:
                                self.wcs = None
                    elif hdu.data.ndim == 1:  # 表格数据（星表）
                        self.catalog_hdu = hdu
                        self.catalog_data = Table(hdu.data)

            # 如果未找到图像HDU，使用第一个HDU
            if self.image_hdu is None and len(self.hdul) > 0:
                self.image_hdu = self.hdul[0]
                if hasattr(self.hdul[0], "data"):
                    self.image_data = self.hdul[0].data

            if self.image_data is None:
                raise ValueError("FITS文件中未找到图像数据")

        except Exception as e:
            raise IOError(f"读取FITS文件失败: {e}")

    def close(self):
        """关闭FITS文件"""
        if self.hdul is not None:
            self.hdul.close()
            self.hdul = None

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

    def get_file_info(self) -> Dict[str, Any]:
        """
        获取文件信息

        返回：
        info : dict
            包含文件信息的字典
        """
        info = {
            "filename": str(self.filename),
            "file_size": self.file_path.stat().st_size
            if self.file_path.exists()
            else 0,
            "num_hdus": len(self.hdul) if self.hdul else 0,
            "has_image": self.image_data is not None,
            "has_catalog": self.catalog_data is not None,
            "has_wcs": self.wcs is not None,
        }

        # 图像信息
        if self.image_data is not None:
            info.update(
                {
                    "image_shape": self.image_data.shape,
                    "image_dtype": str(self.image_data.dtype),
                    "image_min": float(np.nanmin(self.image_data)),
                    "image_max": float(np.nanmax(self.image_data)),
                    "image_mean": float(np.nanmean(self.image_data)),
                    "image_std": float(np.nanstd(self.image_data)),
                }
            )

        # 星表信息
        if self.catalog_data is not None:
            info.update(
                {
                    "catalog_rows": len(self.catalog_data),
                    "catalog_columns": list(self.catalog_data.colnames),
                }
            )

        # 头部信息
        if self.image_hdu is not None and hasattr(self.image_hdu, "header"):
            header = self.image_hdu.header
            info["header_keys"] = list(header.keys())

            # 提取常见关键字
            common_keys = [
                "TELESCOP",
                "INSTRUME",
                "FILTER",
                "EXPTIME",
                "GAIN",
                "ZEROPNT",
                "PIXSCALE",
                "RA",
                "DEC",
                "EQUINOX",
                "RADESYS",
            ]
            for key in common_keys:
                if key in header:
                    info[key] = header[key]

        return info

    def get_header_summary(self, max_keys: int = 20) -> List[Tuple[str, str, str]]:
        """
        获取头部信息摘要

        参数：
        max_keys : int
            最大显示的关键字数量

        返回：
        summary : list of tuples
            每个元组包含(关键字, 值, 注释)
        """
        if self.image_hdu is None or not hasattr(self.image_hdu, "header"):
            return []

        header = self.image_hdu.header
        summary = []

        for i, key in enumerate(header.keys()):
            if i >= max_keys:
                break
            value = header[key]
            comment = header.comments[key] if key in header.comments else ""
            summary.append((key, str(value), comment))

        return summary

    def get_catalog_summary(self, max_rows: int = 10) -> Optional[Table]:
        """
        获取星表摘要

        参数：
        max_rows : int
            最大显示的行数

        返回：
        summary : Table or None
            星表摘要（前max_rows行）
        """
        if self.catalog_data is None:
            return None

        return (
            self.catalog_data[:max_rows]
            if len(self.catalog_data) > max_rows
            else self.catalog_data
        )

    def display_image(
        self,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = "gray",
        stretch: str = "linear",
        percentile: float = 99.0,
        title: Optional[str] = None,
        show_colorbar: bool = True,
        save_path: Optional[str] = None,
        dpi: int = 150,
    ):
        """
        显示图像

        参数：
        figsize : tuple
            图形尺寸 (宽, 高)
        cmap : str
            颜色映射
        stretch : str
            拉伸类型：'linear', 'log', 'sqrt', 'asinh'
        percentile : float
            用于设置显示范围的百分位数
        title : str or None
            图形标题
        show_colorbar : bool
            是否显示颜色条
        save_path : str or None
            保存路径，如果提供则保存图像
        dpi : int
            保存图像时的DPI
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm, PowerNorm

            if self.image_data is None:
                logger.warning("没有图像数据可显示")
                return

            # 准备数据
            data = self.image_data.copy()
            data[~np.isfinite(data)] = (
                np.nanmin(data[np.isfinite(data)]) if np.any(np.isfinite(data)) else 0
            )

            # 计算显示范围
            vmin = np.nanmin(data)
            vmax = np.nanpercentile(data, percentile)

            # 创建图形
            fig, ax = plt.subplots(figsize=figsize)

            # 选择拉伸方式
            if stretch == "log":
                if vmin <= 0:
                    vmin = (
                        np.nanpercentile(data[data > 0], 1) if np.any(data > 0) else 1
                    )
                norm = LogNorm(vmin=vmin, vmax=vmax)
            elif stretch == "sqrt":
                norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
            elif stretch == "asinh":
                norm = PowerNorm(gamma=0.3, vmin=vmin, vmax=vmax)
            else:  # linear
                norm = None

            # 显示图像
            im = ax.imshow(
                data,
                origin="lower",
                cmap=cmap,
                norm=norm,
                vmin=vmin if norm is None else None,
                vmax=vmax if norm is None else None,
            )

            # 添加颜色条
            if show_colorbar:
                cbar = fig.colorbar(im, ax=ax, pad=0.01)
                cbar.set_label("Intensity [ADU]", rotation=270, labelpad=15)

            # 设置标题
            if title is None:
                title = f"FITS Image: {self.file_path.name}"
            ax.set_title(title, fontsize=14, pad=15)

            # 设置坐标轴标签
            ax.set_xlabel("X [pixel]", fontsize=12)
            ax.set_ylabel("Y [pixel]", fontsize=12)

            # 添加网格
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

            # 紧凑布局
            plt.tight_layout()

            # 保存或显示
            if save_path:
                plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
                logger.info(f"图像已保存: {save_path}")
                plt.close(fig)
            else:
                plt.show()

        except ImportError:
            logger.error("显示图像需要matplotlib库。请安装: pip install matplotlib")
        except Exception as e:
            logger.error(f"显示图像失败: {e}")

    def save_as_png(self, output_path: str, **kwargs):
        """
        保存图像为PNG文件

        参数：
        output_path : str
            输出PNG文件路径
        **kwargs : dict
            传递给display_image的参数
        """
        # 设置默认参数
        kwargs.setdefault("save_path", output_path)
        kwargs.setdefault("figsize", (10, 8))
        kwargs.setdefault("cmap", "gray")
        kwargs.setdefault("stretch", "linear")
        kwargs.setdefault("percentile", 99.0)
        kwargs.setdefault("show_colorbar", True)
        kwargs.setdefault("dpi", 150)

        self.display_image(**kwargs)

    def plot_catalog(
        self,
        x_col: str = "X_PIXEL",
        y_col: str = "Y_PIXEL",
        mag_col: str = "MAG",
        figsize: Tuple[int, int] = (10, 8),
        title: Optional[str] = None,
        show_image: bool = True,
        save_path: Optional[str] = None,
        dpi: int = 150,
    ):
        """
        绘制星表数据

        参数：
        x_col : str
            X坐标列名
        y_col : str
            Y坐标列名
        mag_col : str
            星等列名
        figsize : tuple
            图形尺寸
        title : str or None
            图形标题
        show_image : bool
            是否显示背景图像
        save_path : str or None
            保存路径
        dpi : int
            保存图像时的DPI
        """
        try:
            import matplotlib.pyplot as plt

            if self.catalog_data is None:
                logger.warning("没有星表数据可绘制")
                return

            # 检查必要的列
            required_cols = [x_col, y_col, mag_col]
            missing_cols = [
                col for col in required_cols if col not in self.catalog_data.colnames
            ]
            if missing_cols:
                logger.warning(f"星表缺少必要列: {missing_cols}")
                # 尝试使用替代列名
                if (
                    "X_PIXEL" not in self.catalog_data.colnames
                    and "x" in self.catalog_data.colnames
                ):
                    x_col = "x"
                if (
                    "Y_PIXEL" not in self.catalog_data.colnames
                    and "y" in self.catalog_data.colnames
                ):
                    y_col = "y"
                if (
                    "MAG" not in self.catalog_data.colnames
                    and "mag" in self.catalog_data.colnames
                ):
                    mag_col = "mag"

                # 重新检查
                required_cols = [x_col, y_col, mag_col]
                missing_cols = [
                    col
                    for col in required_cols
                    if col not in self.catalog_data.colnames
                ]
                if missing_cols:
                    logger.error(f"无法找到必要的列: {missing_cols}")
                    return

            # 获取数据
            x = self.catalog_data[x_col]
            y = self.catalog_data[y_col]
            mag = self.catalog_data[mag_col]

            # 创建图形
            fig, ax = plt.subplots(figsize=figsize)

            # 显示背景图像（如果需要）
            if show_image and self.image_data is not None:
                ax.imshow(
                    self.image_data,
                    origin="lower",
                    cmap="gray",
                    vmax=np.nanpercentile(self.image_data, 99),
                    alpha=0.7,
                )

            # 绘制星点
            # 根据星等设置点的大小和颜色
            if len(mag) > 0:
                # 星等越小越亮，点越大
                mag_norm = (
                    (mag - np.min(mag)) / (np.max(mag) - np.min(mag))
                    if np.max(mag) > np.min(mag)
                    else 0.5
                )
                sizes = 100 * (1.0 - mag_norm) + 10  # 10-110像素
                colors = plt.cm.viridis(mag_norm)

                scatter = ax.scatter(
                    x,
                    y,
                    s=sizes,
                    c=colors,
                    alpha=0.8,
                    edgecolors="white",
                    linewidth=0.5,
                )

                # 添加颜色条
                sm = plt.cm.ScalarMappable(
                    cmap=plt.cm.viridis,
                    norm=plt.Normalize(vmin=np.min(mag), vmax=np.max(mag)),
                )
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, pad=0.01)
                cbar.set_label("Magnitude", rotation=270, labelpad=15)

            # 设置标题
            if title is None:
                title = f"Catalog Plot: {self.file_path.name}"
            ax.set_title(title, fontsize=14, pad=15)

            # 设置坐标轴标签
            ax.set_xlabel(f"{x_col} [pixel]", fontsize=12)
            ax.set_ylabel(f"{y_col} [pixel]", fontsize=12)

            # 设置坐标轴范围
            if self.image_data is not None:
                ax.set_xlim(0, self.image_data.shape[1])
                ax.set_ylim(0, self.image_data.shape[0])

            # 添加网格
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

            # 紧凑布局
            plt.tight_layout()

            # 保存或显示
            if save_path:
                plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
                logger.info(f"星表图已保存: {save_path}")
                plt.close(fig)
            else:
                plt.show()

        except ImportError:
            logger.error("绘制星表需要matplotlib库。请安装: pip install matplotlib")
        except Exception as e:
            logger.error(f"绘制星表失败: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取图像统计信息

        返回：
        stats : dict
            包含各种统计信息的字典
        """
        if self.image_data is None:
            return {}

        data = self.image_data.copy()
        data[~np.isfinite(data)] = np.nan

        stats = {
            "min": float(np.nanmin(data)),
            "max": float(np.nanmax(data)),
            "mean": float(np.nanmean(data)),
            "median": float(np.nanmedian(data)),
            "std": float(np.nanstd(data)),
            "sum": float(np.nansum(data)),
            "nan_count": int(np.sum(~np.isfinite(self.image_data))),
            "pixel_count": int(np.prod(data.shape)),
        }

        # 百分位数
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            stats[f"percentile_{p}"] = float(np.nanpercentile(data, p))

        # 直方图统计（简化）
        hist, bins = np.histogram(data[~np.isnan(data)], bins=50)
        stats["histogram"] = {"counts": hist.tolist(), "bins": bins.tolist()}

        return stats

    def print_summary(self):
        """打印文件摘要信息"""
        info = self.get_file_info()

        print("=" * 60)
        print(f"FITS文件摘要: {info['filename']}")
        print("=" * 60)

        print(f"\n基本信息:")
        print(f"  文件大小: {info['file_size'] / 1024:.1f} KB")
        print(f"  HDU数量: {info['num_hdus']}")
        print(f"  包含图像: {info['has_image']}")
        print(f"  包含星表: {info['has_catalog']}")
        print(f"  包含WCS: {info['has_wcs']}")

        if info["has_image"]:
            print(f"\n图像信息:")
            print(f"  尺寸: {info['image_shape']}")
            print(f"  数据类型: {info['image_dtype']}")
            print(f"  范围: [{info['image_min']:.2f}, {info['image_max']:.2f}]")
            print(f"  均值±标准差: {info['image_mean']:.2f} ± {info['image_std']:.2f}")

        if info["has_catalog"]:
            print(f"\n星表信息:")
            print(f"  行数: {info['catalog_rows']}")
            print(
                f"  列: {', '.join(info['catalog_columns'][:5])}{'...' if len(info['catalog_columns']) > 5 else ''}"
            )

        # 常见头部信息
        common_keys = ["TELESCOP", "INSTRUME", "FILTER", "EXPTIME", "GAIN", "ZEROPNT"]
        has_common = False
        print(f"\n头部信息:")
        for key in common_keys:
            if key in info:
                print(f"  {key}: {info[key]}")
                has_common = True

        if not has_common:
            print("  (无常见头部信息)")

        print("=" * 60)


class HemisphereStarSimulator:
    """
    半球星图模拟器

    模拟完整的半球天空（理论上在地球上能观测到的最大范围），
    支持交互式天文模拟，包括指定方位、视场、CMOS镜头大小等参数，
    并模拟现实中的望远镜效果。

    参数：
    latitude : float, 可选 (默认=40.0)
        观测点纬度（度），北纬为正，南纬为负
    longitude : float, 可选 (默认=116.0)
        观测点经度（度），东经为正，西经为负
    altitude : float, 可选 (默认=0.0)
        观测点海拔（米）
    timezone : str, 可选 (默认='UTC')
        时区
    star_density : float, 可选 (默认=0.001)
        恒星密度（颗/平方度）
    min_magnitude : float, 可选 (默认=0.0)
        最亮星等（肉眼可见约6等）
    max_magnitude : float, 可选 (默认=6.5)
        最暗星等
    magnitude_law_slope : float, 可选 (默认=0.33)
        星等分布幂律斜率
    """

    def __init__(
        self,
        latitude: float = 40.0,
        longitude: float = 116.0,
        altitude: float = 0.0,
        timezone: str = "UTC",
        star_density: float = 0.001,
        min_magnitude: float = 0.0,
        max_magnitude: float = 6.5,
        magnitude_law_slope: float = 0.33,
    ):
        # 输入验证
        if not (-90 <= latitude <= 90):
            raise ValueError(f"纬度必须在-90到90度之间，当前值: {latitude}")
        if not (-180 <= longitude <= 180):
            raise ValueError(f"经度必须在-180到180度之间，当前值: {longitude}")
        if altitude < 0:
            raise ValueError(f"海拔必须为非负数，当前值: {altitude}")
        if star_density <= 0:
            raise ValueError(f"恒星密度必须为正数，当前值: {star_density}")
        if max_magnitude < min_magnitude:
            raise ValueError(
                f"最大星等必须≥最小星等，当前值: min={min_magnitude}, max={max_magnitude}"
            )

        # 观测点参数
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.timezone = timezone

        # 恒星参数
        self.star_density = star_density
        self.min_magnitude = min_magnitude
        self.max_magnitude = max_magnitude
        self.magnitude_law_slope = magnitude_law_slope

        # 半球总立体角（球面度）
        self.hemisphere_solid_angle = 2 * np.pi  # 2π球面度

        # 计算预期恒星数量
        # 1平方度 = (π/180)^2 ≈ 0.0003046球面度
        deg2_to_sr = (np.pi / 180) ** 2
        self.expected_stars = int(star_density * 180 * 180 * np.pi)  # 半球约32400平方度

        # 存储生成的恒星数据
        self.stars_radec = None  # 赤经赤纬坐标
        self.stars_magnitude = None  # 星等
        self.stars_spectral_type = None  # 光谱类型（用于颜色）

        # 望远镜/相机参数（默认值）
        self.telescope_params = {
            "aperture": 0.1,  # 口径（米）
            "focal_length": 1.0,  # 焦距（米）
            "focal_ratio": 10.0,  # 焦比
            "sensor_width": 36.0,  # 传感器宽度（mm）
            "sensor_height": 24.0,  # 传感器高度（mm）
            "pixel_size": 5.0,  # 像素大小（微米）
            "quantum_efficiency": 0.8,  # 量子效率
            "read_noise": 3.0,  # 读出噪声（e-）
            "dark_current": 0.1,  # 暗电流（e-/像素/秒）
        }

        # 大气参数
        self.atmosphere_params = {
            "pressure": 1013.25,  # 气压（hPa）
            "temperature": 15.0,  # 温度（℃）
            "humidity": 50.0,  # 湿度（%）
            "seeing": 2.0,  # 视宁度（角秒）
            "extinction_coefficient": 0.2,  # 大气消光系数（星等/大气质量）
        }

        # 光学畸变参数
        self.optical_distortion = {
            "radial_k1": -0.1,  # 径向畸变系数k1
            "radial_k2": 0.01,  # 径向畸变系数k2
            "tangential_p1": 0.001,  # 切向畸变系数p1
            "tangential_p2": 0.001,  # 切向畸变系数p2
        }

        logger.info(
            "半球星图模拟器初始化: 纬度=%.1f°, 经度=%.1f°, 海拔=%.0fm, 期望恒星数=%d",
            latitude,
            longitude,
            altitude,
            self.expected_stars,
        )

    def generate_hemisphere_stars(self, random_seed: Optional[int] = None):
        """
        生成半球星图恒星数据

        参数：
        random_seed : int or None
            随机种子，用于可重复性

        返回：
        stars_data : dict
            包含恒星数据的字典，键为：
            'ra' (赤经，度), 'dec' (赤纬，度),
            'magnitude' (星等), 'spectral_type' (光谱类型)
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # 计算实际恒星数量（泊松分布）
        actual_stars = np.random.poisson(self.expected_stars)

        # 生成均匀分布在半球上的点
        # 使用球面坐标：赤纬从-90到90度，赤经从0到360度
        # 但为了半球（可见天空），我们生成整个天球的恒星

        # 整个天球的恒星数量（加倍以确保半球有足够恒星）
        total_stars = actual_stars * 2

        # 均匀分布在球面上
        # 生成随机方向向量
        u = np.random.random(total_stars)
        v = np.random.random(total_stars)

        # 转换为球面坐标（赤经赤纬）
        ra = 360 * u  # 赤经：0-360度
        dec = np.degrees(np.arcsin(2 * v - 1))  # 赤纬：-90到90度

        # 生成星等（幂律分布）
        # dN/dm ∝ 10^(slope*m)
        u_mag = np.random.random(total_stars)
        magnitude = (
            self.min_magnitude + (self.max_magnitude - self.min_magnitude) * u_mag
        )

        # 应用幂律分布
        if abs(self.magnitude_law_slope) > 1e-10:
            c = 10 ** (self.magnitude_law_slope * self.min_magnitude)
            d = 10 ** (self.magnitude_law_slope * self.max_magnitude)
            magnitude = (1.0 / self.magnitude_law_slope) * np.log10(c + u_mag * (d - c))

        # 生成光谱类型（简化模型）
        # 基于星等和随机分布
        spectral_types = ["O", "B", "A", "F", "G", "K", "M"]
        spectral_probs = [0.00001, 0.001, 0.01, 0.03, 0.08, 0.12, 0.76]  # M型星最多

        # 较亮的恒星更可能是早期类型（OBA）
        spectral_idx = np.random.choice(
            len(spectral_types), total_stars, p=spectral_probs
        )
        # 调整：较亮恒星更可能是早期类型
        brightness_factor = 1.0 - (magnitude - self.min_magnitude) / (
            self.max_magnitude - self.min_magnitude
        )
        early_type_bias = brightness_factor * 0.3  # 亮度对早期类型的偏置
        spectral_idx = np.clip(
            spectral_idx - (early_type_bias * len(spectral_types)).astype(int),
            0,
            len(spectral_types) - 1,
        )

        spectral_type = [spectral_types[i] for i in spectral_idx]

        # 存储数据
        self.stars_radec = np.column_stack([ra, dec])
        self.stars_magnitude = magnitude
        self.stars_spectral_type = spectral_type

        stars_data = {
            "ra": ra,
            "dec": dec,
            "magnitude": magnitude,
            "spectral_type": spectral_type,
            "count": len(ra),
        }

        logger.info(
            "生成半球星图: %d颗恒星, 星等范围: %.1f-%.1f",
            len(ra),
            magnitude.min(),
            magnitude.max(),
        )

        return stars_data

    def radec_to_altaz(
        self,
        ra: np.ndarray,
        dec: np.ndarray,
        observation_time: Optional[datetime.datetime] = None,
        lst: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将赤道坐标转换为地平坐标（高度角/方位角）

        参数：
        ra : ndarray
            赤经（度）
        dec : ndarray
            赤纬（度）
        observation_time : datetime or None
            观测时间，如果为None则使用当前时间
        lst : float or None
            地方恒星时（度），如果提供则忽略observation_time

        返回：
        altitude, azimuth : ndarray, ndarray
            高度角（度），方位角（度）
        """
        if observation_time is None and lst is None:
            observation_time = datetime.datetime.now(datetime.timezone.utc)

        # 简化转换：假设观测点在春分点，忽略岁差、章动等
        # 实际应用中应使用更精确的天文学库

        if lst is None:
            # 简化计算地方恒星时（LST）
            # 实际应用应使用更精确公式
            from datetime import datetime

            utc_time = observation_time
            # 简化的LST计算
            jd = self._datetime_to_jd(utc_time)
            t = (jd - 2451545.0) / 36525.0
            # 格林尼治恒星时（度）
            gst = (
                280.46061837
                + 360.98564736629 * (jd - 2451545.0)
                + 0.000387933 * t**2
                - t**3 / 38710000.0
            )
            gst = gst % 360.0
            lst = (gst + self.longitude) % 360.0
        else:
            lst = lst % 360.0

        # 转换赤经为时角
        ha = (lst - ra) % 360.0
        ha_rad = np.radians(ha)
        dec_rad = np.radians(dec)
        lat_rad = np.radians(self.latitude)

        # 计算高度角（altitude）
        sin_alt = np.sin(dec_rad) * np.sin(lat_rad) + np.cos(dec_rad) * np.cos(
            lat_rad
        ) * np.cos(ha_rad)
        altitude = np.degrees(np.arcsin(sin_alt))

        # 计算方位角（azimuth）
        cos_az = (np.sin(dec_rad) - np.sin(lat_rad) * np.sin(altitude)) / (
            np.cos(lat_rad) * np.cos(altitude)
        )
        cos_az = np.clip(cos_az, -1.0, 1.0)
        azimuth = np.degrees(np.arccos(cos_az))

        # 根据时角调整方位角象限
        mask = np.sin(ha_rad) >= 0
        azimuth[mask] = 360 - azimuth[mask]

        return altitude, azimuth

    def _datetime_to_jd(self, dt: datetime.datetime) -> float:
        """将datetime转换为儒略日（简化版本）"""
        # 简化计算，实际应用应使用更精确公式
        year = dt.year
        month = dt.month
        day = dt.day
        hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0

        if month <= 2:
            year -= 1
            month += 12

        a = year // 100
        b = 2 - a + a // 4

        jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
        jd += hour / 24.0

        return jd

    def get_visible_stars(
        self,
        min_altitude: float = 0.0,
        observation_time: Optional[datetime.datetime] = None,
        fov_width: float = 60.0,
        fov_height: float = 60.0,
        center_azimuth: float = 180.0,
        center_altitude: float = 45.0,
    ) -> Dict[str, np.ndarray]:
        """
        获取指定视场内的可见恒星

        参数：
        min_altitude : float
            最小高度角（度），低于此值的恒星不可见
        observation_time : datetime or None
            观测时间
        fov_width : float
            视场宽度（度）
        fov_height : float
            视场高度（度）
        center_azimuth : float
            中心方位角（度，北=0，东=90）
        center_altitude : float
            中心高度角（度）

        返回：
        visible_stars : dict
            可见恒星数据，包含像素坐标、星等、光谱类型等
        """
        if self.stars_radec is None:
            self.generate_hemisphere_stars()

        ra = self.stars_radec[:, 0]
        dec = self.stars_radec[:, 1]

        # 转换为地平坐标
        altitude, azimuth = self.radec_to_altaz(ra, dec, observation_time)

        # 筛选在地平线以上的恒星
        above_horizon = altitude >= min_altitude

        if not np.any(above_horizon):
            return {
                "pixel_x": np.array([]),
                "pixel_y": np.array([]),
                "magnitude": np.array([]),
                "altitude": np.array([]),
                "azimuth": np.array([]),
                "spectral_type": [],
                "count": 0,
            }

        # 筛选在视场内的恒星
        az_diff = (azimuth[above_horizon] - center_azimuth) % 360.0
        az_diff = np.where(az_diff > 180, az_diff - 360, az_diff)
        alt_diff = altitude[above_horizon] - center_altitude

        in_fov = (np.abs(az_diff) <= fov_width / 2) & (
            np.abs(alt_diff) <= fov_height / 2
        )

        if not np.any(in_fov):
            return {
                "pixel_x": np.array([]),
                "pixel_y": np.array([]),
                "magnitude": np.array([]),
                "altitude": np.array([]),
                "azimuth": np.array([]),
                "spectral_type": [],
                "count": 0,
            }

        # 获取视场内的恒星数据
        visible_az = azimuth[above_horizon][in_fov]
        visible_alt = altitude[above_horizon][in_fov]
        visible_mag = self.stars_magnitude[above_horizon][in_fov]
        visible_spectral = [
            self.stars_spectral_type[i] for i in np.where(above_horizon)[0][in_fov]
        ]

        # 转换为像素坐标（假设简单的正投影）
        # 方位角转换为X坐标，高度角转换为Y坐标
        pixel_x = ((visible_az - center_azimuth) % 360.0) / fov_width * 1024 + 512
        pixel_x = np.where(pixel_x > 1024, pixel_x - 1024, pixel_x)
        pixel_y = (visible_alt - center_altitude) / fov_height * 1024 + 512

        # 应用大气折射修正（近地面）
        refracted_alt = self._apply_atmospheric_refraction(visible_alt)
        # 更新像素坐标考虑折射
        pixel_y_refracted = (refracted_alt - center_altitude) / fov_height * 1024 + 512

        # 应用光学畸变
        if (
            self.optical_distortion["radial_k1"] != 0
            or self.optical_distortion["radial_k2"] != 0
        ):
            pixel_x, pixel_y_refracted = self._apply_optical_distortion(
                pixel_x, pixel_y_refracted
            )

        visible_stars = {
            "pixel_x": pixel_x,
            "pixel_y": pixel_y_refracted,  # 使用经过折射和畸变修正的Y坐标
            "magnitude": visible_mag,
            "altitude": visible_alt,
            "azimuth": visible_az,
            "refracted_altitude": refracted_alt,
            "spectral_type": visible_spectral,
            "count": len(pixel_x),
        }

        logger.info(
            "获取可见恒星: %d颗 (视场: %.1f°×%.1f°, 中心: 方位%.1f°, 高度%.1f°)",
            len(pixel_x),
            fov_width,
            fov_height,
            center_azimuth,
            center_altitude,
        )

        return visible_stars

    def _apply_atmospheric_refraction(self, altitude: np.ndarray) -> np.ndarray:
        """应用大气折射修正"""
        # 简化的大气折射公式（近地面）
        # 实际折射与气压、温度、湿度等有关
        altitude_rad = np.radians(altitude)

        # 标准大气条件下的折射（单位：角分）
        refraction_arcmin = 1.0 / np.tan(
            altitude_rad + 7.31 / (altitude_rad * 180 / np.pi + 4.4)
        )

        # 考虑大气条件修正
        pressure_factor = self.atmosphere_params["pressure"] / 1013.25
        temperature_factor = 283.0 / (273.0 + self.atmosphere_params["temperature"])

        refraction_arcmin *= pressure_factor * temperature_factor

        # 转换为度并添加到高度角
        refraction_deg = refraction_arcmin / 60.0

        # 对于低高度角，折射效应更强
        low_alt_mask = altitude < 20
        if np.any(low_alt_mask):
            extra_refraction = 0.5 * (20 - altitude[low_alt_mask]) / 20.0
            refraction_deg[low_alt_mask] += extra_refraction

        refracted_altitude = altitude + refraction_deg

        # 确保不超过90度
        refracted_altitude = np.clip(refracted_altitude, 0, 90)

        return refracted_altitude

    def _apply_optical_distortion(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """应用光学畸变（径向和切向）"""
        # 归一化坐标到[-1, 1]范围
        cx, cy = 512, 512  # 图像中心
        x_norm = (x - cx) / 512.0
        y_norm = (y - cy) / 512.0

        # 计算径向距离
        r2 = x_norm**2 + y_norm**2
        r4 = r2**2

        # 径向畸变
        radial_dist = (
            1
            + self.optical_distortion["radial_k1"] * r2
            + self.optical_distortion["radial_k2"] * r4
        )

        # 切向畸变
        tangential_x = 2 * self.optical_distortion["tangential_p1"] * x_norm * y_norm
        tangential_x += self.optical_distortion["tangential_p2"] * (r2 + 2 * x_norm**2)

        tangential_y = self.optical_distortion["tangential_p1"] * (r2 + 2 * y_norm**2)
        tangential_y += 2 * self.optical_distortion["tangential_p2"] * x_norm * y_norm

        # 应用畸变
        x_distorted = x_norm * radial_dist + tangential_x
        y_distorted = y_norm * radial_dist + tangential_y

        # 转换回像素坐标
        x_final = x_distorted * 512.0 + cx
        y_final = y_distorted * 512.0 + cy

        return x_final, y_final

    def simulate_telescope_image(
        self,
        fov_width: float = 2.0,
        fov_height: float = 2.0,
        center_azimuth: float = 180.0,
        center_altitude: float = 45.0,
        sensor_width_px: int = 2048,
        sensor_height_px: int = 2048,
        exposure_time: float = 30.0,
        iso: float = 800.0,
        include_noise: bool = True,
        monochrome: bool = False,
        observation_time: Optional[datetime.datetime] = None,
    ) -> np.ndarray:
        """
        模拟望远镜图像

        参数：
        fov_width : float
            视场宽度（度）
        fov_height : float
            视场高度（度）
        center_azimuth : float
            中心方位角（度）
        center_altitude : float
            中心高度角（度）
        sensor_width_px : int
            传感器宽度（像素）
        sensor_height_px : int
            传感器高度（像素）
        exposure_time : float
            曝光时间（秒）
        iso : float
            ISO感光度
        include_noise : bool
            是否包含噪声
        monochrome : bool
            是否单色（黑白）模式
        observation_time : datetime or None
            观测时间

        返回：
        image : ndarray
            模拟的望远镜图像（2D数组）
        """
        # 获取可见恒星
        visible_stars = self.get_visible_stars(
            min_altitude=0.0,
            observation_time=observation_time,
            fov_width=fov_width,
            fov_height=fov_height,
            center_azimuth=center_azimuth,
            center_altitude=center_altitude,
        )

        if visible_stars["count"] == 0:
            # 创建空白图像
            image = np.zeros((sensor_height_px, sensor_width_px))
            logger.warning("视场内无可见恒星，返回空白图像")
            return image

        # 创建空白图像
        image = np.zeros((sensor_height_px, sensor_width_px))

        # 计算恒星在传感器上的位置
        # 将方位角/高度角坐标映射到像素坐标
        pixel_x = visible_stars["pixel_x"]
        pixel_y = visible_stars["pixel_y"]

        # 缩放和偏移到传感器尺寸
        scale_x = sensor_width_px / 1024.0
        scale_y = sensor_height_px / 1024.0

        sensor_x = pixel_x * scale_x
        sensor_y = pixel_y * scale_y

        # 确保在传感器范围内
        valid_mask = (
            (sensor_x >= 0)
            & (sensor_x < sensor_width_px)
            & (sensor_y >= 0)
            & (sensor_y < sensor_height_px)
        )

        if not np.any(valid_mask):
            logger.warning("无恒星在传感器范围内")
            return image

        sensor_x = sensor_x[valid_mask]
        sensor_y = sensor_y[valid_mask]
        magnitudes = visible_stars["magnitude"][valid_mask]
        spectral_types = [
            visible_stars["spectral_type"][i] for i in np.where(valid_mask)[0]
        ]

        # 计算恒星通量（简化模型）
        # 星等到通量的转换，考虑曝光时间和ISO
        base_flux = 1000.0 * iso / 100.0 * exposure_time

        # 星等越暗，通量越小（每差5等，通量差100倍）
        flux = base_flux * 10 ** (-0.4 * magnitudes)

        # 应用大气消光（高度角越低，消光越强）
        altitude = visible_stars["altitude"][valid_mask]
        airmass = 1.0 / np.sin(np.radians(np.maximum(altitude, 1.0)))  # 避免除零
        extinction = self.atmosphere_params["extinction_coefficient"] * airmass
        flux *= 10 ** (-0.4 * extinction)

        # 应用望远镜效率
        telescope_efficiency = 0.7  # 简化假设
        flux *= telescope_efficiency

        # 创建PSF（点扩散函数）
        seeing_arcsec = self.atmosphere_params["seeing"]  # 视宁度
        pixel_scale = fov_width / sensor_width_px * 3600.0  # 角秒/像素
        psf_sigma = (
            seeing_arcsec / pixel_scale / (2 * np.sqrt(2 * np.log(2)))
        )  # 转换为像素sigma

        # 为每个恒星添加PSF
        psf_size = int(np.ceil(5 * psf_sigma))  # 5sigma范围
        if psf_size < 3:
            psf_size = 3

        # 生成PSF核（高斯）
        y_psf, x_psf = np.mgrid[-psf_size : psf_size + 1, -psf_size : psf_size + 1]
        psf_kernel = np.exp(-(x_psf**2 + y_psf**2) / (2 * psf_sigma**2))
        psf_kernel /= psf_kernel.sum()  # 归一化

        # 添加恒星到图像
        for i, (x, y, f) in enumerate(zip(sensor_x, sensor_y, flux)):
            x_int = int(np.round(x))
            y_int = int(np.round(y))

            # 计算PSF在图像中的位置
            x_start = max(0, x_int - psf_size)
            x_end = min(sensor_width_px, x_int + psf_size + 1)
            y_start = max(0, y_int - psf_size)
            y_end = min(sensor_height_px, y_int + psf_size + 1)

            # PSF核的对应部分
            psf_x_start = max(0, psf_size - (x_int - x_start))
            psf_x_end = psf_size + 1 + min(0, sensor_width_px - (x_int + psf_size + 1))
            psf_y_start = max(0, psf_size - (y_int - y_start))
            psf_y_end = psf_size + 1 + min(0, sensor_height_px - (y_int + psf_size + 1))

            # 添加恒星通量
            image[y_start:y_end, x_start:x_end] += (
                f * psf_kernel[psf_y_start:psf_y_end, psf_x_start:psf_x_end]
            )

        # 添加天空背景（光污染+气辉）
        sky_brightness = 19.0  # 星等/平方角秒（郊区天空）
        sky_flux_per_pixel = (
            base_flux * 10 ** (-0.4 * sky_brightness) * (pixel_scale / 3600.0) ** 2
        )
        image += sky_flux_per_pixel

        # 添加噪声
        if include_noise:
            # 光子噪声（泊松）
            image = np.random.poisson(image)

            # 读出噪声（高斯）
            read_noise = self.telescope_params["read_noise"]
            image += np.random.normal(0, read_noise, image.shape)

            # 暗电流
            dark_current = self.telescope_params["dark_current"] * exposure_time
            image += np.random.poisson(dark_current, image.shape)

        # 转换为8位或16位整数
        image_max = np.max(image)
        if image_max > 0:
            if image_max < 256:
                image = np.clip(image, 0, 255).astype(np.uint8)
            else:
                image = np.clip(image / image_max * 65535, 0, 65535).astype(np.uint16)

        logger.info(
            "望远镜图像模拟完成: 尺寸=%dx%d, 恒星数=%d, 最大亮度=%.1f",
            sensor_width_px,
            sensor_height_px,
            len(sensor_x),
            image_max,
        )

        return image

    def visualize_hemisphere(
        self,
        observation_time: Optional[datetime.datetime] = None,
        projection: str = "mollweide",
        figsize: Tuple[int, int] = (12, 8),
        show_constellations: bool = True,
        show_grid: bool = True,
        save_path: Optional[str] = None,
        dpi: int = 150,
    ):
        """
        可视化整个半球星图

        参数：
        observation_time : datetime or None
            观测时间
        projection : str
            投影类型：'mollweide'（摩尔维德）, 'aitoff'（艾托夫）, 'polar'（极坐标）
        figsize : tuple
            图形尺寸
        show_constellations : bool
            是否显示星座连线（简化版）
        show_grid : bool
            是否显示网格
        save_path : str or None
            保存路径
        dpi : int
            DPI
        """
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle

            if self.stars_radec is None:
                self.generate_hemisphere_stars()

            ra = self.stars_radec[:, 0]
            dec = self.stars_radec[:, 1]
            magnitude = self.stars_magnitude

            # 转换为观测时刻的地平坐标
            altitude, azimuth = self.radec_to_altaz(ra, dec, observation_time)

            # 只显示地平线以上的恒星
            above_horizon = altitude >= 0
            if not np.any(above_horizon):
                logger.warning("无恒星在地平线以上")
                return

            az_visible = azimuth[above_horizon]
            alt_visible = altitude[above_horizon]
            mag_visible = magnitude[above_horizon]

            # 根据投影类型创建图形
            fig, ax = plt.subplots(
                figsize=figsize, subplot_kw={"projection": projection}
            )

            # 根据星等设置点的大小和颜色
            # 星等越小越亮，点越大
            mag_norm = (mag_visible - self.min_magnitude) / (
                self.max_magnitude - self.min_magnitude
            )
            mag_norm = np.clip(mag_norm, 0, 1)
            sizes = 50 * (1.0 - mag_norm) + 5  # 5-55像素

            # 颜色：较亮恒星偏蓝，较暗恒星偏黄
            colors = plt.cm.plasma(mag_norm)  # 使用plasma颜色映射

            # 绘制恒星
            if projection == "mollweide" or projection == "aitoff":
                # 将方位角/高度角转换为投影坐标
                # 对于全天球投影，使用赤道坐标更合适
                ra_rad = np.radians(ra[above_horizon])
                dec_rad = np.radians(dec[above_horizon])

                # 调整赤经范围
                ra_rad = np.where(ra_rad > np.pi, ra_rad - 2 * np.pi, ra_rad)

                scatter = ax.scatter(
                    ra_rad,
                    dec_rad,
                    s=sizes,
                    c=colors,
                    alpha=0.8,
                    edgecolors="white",
                    linewidth=0.3,
                )
            else:  # 极坐标投影（方位角/高度角）
                # 转换为极坐标
                theta = np.radians(az_visible)
                r = 90 - alt_visible  # 中心为天顶

                scatter = ax.scatter(
                    theta,
                    r,
                    s=sizes,
                    c=colors,
                    alpha=0.8,
                    edgecolors="white",
                    linewidth=0.3,
                )

            # 添加颜色条
            sm = plt.cm.ScalarMappable(
                cmap=plt.cm.plasma,
                norm=plt.Normalize(vmin=self.min_magnitude, vmax=self.max_magnitude),
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, pad=0.05)
            cbar.set_label("星等", rotation=270, labelpad=15)

            # 显示星座连线（简化版）
            if show_constellations:
                self._add_constellation_lines(ax, projection, observation_time)

            # 显示网格
            if show_grid:
                ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

            # 设置标题和标签
            if observation_time is None:
                time_str = "当前时间"
            else:
                time_str = observation_time.strftime("%Y-%m-%d %H:%M:%S")

            title = f"半球星图模拟\n观测点: 纬度={self.latitude:.1f}°, 经度={self.longitude:.1f}° | 时间: {time_str}"
            ax.set_title(title, fontsize=14, pad=20)

            if projection in ["mollweide", "aitoff"]:
                ax.set_xlabel("赤经", fontsize=12)
                ax.set_ylabel("赤纬", fontsize=12)
            else:
                ax.set_xlabel("方位角", fontsize=12)
                ax.set_ylabel("90°-高度角", fontsize=12)

            # 添加地平线
            if projection == "polar":
                horizon = Circle(
                    (0, 0),
                    90,
                    fill=False,
                    linestyle="-",
                    linewidth=2,
                    alpha=0.5,
                    color="red",
                )
                ax.add_patch(horizon)
                ax.text(
                    0, 95, "地平线", ha="center", va="bottom", fontsize=10, color="red"
                )

            plt.tight_layout()

            # 保存或显示
            if save_path:
                plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
                logger.info(f"半球星图已保存: {save_path}")
                plt.close(fig)
            else:
                plt.show()

        except ImportError:
            logger.error("可视化需要matplotlib库。请安装: pip install matplotlib")
        except Exception as e:
            logger.error(f"可视化失败: {e}")

    def _add_constellation_lines(
        self, ax, projection: str, observation_time: Optional[datetime.datetime] = None
    ):
        """添加星座连线（简化版）"""
        # 简化的星座数据（几个主要星座）
        constellations = {
            "大熊座": [
                (165, 55),
                (160, 50),
                (155, 48),
                (150, 45),
                (145, 42),
                (140, 40),
                (135, 38),
            ],
            "猎户座": [(85, 0), (80, -5), (75, -8), (85, -10), (90, -12), (95, -15)],
            "天鹅座": [(310, 40), (305, 35), (300, 30), (295, 25), (290, 20)],
            "天琴座": [(280, 38), (278, 36), (276, 34), (274, 32)],
        }

        try:
            for name, stars in constellations.items():
                # 将赤经赤纬转换为观测时刻的地平坐标
                ra = np.array([s[0] for s in stars])
                dec = np.array([s[1] for s in stars])

                altitude, azimuth = self.radec_to_altaz(ra, dec, observation_time)

                # 只绘制地平线以上的部分
                above_horizon = altitude >= 0
                if np.sum(above_horizon) < 2:
                    continue

                az_visible = azimuth[above_horizon]
                alt_visible = altitude[above_horizon]

                if projection in ["mollweide", "aitoff"]:
                    # 使用赤道坐标
                    ra_rad = np.radians(ra[above_horizon])
                    dec_rad = np.radians(dec[above_horizon])
                    ra_rad = np.where(ra_rad > np.pi, ra_rad - 2 * np.pi, ra_rad)

                    # 绘制连线
                    ax.plot(ra_rad, dec_rad, "w-", alpha=0.5, linewidth=1)
                    # 添加星座名称
                    mid_idx = len(ra_rad) // 2
                    ax.text(
                        ra_rad[mid_idx],
                        dec_rad[mid_idx],
                        name,
                        fontsize=8,
                        color="white",
                        alpha=0.7,
                        ha="center",
                    )
                else:  # 极坐标
                    theta = np.radians(az_visible)
                    r = 90 - alt_visible

                    ax.plot(theta, r, "w-", alpha=0.5, linewidth=1)
                    mid_idx = len(theta) // 2
                    ax.text(
                        theta[mid_idx],
                        r[mid_idx],
                        name,
                        fontsize=8,
                        color="white",
                        alpha=0.7,
                        ha="center",
                    )
        except Exception as e:
            logger.debug(f"添加星座连线失败: {e}")

    def interactive_viewer(self):
        """启动交互式查看器（简化版）"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.widgets import Button, Slider

            # 创建图形和坐标轴
            fig, (ax_image, ax_sky) = plt.subplots(1, 2, figsize=(14, 6))
            plt.subplots_adjust(bottom=0.35)

            # 初始参数
            init_params = {
                "azimuth": 180.0,
                "altitude": 45.0,
                "fov": 5.0,
                "exposure": 30.0,
                "iso": 800.0,
            }

            # 初始图像
            if self.stars_radec is None:
                self.generate_hemisphere_stars()

            # 模拟望远镜图像
            image = self.simulate_telescope_image(
                fov_width=init_params["fov"],
                fov_height=init_params["fov"],
                center_azimuth=init_params["azimuth"],
                center_altitude=init_params["altitude"],
                exposure_time=init_params["exposure"],
                iso=init_params["iso"],
            )

            # 显示图像
            im = ax_image.imshow(image, cmap="gray", origin="lower")
            ax_image.set_title(f"望远镜视图 (FOV: {init_params['fov']}°)")
            ax_image.set_xlabel("X (像素)")
            ax_image.set_ylabel("Y (像素)")

            # 显示天空视图
            self._plot_sky_view(
                ax_sky,
                init_params["azimuth"],
                init_params["altitude"],
                init_params["fov"],
            )

            # 创建滑块
            axcolor = "lightgoldenrodyellow"

            # 方位角滑块
            ax_azimuth = plt.axes([0.15, 0.25, 0.65, 0.03], facecolor=axcolor)
            azimuth_slider = Slider(
                ax_azimuth, "方位角", 0.0, 360.0, valinit=init_params["azimuth"]
            )

            # 高度角滑块
            ax_altitude = plt.axes([0.15, 0.20, 0.65, 0.03], facecolor=axcolor)
            altitude_slider = Slider(
                ax_altitude, "高度角", 0.0, 90.0, valinit=init_params["altitude"]
            )

            # 视场滑块
            ax_fov = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
            fov_slider = Slider(ax_fov, "视场", 0.5, 30.0, valinit=init_params["fov"])

            # 曝光时间滑块
            ax_exposure = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=axcolor)
            exposure_slider = Slider(
                ax_exposure, "曝光时间", 1.0, 300.0, valinit=init_params["exposure"]
            )

            # ISO滑块
            ax_iso = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)
            iso_slider = Slider(
                ax_iso, "ISO", 100.0, 6400.0, valinit=init_params["iso"]
            )

            # 更新函数
            def update(val):
                # 获取当前滑块值
                azimuth = azimuth_slider.val
                altitude = altitude_slider.val
                fov = fov_slider.val
                exposure = exposure_slider.val
                iso = iso_slider.val

                # 更新望远镜图像
                new_image = self.simulate_telescope_image(
                    fov_width=fov,
                    fov_height=fov,
                    center_azimuth=azimuth,
                    center_altitude=altitude,
                    exposure_time=exposure,
                    iso=iso,
                )

                # 更新图像显示
                im.set_data(new_image)
                im.set_clim(vmin=new_image.min(), vmax=new_image.max())
                ax_image.set_title(f"望远镜视图 (FOV: {fov:.1f}°)")

                # 更新天空视图
                ax_sky.clear()
                self._plot_sky_view(ax_sky, azimuth, altitude, fov)

                fig.canvas.draw_idle()

            # 连接滑块事件
            azimuth_slider.on_changed(update)
            altitude_slider.on_changed(update)
            fov_slider.on_changed(update)
            exposure_slider.on_changed(update)
            iso_slider.on_changed(update)

            # 重置按钮
            resetax = plt.axes([0.8, 0.01, 0.1, 0.04])
            button = Button(resetax, "重置", color=axcolor, hovercolor="0.975")

            def reset(event):
                azimuth_slider.reset()
                altitude_slider.reset()
                fov_slider.reset()
                exposure_slider.reset()
                iso_slider.reset()

            button.on_clicked(reset)

            plt.show()

        except ImportError:
            logger.error("交互式查看器需要matplotlib库。请安装: pip install matplotlib")
        except Exception as e:
            logger.error(f"启动交互式查看器失败: {e}")

    def _plot_sky_view(self, ax, azimuth: float, altitude: float, fov: float):
        """绘制天空视图"""
        # 获取可见恒星
        visible_stars = self.get_visible_stars(
            min_altitude=0.0,
            fov_width=fov,
            fov_height=fov,
            center_azimuth=azimuth,
            center_altitude=altitude,
        )

        if visible_stars["count"] > 0:
            # 绘制恒星
            sizes = (
                100
                * (
                    1.0
                    - (visible_stars["magnitude"] - self.min_magnitude)
                    / (self.max_magnitude - self.min_magnitude)
                )
                + 10
            )
            scatter = ax.scatter(
                visible_stars["pixel_x"],
                visible_stars["pixel_y"],
                s=sizes,
                c="white",
                alpha=0.8,
                edgecolors="yellow",
                linewidth=0.5,
            )

        # 设置图形属性
        ax.set_xlim(0, 1024)
        ax.set_ylim(0, 1024)
        ax.set_aspect("equal")
        ax.set_facecolor("navy")
        ax.set_title(f"天空视图 (中心: 方位{azimuth:.1f}°, 高度{altitude:.1f}°)")
        ax.set_xlabel("X (相对坐标)")
        ax.set_ylabel("Y (相对坐标)")

        # 添加视场框
        fov_box = plt.Rectangle(
            (256, 256),
            512,
            512,
            fill=False,
            edgecolor="red",
            linewidth=2,
            linestyle="--",
        )
        ax.add_patch(fov_box)

        # 添加中心标记
        ax.plot(512, 512, "r+", markersize=10, markeredgewidth=2)


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 示例用法
    try:
        # 初始化模拟器（使用更多参数）
        sim = AstronomicalSimulator(
            image_size=4096,
            pixel_scale=0.3,
            zeropoint=25.0,
            gain=2.0,
            exposure_time=30.0,  # 30秒曝光
            ra_center=150.0,  # 不同中心坐标
            dec_center=30.0,
            wcs_projection="TAN",
        )

        # 生成恒星参数（使用幂律分布）
        stars = sim.generate_stars(
            num_stars=20000,
            min_mag=18,
            max_mag=26,
            distribution="clustered",
            magnitude_law="powerlaw",
            magnitude_slope=0.33,
        )

        # 生成PSF核（Moffat分布，带椭圆）
        psf = sim.generate_psf(
            fwhm=3.0, profile="moffat", beta=3.5, ellipticity=0.2, position_angle=45.0
        )

        # 生成图像（包含更多噪声成分）
        image = sim.generate_image(
            stars=stars,
            psf_kernel=psf,
            sky_brightness=20.5,
            read_noise=3.0,
            dark_current=0.05,
            include_cosmic_rays=True,
            cosmic_ray_rate=0.0005,
        )

        # 生成星表（包含误差）
        catalog = sim.generate_catalog(stars, include_errors=True)

        # 验证模拟
        validation = sim.validate_simulation(stars, image)

        # 保存文件
        sim.save_to_fits(
            image=image,
            catalog=catalog,
            filename="simulated_observation_optimized.fits",
            overwrite=True,
            compression=True,
        )

        print("模拟完成！")
        print(f"恒星数量: {len(stars['x'])}")
        print(f"图像尺寸: {image.shape}")
        print(f"图像统计: 均值={image.mean():.2f} ADU, 标准差={image.std():.2f} ADU")
        print("文件已保存: simulated_observation_optimized.fits")

        with FITSReader('simulated_observation_optimized.fits') as reader:
            # 自定义显示参数
            reader.display_image(
                figsize=(14, 12),
                stretch='asinh',  # 使用反双曲正弦拉伸，适合深场图像
                percentile=99.9,  # 使用99.9%百分位作为显示上限
                title='Deep field astronomical image',
                show_colorbar=True
            )

            # 如果有星表，绘制高级星表图
            if reader.catalog_data is not None:
                reader.plot_catalog(
                    x_col='RA',  # 使用赤经坐标
                    y_col='DEC',  # 使用赤纬坐标
                    mag_col='MAG',
                    show_image=False,  # 不显示背景图像
                    title='Star table distribution (celestial coordinates)',
                    figsize=(12, 10)
                )

    except Exception as e:
        logger.error(f"模拟失败: {e}")
        import traceback

        traceback.print_exc()
