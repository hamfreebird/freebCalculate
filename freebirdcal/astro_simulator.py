import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.convolution import convolve
from astropy.wcs import WCS
from scipy.signal import convolve2d


class AstronomicalSimulator:
    """
    天文模拟数据生成器

    参数：
    image_size : int, 可选 (默认=1024)
        生成图像的边长（像素）
    pixel_scale : float, 可选 (默认=0.2)
        像素比例（角秒/像素）
    zeropoint : float, 可选 (默认=25.0)
        星等零点（m = -2.5*log10(flux) + zeropoint）
    gain : float, 可选 (默认=2.0)
        相机增益（e-/ADU）
    """

    def __init__(self, image_size=1024, pixel_scale=0.2, zeropoint=25.0, gain=2.0):
        self.image_size = image_size
        self.pixel_scale = pixel_scale
        self.zeropoint = zeropoint
        self.gain = gain

        # 初始化WCS（世界坐标系）
        self.wcs = self._create_wcs()

    def _create_wcs(self):
        """创建基本的WCS坐标系"""
        w = WCS(naxis=2)
        w.wcs.crpix = [self.image_size / 2, self.image_size / 2]
        w.wcs.crval = [180.0, 0.0]  # 中心坐标（赤经180度，赤纬0度）
        w.wcs.cdelt = np.array([-self.pixel_scale / 3600, self.pixel_scale / 3600])
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        return w

    def generate_psf(self, fwhm=2.5, profile='gaussian'):
        """
        生成点扩散函数（PSF）核

        参数：
        fwhm : float, 可选 (默认=2.5)
            全宽半最大值（像素）
        profile : str, 可选 ('gaussian'或'moffat')
            PSF类型

        返回：
        kernel : ndarray
            归一化的PSF核
        """
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

        if profile == 'moffat':
            alpha = 2.5  # 典型Moffat参数
            beta = 4.0
            size = int(fwhm * 10)
            y, x = np.mgrid[-size:size + 1, -size:size + 1]
            kernel = (beta - 1) / (np.pi * alpha ** 2) * (1 + (x ** 2 + y ** 2) / alpha ** 2) ** (-beta)
        else:  # 高斯
            size = int(fwhm * 3)
            y, x = np.mgrid[-size:size + 1, -size:size + 1]
            kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

        return kernel / kernel.sum()

    def generate_stars(self, num_stars=100, min_mag=18, max_mag=24):
        """
        生成恒星参数

        返回：
        stars : dict
            包含位置、流量、星等的字典
        """
        # 均匀分布位置
        x = np.random.uniform(0, self.image_size, num_stars)
        y = np.random.uniform(0, self.image_size, num_stars)

        # 生成星等（对数均匀分布）
        mag = np.random.uniform(min_mag, max_mag, num_stars)
        flux = 10 ** (-0.4 * (mag - self.zeropoint))

        return {'x': x, 'y': y, 'mag': mag, 'flux': flux}

    def generate_image(self, stars, psf_kernel, sky_brightness=21.0):
        """
        生成科学图像

        参数：
        stars : dict
            恒星参数
        psf_kernel : ndarray
            PSF卷积核
        sky_brightness : float
            天光背景星等（mag/arcsec²）

        返回：
        image : ndarray
            生成的图像数据（ADU）
        """
        # 计算天光背景（ADU/像素）
        sky_flux_per_pix = 10 ** (-0.4 * (sky_brightness - self.zeropoint)) * (self.pixel_scale ** 2)
        image = np.random.poisson(sky_flux_per_pix * self.gain,
                                  (self.image_size, self.image_size)).astype(float) / self.gain

        # 添加恒星
        for x, y, flux in zip(stars['x'], stars['y'], stars['flux']):
            x_floor = int(np.floor(x))
            y_floor = int(np.floor(y))
            offset_x = x - x_floor
            offset_y = y - y_floor

            # 生成星点
            star = np.zeros_like(psf_kernel)
            star[psf_kernel.shape[0] // 2, psf_kernel.shape[1] // 2] = flux

            # PSF卷积
            star_conv = convolve2d(star, psf_kernel, mode='same')

            # 插入到图像
            x_start = x_floor - psf_kernel.shape[0] // 2
            x_end = x_start + psf_kernel.shape[0]
            y_start = y_floor - psf_kernel.shape[1] // 2
            y_end = y_start + psf_kernel.shape[1]

            # 处理边界
            x_slice = slice(max(0, x_start), min(self.image_size, x_end))
            y_slice = slice(max(0, y_start), min(self.image_size, y_end))

            img_x_start = max(0, -x_start)
            img_x_end = psf_kernel.shape[0] + min(0, self.image_size - x_end)
            img_y_start = max(0, -y_start)
            img_y_end = psf_kernel.shape[1] + min(0, self.image_size - y_end)

            image[x_slice, y_slice] += star_conv[img_x_start:img_x_end, img_y_start:img_y_end]

        # 添加噪声
        image = np.random.poisson(image * self.gain).astype(float) / self.gain
        return image

    def generate_catalog(self, stars):
        """生成星表表格"""
        # 转换像素坐标到天文坐标
        ra, dec = self.wcs.all_pix2world(stars['x'], stars['y'], 0)

        return Table({
            'ID': np.arange(1, len(stars['x']) + 1),
            'RA': ra,
            'DEC': dec,
            'MAG': stars['mag'],
            'FLUX': stars['flux'],
            'X_PIXEL': stars['x'],
            'Y_PIXEL': stars['y']
        })

    def save_to_fits(self, image, catalog, filename):
        """保存为FITS文件"""
        # 主HDU（图像）
        primary_hdu = fits.PrimaryHDU(data=image)
        primary_hdu.header.update(self.wcs.to_header())
        primary_hdu.header['BUNIT'] = ('ADU', 'Data_unit')
        primary_hdu.header['GAIN'] = (self.gain, 'e-/ADU')
        primary_hdu.header['ZEROPNT'] = (self.zeropoint, 'Star_zero')

        # 表格HDU
        table_hdu = fits.BinTableHDU(catalog)
        table_hdu.header['EXTNAME'] = 'CATALOG'

        fits.HDUList([primary_hdu, table_hdu]).writeto(filename, overwrite=True)


if __name__ == '__main__':
    # 示例用法
    sim = AstronomicalSimulator(image_size=2048, pixel_scale=0.3)

    # 生成参数
    stars = sim.generate_stars(num_stars=10000)
    psf = sim.generate_psf(fwhm=3.0, profile='moffat')

    # 生成图像
    image = sim.generate_image(stars, psf, sky_brightness=20.5)

    # 生成星表
    catalog = sim.generate_catalog(stars)

    # 保存文件
    sim.save_to_fits(image, catalog, 'simulated_observation.fits')

