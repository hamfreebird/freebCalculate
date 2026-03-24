"""
Tests for astro_simulator module
"""

import io
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from freebirdcal.freeastro.astro_simulator import (
    AstronomicalSimulator,
    FITSReader,
    create_simple_simulator,
)


class TestAstronomicalSimulator(unittest.TestCase):
    """Test cases for AstronomicalSimulator class"""

    def setUp(self):
        """Set up test fixture"""
        # Set random seed for reproducibility
        np.random.seed(42)

    def test_initialization_default(self):
        """Test initialization with default parameters"""
        sim = AstronomicalSimulator()

        # Check default values
        self.assertEqual(sim.image_size, 1024)
        self.assertEqual(sim.pixel_scale, 0.2)
        self.assertEqual(sim.zeropoint, 25.0)
        self.assertEqual(sim.gain, 2.0)
        self.assertAlmostEqual(sim.exposure_time, 1.0)
        self.assertAlmostEqual(sim.ra_center, 180.0)
        self.assertAlmostEqual(sim.dec_center, 0.0)
        self.assertEqual(sim.wcs_projection, "TAN")

    def test_initialization_custom(self):
        """Test initialization with custom parameters"""
        sim = AstronomicalSimulator(
            image_size=512,
            pixel_scale=0.4,
            zeropoint=22.5,
            gain=1.8,
            exposure_time=30.0,
            ra_center=90.0,
            dec_center=45.0,
            wcs_projection="SIN",
        )

        self.assertEqual(sim.image_size, 512)
        self.assertEqual(sim.pixel_scale, 0.4)
        self.assertEqual(sim.zeropoint, 22.5)
        self.assertEqual(sim.gain, 1.8)
        self.assertEqual(sim.exposure_time, 30.0)
        self.assertEqual(sim.ra_center, 90.0)
        self.assertEqual(sim.dec_center, 45.0)
        self.assertEqual(sim.wcs_projection, "SIN")

    def test_initialization_invalid(self):
        """Test initialization with invalid parameters"""
        # Test negative image size
        with self.assertRaises(ValueError):
            AstronomicalSimulator(image_size=-100)

        # Test zero pixel scale
        with self.assertRaises(ValueError):
            AstronomicalSimulator(pixel_scale=0)

        # Test negative gain
        with self.assertRaises(ValueError):
            AstronomicalSimulator(gain=-1.0)

        # Test negative exposure time
        with self.assertRaises(ValueError):
            AstronomicalSimulator(exposure_time=-10.0)

    def test_generate_stars_basic(self):
        """Test basic star generation"""
        sim = AstronomicalSimulator(image_size=256)
        stars = sim.generate_stars(num_stars=50, min_mag=18, max_mag=22)

        # Check that stars dictionary has required keys
        required_keys = ["x", "y", "mag", "flux", "flux_total", "ra", "dec"]
        for key in required_keys:
            self.assertIn(key, stars)

        # Check array lengths
        for key in ["x", "y", "mag", "flux", "flux_total", "ra", "dec"]:
            self.assertEqual(len(stars[key]), 50)

        # Check that coordinates are within image bounds
        self.assertTrue(np.all(stars["x"] >= 0))
        self.assertTrue(np.all(stars["x"] < 256))
        self.assertTrue(np.all(stars["y"] >= 0))
        self.assertTrue(np.all(stars["y"] < 256))

        # Check magnitude range
        self.assertTrue(np.all(stars["mag"] >= 18))
        self.assertTrue(np.all(stars["mag"] <= 22))

        # Check that flux is positive
        self.assertTrue(np.all(stars["flux"] > 0))
        self.assertTrue(np.all(stars["flux_total"] > 0))

    def test_generate_stars_clustered(self):
        """Test star generation with clustered distribution"""
        sim = AstronomicalSimulator(image_size=256)
        stars = sim.generate_stars(
            num_stars=30,
            distribution="clustered",
            magnitude_law="powerlaw",
            magnitude_slope=0.33,
        )

        # Should have required keys
        self.assertIn("x", stars)
        self.assertIn("y", stars)
        self.assertEqual(len(stars["x"]), 30)

        # Coordinates should be within bounds
        self.assertTrue(np.all(stars["x"] >= 0))
        self.assertTrue(np.all(stars["x"] < 256))
        self.assertTrue(np.all(stars["y"] >= 0))
        self.assertTrue(np.all(stars["y"] < 256))

    def test_generate_stars_uniform(self):
        """Test star generation with uniform distribution"""
        sim = AstronomicalSimulator(image_size=256)
        stars = sim.generate_stars(
            num_stars=20,
            distribution="uniform",
            magnitude_law="uniform",
        )

        self.assertEqual(len(stars["x"]), 20)

    def test_generate_stars_invalid(self):
        """Test star generation with invalid parameters"""
        sim = AstronomicalSimulator(image_size=256)

        # Test negative number of stars
        with self.assertRaises(ValueError):
            sim.generate_stars(num_stars=-10)

        # Test max_mag < min_mag
        with self.assertRaises(ValueError):
            sim.generate_stars(min_mag=22, max_mag=18)

        # Test invalid distribution
        with self.assertRaises(ValueError):
            sim.generate_stars(distribution="invalid")

        # Test invalid magnitude law
        with self.assertRaises(ValueError):
            sim.generate_stars(magnitude_law="invalid")

    def test_generate_psf_gaussian(self):
        """Test Gaussian PSF generation"""
        sim = AstronomicalSimulator()
        psf = sim.generate_psf(fwhm=3.0, profile="gaussian")

        # PSF should be a 2D array
        self.assertEqual(psf.ndim, 2)

        # PSF should have odd dimensions (center pixel)
        self.assertEqual(psf.shape[0] % 2, 1)
        self.assertEqual(psf.shape[1] % 2, 1)

        # PSF should be normalized (sum ~= 1)
        self.assertAlmostEqual(psf.sum(), 1.0, places=5)

        # PSF should be symmetric
        self.assertTrue(np.allclose(psf, psf.T))

    def test_generate_psf_moffat(self):
        """Test Moffat PSF generation"""
        sim = AstronomicalSimulator()
        psf = sim.generate_psf(fwhm=3.5, profile="moffat", beta=4.5)

        self.assertEqual(psf.ndim, 2)
        self.assertEqual(psf.shape[0] % 2, 1)
        self.assertEqual(psf.shape[1] % 2, 1)
        self.assertAlmostEqual(psf.sum(), 1.0, places=5)

    def test_generate_psf_with_ellipticity(self):
        """Test PSF generation with ellipticity"""
        sim = AstronomicalSimulator()
        psf = sim.generate_psf(
            fwhm=2.5,
            profile="gaussian",
            ellipticity=0.3,
            position_angle=45.0,
        )

        self.assertEqual(psf.ndim, 2)
        self.assertAlmostEqual(psf.sum(), 1.0, places=5)

    def test_generate_psf_invalid(self):
        """Test PSF generation with invalid parameters"""
        sim = AstronomicalSimulator()

        # Test zero FWHM
        with self.assertRaises(ValueError):
            sim.generate_psf(fwhm=0)

        # Test negative FWHM
        with self.assertRaises(ValueError):
            sim.generate_psf(fwhm=-1.0)

        # Test invalid profile
        with self.assertRaises(ValueError):
            sim.generate_psf(profile="invalid")

        # Test invalid beta for Moffat
        with self.assertRaises(ValueError):
            sim.generate_psf(profile="moffat", beta=0.5)  # beta must be > 1

    def test_generate_image_basic(self):
        """Test basic image generation"""
        sim = AstronomicalSimulator(image_size=128)
        stars = sim.generate_stars(num_stars=20, min_mag=18, max_mag=20)
        psf = sim.generate_psf(fwhm=2.5)

        image = sim.generate_image(stars=stars, psf_kernel=psf)

        # Image should be 2D array with correct size
        self.assertEqual(image.ndim, 2)
        self.assertEqual(image.shape, (128, 128))

        # Image should have non-negative values
        self.assertTrue(np.all(image >= 0))

        # Image should have finite values
        self.assertTrue(np.all(np.isfinite(image)))

    def test_generate_image_with_noise(self):
        """Test image generation with noise parameters"""
        sim = AstronomicalSimulator(image_size=128, exposure_time=10.0)
        stars = sim.generate_stars(num_stars=15)
        psf = sim.generate_psf(fwhm=2.0)

        image = sim.generate_image(
            stars=stars,
            psf_kernel=psf,
            sky_brightness=21.0,
            read_noise=3.0,
            dark_current=0.05,
            include_cosmic_rays=False,
        )

        self.assertEqual(image.shape, (128, 128))
        self.assertTrue(np.all(image >= 0))
        self.assertTrue(np.all(np.isfinite(image)))

    def test_generate_image_with_cosmic_rays(self):
        """Test image generation with cosmic rays"""
        sim = AstronomicalSimulator(image_size=128)
        stars = sim.generate_stars(num_stars=10)
        psf = sim.generate_psf(fwhm=2.0)

        image = sim.generate_image(
            stars=stars,
            psf_kernel=psf,
            include_cosmic_rays=True,
            cosmic_ray_rate=0.001,
        )

        self.assertEqual(image.shape, (128, 128))
        self.assertTrue(np.all(image >= 0))

    def test_generate_catalog(self):
        """Test catalog generation"""
        sim = AstronomicalSimulator(image_size=256)
        stars = sim.generate_stars(num_stars=25)

        # Generate catalog without errors
        catalog = sim.generate_catalog(stars, include_errors=False)

        # Should have required columns
        required_cols = [
            "ID",
            "RA",
            "DEC",
            "MAG",
            "FLUX",
            "FLUX_TOTAL",
            "X_PIXEL",
            "Y_PIXEL",
        ]
        for col in required_cols:
            self.assertIn(col, catalog.dtype.names)

        # Should have correct number of rows
        self.assertEqual(len(catalog), 25)

        # Check data types
        self.assertTrue(np.issubdtype(catalog["ID"].dtype, np.integer))
        self.assertTrue(np.issubdtype(catalog["RA"].dtype, np.floating))
        self.assertTrue(np.issubdtype(catalog["DEC"].dtype, np.floating))

        # Generate catalog with errors
        catalog_with_errors = sim.generate_catalog(stars, include_errors=True)
        error_cols = ["MAG_ERR", "FLUX_ERR", "SNR"]
        for col in error_cols:
            self.assertIn(col, catalog_with_errors.dtype.names)

    def test_validate_simulation(self):
        """Test simulation validation"""
        sim = AstronomicalSimulator(image_size=128)
        stars = sim.generate_stars(num_stars=20)
        psf = sim.generate_psf(fwhm=2.5)
        image = sim.generate_image(stars=stars, psf_kernel=psf)

        validation = sim.validate_simulation(stars, image)

        # Should return a dictionary with validation metrics
        self.assertIsInstance(validation, dict)

        # Should have required keys
        required_keys = [
            "num_stars_input",
            "total_flux_input",
            "image_mean",
            "image_std",
            "background_median",
            "background_mad",
            "snr_approx",
        ]
        for key in required_keys:
            self.assertIn(key, validation)

        # Check value ranges
        self.assertEqual(validation["num_stars_input"], 20)
        self.assertGreater(validation["total_flux_input"], 0)
        self.assertGreater(validation["image_mean"], 0)
        self.assertGreaterEqual(validation["image_std"], 0)

    def test_save_to_fits(self):
        """Test saving to FITS file (using temporary file)"""
        sim = AstronomicalSimulator(image_size=64)
        stars = sim.generate_stars(num_stars=10)
        psf = sim.generate_psf(fwhm=2.0)
        image = sim.generate_image(stars=stars, psf_kernel=psf)
        catalog = sim.generate_catalog(stars)

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Save to FITS
            sim.save_to_fits(
                image=image,
                catalog=catalog,
                filename=tmp_path,
                overwrite=True,
                compression=False,
            )

            # Check that file was created and has content
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 0)

        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_create_simple_simulator(self):
        """Test create_simple_simulator function"""
        sim = create_simple_simulator()

        # Should return an AstronomicalSimulator instance
        self.assertIsInstance(sim, AstronomicalSimulator)

        # Should have reasonable default parameters
        self.assertGreater(sim.image_size, 0)
        self.assertGreater(sim.pixel_scale, 0)
        self.assertGreater(sim.exposure_time, 0)


class TestFITSReader(unittest.TestCase):
    """Test cases for FITSReader class"""

    def setUp(self):
        """Set up test fixture"""
        # Create a temporary FITS file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.fits_path = os.path.join(self.temp_dir, "test.fits")

        # Create a simple simulated FITS file using AstronomicalSimulator
        sim = AstronomicalSimulator(image_size=64)
        stars = sim.generate_stars(num_stars=5)
        psf = sim.generate_psf(fwhm=2.0)
        image = sim.generate_image(stars=stars, psf_kernel=psf)
        catalog = sim.generate_catalog(stars)

        # Save to FITS
        sim.save_to_fits(
            image=image,
            catalog=catalog,
            filename=self.fits_path,
            overwrite=True,
            compression=False,
        )

    def tearDown(self):
        """Clean up test fixture"""
        # Remove temporary directory and files
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test FITSReader initialization"""
        reader = FITSReader(self.fits_path)

        # Check that file was loaded
        self.assertIsNotNone(reader.image_hdu)
        self.assertIsNotNone(reader.image_data)

        # Check catalog data (may be None if no catalog)
        # In our test file, there should be a catalog
        self.assertIsNotNone(reader.catalog_data)

        reader.close()

    def test_context_manager(self):
        """Test FITSReader as context manager"""
        with FITSReader(self.fits_path) as reader:
            # Should work without errors
            self.assertIsNotNone(reader.image_data)
            self.assertIsNotNone(reader.image_hdu)

        # Reader should be closed automatically

    def test_get_file_info(self):
        """Test get_file_info method"""
        with FITSReader(self.fits_path) as reader:
            info = reader.get_file_info()

            # Should return a dictionary with file information
            self.assertIsInstance(info, dict)

            # Should have required keys
            required_keys = ["filename", "image_shape", "data_type", "num_hdus"]
            for key in required_keys:
                self.assertIn(key, info)

            # Check values
            self.assertEqual(info["filename"], os.path.basename(self.fits_path))
            self.assertEqual(info["image_shape"], (64, 64))
            self.assertEqual(info["num_hdus"], 2)  # Image + catalog

    def test_get_header_summary(self):
        """Test get_header_summary method"""
        with FITSReader(self.fits_path) as reader:
            summary = reader.get_header_summary(max_keys=10)

            # Should return a string
            self.assertIsInstance(summary, str)

            # Should contain header information
            self.assertIn("NAXIS", summary)
            self.assertIn("WCS", summary)

    def test_get_catalog_summary(self):
        """Test get_catalog_summary method"""
        with FITSReader(self.fits_path) as reader:
            summary = reader.get_catalog_summary(max_rows=5)

            # Should return a string
            self.assertIsInstance(summary, str)

            # Should contain catalog information
            self.assertIn("ID", summary)
            self.assertIn("RA", summary)
            self.assertIn("DEC", summary)

    def test_get_statistics(self):
        """Test get_statistics method"""
        with FITSReader(self.fits_path) as reader:
            stats = reader.get_statistics()

            # Should return a dictionary with statistics
            self.assertIsInstance(stats, dict)

            # Should have required keys
            required_keys = ["min", "max", "mean", "std", "median"]
            for key in required_keys:
                self.assertIn(key, stats)

            # Check value ranges
            self.assertLessEqual(stats["min"], stats["max"])
            self.assertLessEqual(stats["min"], stats["mean"])
            self.assertLessEqual(stats["mean"], stats["max"])
            self.assertGreaterEqual(stats["std"], 0)

    def test_print_summary(self):
        """Test print_summary method (capture output)"""
        with FITSReader(self.fits_path) as reader:
            # Capture stdout
            import sys
            from io import StringIO

            old_stdout = sys.stdout
            sys.stdout = StringIO()

            try:
                reader.print_summary()
                output = sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout

            # Should print summary information
            self.assertIn("FITS File Summary", output)
            self.assertIn("Image", output)
            self.assertIn("Catalog", output)

    @patch("matplotlib.pyplot.show")
    def test_display_image(self, mock_show):
        """Test display_image method (mocked)"""
        with FITSReader(self.fits_path) as reader:
            # Should not raise errors
            reader.display_image(
                figsize=(8, 6),
                cmap="gray",
                stretch="linear",
                percentile=99.5,
                title="Test Image",
            )

            # Should have been called with matplotlib
            # (We can't easily test the actual plot, but we can test it doesn't crash)

    def test_save_as_png(self):
        """Test save_as_png method"""
        output_path = os.path.join(self.temp_dir, "output.png")

        with FITSReader(self.fits_path) as reader:
            reader.save_as_png(output_path, dpi=100)

            # Check that file was created
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

    @patch("matplotlib.pyplot.show")
    def test_plot_catalog(self, mock_show):
        """Test plot_catalog method (mocked)"""
        with FITSReader(self.fits_path) as reader:
            # Should not raise errors
            reader.plot_catalog(
                x_col="X_PIXEL",
                y_col="Y_PIXEL",
                mag_col="MAG",
                show_image=True,
                title="Test Catalog",
            )

            # Test with different columns
            reader.plot_catalog(
                x_col="RA",
                y_col="DEC",
                mag_col="MAG",
                show_image=False,
                title="Sky Coordinates",
            )

    def test_nonexistent_file(self):
        """Test initialization with non-existent file"""
        with self.assertRaises(FileNotFoundError):
            FITSReader("nonexistent_file.fits")


class TestAstroSimulatorIntegration(unittest.TestCase):
    """Integration tests for astro_simulator module"""

    def test_end_to_end_simulation(self):
        """Test complete simulation workflow"""
        # Create simulator
        sim = AstronomicalSimulator(image_size=128)

        # Generate stars
        stars = sim.generate_stars(
            num_stars=30,
            min_mag=18,
            max_mag=22,
            distribution="clustered",
            magnitude_law="powerlaw",
        )

        # Generate PSF
        psf = sim.generate_psf(
            fwhm=2.5,
            profile="moffat",
            beta=3.5,
            ellipticity=0.1,
            position_angle=30.0,
        )

        # Generate image
        image = sim.generate_image(
            stars=stars,
            psf_kernel=psf,
            sky_brightness=21.0,
            read_noise=2.5,
            dark_current=0.02,
            include_cosmic_rays=True,
            cosmic_ray_rate=0.0005,
        )

        # Generate catalog
        catalog = sim.generate_catalog(stars, include_errors=True)

        # Validate simulation
        validation = sim.validate_simulation(stars, image)

        # Check that all steps completed successfully
        self.assertEqual(len(stars["x"]), 30)
        self.assertEqual(psf.ndim, 2)
        self.assertEqual(image.shape, (128, 128))
        self.assertEqual(len(catalog), 30)
        self.assertIsInstance(validation, dict)

        # Verify consistency between stars and catalog
        self.assertTrue(np.allclose(stars["x"], catalog["X_PIXEL"]))
        self.assertTrue(np.allclose(stars["y"], catalog["Y_PIXEL"]))
        self.assertTrue(np.allclose(stars["mag"], catalog["MAG"]))

    def test_multiband_simulation(self):
        """Test multi-band simulation example from docs"""
        sim = AstronomicalSimulator(image_size=256, pixel_scale=0.4)

        # Generate stars (same spatial distribution for all bands)
        stars = sim.generate_stars(num_stars=50)

        # Simulate different bands
        bands = {
            "B": {"psf_fwhm": 2.5, "sky_brightness": 22.5, "zeropoint": 25.0},
            "V": {"psf_fwhm": 2.0, "sky_brightness": 21.5, "zeropoint": 25.0},
            "R": {"psf_fwhm": 1.8, "sky_brightness": 20.5, "zeropoint": 25.0},
        }

        images = {}
        for band, params in bands.items():
            psf = sim.generate_psf(fwhm=params["psf_fwhm"])
            image = sim.generate_image(
                stars, psf, sky_brightness=params["sky_brightness"]
            )
            images[band] = image

        # Check that all images were generated
        self.assertEqual(len(images), 3)
        for band in ["B", "V", "R"]:
            self.assertIn(band, images)
            self.assertEqual(images[band].shape, (256, 256))
            self.assertTrue(np.all(images[band] >= 0))

        # V band should have lower sky background than B band
        self.assertLess(images["V"].mean(), images["B"].mean())

    def test_time_series_simulation(self):
        """Test time series simulation example from docs"""
        sim = AstronomicalSimulator(image_size=128, exposure_time=60.0)

        # Generate base stars
        stars = sim.generate_stars(num_stars=20)

        # Simulate time series
        n_frames = 5
        time_series = []

        for i in range(n_frames):
            # Copy stars to avoid modifying original
            stars_copy = stars.copy()

            # Add random variation to simulate variable stars
            n_variable = 3
            if len(stars_copy["mag"]) > 0:
                variable_indices = np.random.choice(
                    len(stars_copy["mag"]),
                    min(n_variable, len(stars_copy["mag"])),
                    replace=False,
                )
                variation = np.random.uniform(-0.2, 0.2, len(variable_indices))
                stars_copy["mag"][variable_indices] += variation

                # Recalculate flux for variable stars
                stars_copy["flux"][variable_indices] = 10 ** (
                    -0.4 * (stars_copy["mag"][variable_indices] - sim.zeropoint)
                )
                stars_copy["flux_total"][variable_indices] = (
                    stars_copy["flux"][variable_indices] * sim.exposure_time
                )

            # Generate image
            psf = sim.generate_psf()
            image = sim.generate_image(stars_copy, psf)
            time_series.append(image)

        # Check time series
        self.assertEqual(len(time_series), n_frames)
        for frame in time_series:
            self.assertEqual(frame.shape, (128, 128))
            self.assertTrue(np.all(frame >= 0))


if __name__ == "__main__":
    unittest.main()
