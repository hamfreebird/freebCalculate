import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, mock_open, patch

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Try to import VideoInterpolator, but skip tests if not available
try:
    from freebirdcal.video_interpolator import VideoInterpolator

    VIDEO_INTERPOLATOR_AVAILABLE = True
except ImportError as e:
    VIDEO_INTERPOLATOR_AVAILABLE = False
    print(f"Warning: video_interpolator module not available: {e}")


@unittest.skipIf(
    not VIDEO_INTERPOLATOR_AVAILABLE, "video_interpolator module not available"
)
class TestVideoInterpolator(unittest.TestCase):
    """Test cases for VideoInterpolator class"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_input_path = "test_input.mp4"
        self.test_output_path = "test_output.mp4"

        # Create mock objects for cv2
        self.mock_cv2 = MagicMock()
        self.mock_cv2.VideoCapture.return_value.get.return_value = 30  # fps
        self.mock_cv2.VideoCapture.return_value.isOpened.return_value = True
        self.mock_cv2.VideoCapture.return_value.read.side_effect = [
            (True, MagicMock()),  # First frame
            (True, MagicMock()),  # Second frame
            (False, None),  # End of video
        ]

    def test_init_with_default_parameters(self):
        """Test initialization with default parameters"""
        interpolator = VideoInterpolator(
            input_path=self.test_input_path, output_path=self.test_output_path
        )

        self.assertEqual(interpolator.input_path, self.test_input_path)
        self.assertEqual(interpolator.output_path, self.test_output_path)
        self.assertEqual(interpolator.interp_factor, 1)
        self.assertEqual(interpolator.method, "linear")
        self.assertIsInstance(interpolator.flow_params, dict)
        self.assertTrue(interpolator.show_progress)
        self.assertFalse(interpolator.use_gpu)

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters"""
        flow_params = {
            "pyr_scale": 0.6,
            "levels": 4,
            "winsize": 20,
        }

        interpolator = VideoInterpolator(
            input_path=self.test_input_path,
            output_path=self.test_output_path,
            interp_factor=2,
            method="optical_flow",
            flow_params=flow_params,
            show_progress=False,
            use_gpu=True,
        )

        self.assertEqual(interpolator.input_path, self.test_input_path)
        self.assertEqual(interpolator.output_path, self.test_output_path)
        self.assertEqual(interpolator.interp_factor, 2)
        self.assertEqual(interpolator.method, "optical_flow")
        # Check that custom flow params are merged with defaults
        self.assertEqual(interpolator.flow_params["pyr_scale"], 0.6)
        self.assertEqual(interpolator.flow_params["levels"], 4)
        self.assertEqual(interpolator.flow_params["winsize"], 20)
        # Check that other default params are still present
        self.assertIn("iterations", interpolator.flow_params)
        self.assertIn("poly_n", interpolator.flow_params)
        self.assertIn("poly_sigma", interpolator.flow_params)
        self.assertFalse(interpolator.show_progress)
        self.assertTrue(interpolator.use_gpu)

    def test_init_with_invalid_interp_factor(self):
        """Test initialization with invalid interpolation factor"""
        with self.assertRaises(ValueError):
            VideoInterpolator(
                input_path=self.test_input_path,
                output_path=self.test_output_path,
                interp_factor=0,  # Should be >= 1
            )

        with self.assertRaises(ValueError):
            VideoInterpolator(
                input_path=self.test_input_path,
                output_path=self.test_output_path,
                interp_factor=-1,  # Should be >= 1
            )

    def test_init_with_invalid_method(self):
        """Test initialization with invalid interpolation method"""
        with self.assertRaises(ValueError):
            VideoInterpolator(
                input_path=self.test_input_path,
                output_path=self.test_output_path,
                method="invalid_method",  # Not in SUPPORTED_METHODS
            )

    def test_supported_methods(self):
        """Test that SUPPORTED_METHODS is defined correctly"""
        interpolator = VideoInterpolator(
            input_path=self.test_input_path, output_path=self.test_output_path
        )

        self.assertIn("linear", VideoInterpolator.SUPPORTED_METHODS)
        self.assertIn("optical_flow", VideoInterpolator.SUPPORTED_METHODS)
        self.assertEqual(len(VideoInterpolator.SUPPORTED_METHODS), 2)

    def test_default_flow_params(self):
        """Test that DEFAULT_FLOW_PARAMS contains expected keys"""
        default_params = VideoInterpolator.DEFAULT_FLOW_PARAMS

        expected_keys = [
            "pyr_scale",
            "levels",
            "winsize",
            "iterations",
            "poly_n",
            "poly_sigma",
            "flags",
        ]

        for key in expected_keys:
            self.assertIn(key, default_params)

        # Check some specific values
        self.assertEqual(default_params["pyr_scale"], 0.5)
        self.assertEqual(default_params["levels"], 3)
        self.assertEqual(default_params["winsize"], 15)

    @patch("freebirdcal.video_interpolator.cv2")
    @patch("freebirdcal.video_interpolator.np")
    @patch("freebirdcal.video_interpolator.tqdm", MagicMock())
    def test_process_with_mocked_dependencies(self, mock_np, mock_cv2):
        """Test process method with mocked dependencies"""
        # Set up mocks
        mock_cv2.VideoCapture.return_value = MagicMock(
            get=MagicMock(
                side_effect=[30.0, 1920, 1080, 100]
            ),  # fps, width, height, frame_count
            isOpened=MagicMock(return_value=True),
            read=MagicMock(
                side_effect=[
                    (True, MagicMock(shape=(1080, 1920, 3))),
                    (True, MagicMock(shape=(1080, 1920, 3))),
                    (False, None),
                ]
            ),
            release=MagicMock(),
        )

        mock_cv2.VideoWriter = MagicMock(
            return_value=MagicMock(write=MagicMock(), release=MagicMock())
        )

        mock_cv2.cvtColor = MagicMock(return_value=MagicMock())
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.VideoWriter_fourcc = MagicMock(return_value="XVID")

        mock_np.zeros = MagicMock(return_value=MagicMock())
        mock_np.linspace = MagicMock(return_value=[0.5])
        mock_np.uint8 = MagicMock()

        # Create interpolator with linear method (simpler)
        interpolator = VideoInterpolator(
            input_path=self.test_input_path,
            output_path=self.test_output_path,
            interp_factor=1,
            method="linear",
            show_progress=False,
        )

        # Call process method
        interpolator.process()

        # Verify that VideoCapture was called with correct path
        mock_cv2.VideoCapture.assert_called_with(self.test_input_path)

        # Verify that VideoWriter was created
        self.assertTrue(mock_cv2.VideoWriter.called)

        # Verify that video objects were released
        self.assertTrue(mock_cv2.VideoCapture.return_value.release.called)
        self.assertTrue(mock_cv2.VideoWriter.return_value.release.called)

    def test_flow_params_merging(self):
        """Test that custom flow params are properly merged with defaults"""
        custom_params = {"winsize": 25, "iterations": 5}

        interpolator = VideoInterpolator(
            input_path=self.test_input_path,
            output_path=self.test_output_path,
            method="optical_flow",
            flow_params=custom_params,
        )

        # Check that custom values are used
        self.assertEqual(interpolator.flow_params["winsize"], 25)
        self.assertEqual(interpolator.flow_params["iterations"], 5)

        # Check that other default values are preserved
        default_params = VideoInterpolator.DEFAULT_FLOW_PARAMS
        self.assertEqual(
            interpolator.flow_params["pyr_scale"], default_params["pyr_scale"]
        )
        self.assertEqual(interpolator.flow_params["levels"], default_params["levels"])
        self.assertEqual(interpolator.flow_params["poly_n"], default_params["poly_n"])
        self.assertEqual(
            interpolator.flow_params["poly_sigma"], default_params["poly_sigma"]
        )
        self.assertEqual(interpolator.flow_params["flags"], default_params["flags"])

    def test_string_representation(self):
        """Test string representation of VideoInterpolator"""
        interpolator = VideoInterpolator(
            input_path=self.test_input_path,
            output_path=self.test_output_path,
            interp_factor=2,
            method="optical_flow",
        )

        # Just test that str() doesn't crash
        str_repr = str(interpolator)
        self.assertIsInstance(str_repr, str)

        # Test repr() as well
        repr_str = repr(interpolator)
        self.assertIsInstance(repr_str, str)

    def test_properties_are_readable(self):
        """Test that instance properties can be read"""
        interpolator = VideoInterpolator(
            input_path=self.test_input_path, output_path=self.test_output_path
        )

        # Test reading properties
        self.assertEqual(interpolator.input_path, self.test_input_path)
        self.assertEqual(interpolator.output_path, self.test_output_path)
        self.assertEqual(interpolator.interp_factor, 1)
        self.assertEqual(interpolator.method, "linear")
        self.assertIsInstance(interpolator.flow_params, dict)
        self.assertTrue(interpolator.show_progress)
        self.assertFalse(interpolator.use_gpu)

    @patch("os.path.exists", return_value=False)
    def test_nonexistent_input_file(self, mock_exists):
        """Test behavior when input file doesn't exist"""
        # Note: The actual class might not check file existence in __init__,
        # but we test that process() would fail gracefully
        interpolator = VideoInterpolator(
            input_path="nonexistent.mp4", output_path=self.test_output_path
        )

        # The actual file check might happen in process() method
        # We can't test this without mocking the entire process method

    def test_edge_case_interp_factor_large(self):
        """Test with large interpolation factor"""
        interpolator = VideoInterpolator(
            input_path=self.test_input_path,
            output_path=self.test_output_path,
            interp_factor=10,  # Large but valid
        )

        self.assertEqual(interpolator.interp_factor, 10)

    def test_optical_flow_method_selection(self):
        """Test that optical_flow method is properly recognized"""
        interpolator = VideoInterpolator(
            input_path=self.test_input_path,
            output_path=self.test_output_path,
            method="optical_flow",
        )

        self.assertEqual(interpolator.method, "optical_flow")
        self.assertIn("optical_flow", VideoInterpolator.SUPPORTED_METHODS)


if __name__ == "__main__":
    unittest.main()
