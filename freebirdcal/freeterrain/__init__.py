"""
Freeterrain - Terrain generation and contour mapping module

This subpackage provides tools for:
- Virtual contour map generation
- Advanced terrain generation with erosion simulation
- Topographic visualization and analysis
- Geographic data processing and export
- Elevation data analysis and contour extraction from raster data
"""

import logging

__version__ = "0.1.3"
__author__ = "freebird"
__license__ = "GPLv3"

# Setup logging for import warnings
logger = logging.getLogger(__name__)

# Import contour_map classes
try:
    from .contour_map import VirtualContourMapGenerator
except ImportError as e:
    VirtualContourMapGenerator = None
    logger.error(f"Error: {e}")

# Import terrain_generator classes
try:
    from .terrain_generator import TerrainGenerator
except ImportError as e:
    TerrainGenerator = None
    logger.error(f"Error: {e}")

# Import elevation_analyzer classes
try:
    from .elevation_analyzer import ElevationAnalyzer
except ImportError as e:
    ElevationAnalyzer = None
    logger.error(f"Error: {e}")

# List of all available modules and functions
__all__ = [
    # Contour mapping
    "VirtualContourMapGenerator",
    # Terrain generation
    "TerrainGenerator",
    # Elevation analysis
    "ElevationAnalyzer",
]

# Optional: Export Landlab availability flag if needed
try:
    from .terrain_generator import LANDLAB_AVAILABLE

    __all__.append("LANDLAB_AVAILABLE")
except ImportError:
    LANDLAB_AVAILABLE = False
    logger.warning(
        "LANDLAB_AVAILABLE flag not available. "
        "Landlab may not be installed. "
        "Try: pip install landlab"
    )

# Re-export commonly used constants if available
try:
    from .contour_map import logger as contour_map_logger

    # Re-export the contour_map logger for convenience
    logger = contour_map_logger
    __all__.append("logger")
except ImportError:
    # Keep the default logger we already set up
    pass
