"""
Freeterrain - Terrain generation and contour mapping module

This subpackage provides tools for:
- Virtual contour map generation
- Advanced terrain generation with erosion simulation
- Topographic visualization and analysis
- Geographic data processing and export
- Elevation data analysis and contour extraction from raster data
"""

__version__ = "0.1.3"
__author__ = "freebird"
__license__ = "GPLv3"

# Import contour_map classes
try:
    from .contour_map import VirtualContourMapGenerator
except ImportError:
    VirtualContourMapGenerator = None

# Import terrain_generator classes
try:
    from .terrain_generator import TerrainGenerator
except ImportError:
    TerrainGenerator = None

# Import elevation_analyzer classes
try:
    from .elevation_analyzer import ElevationAnalyzer
except ImportError:
    ElevationAnalyzer = None

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

# Optional: Export additional functions from contour_map if needed
try:
    from .contour_map import generate_contour_geojson

    __all__.append("generate_contour_geojson")
except ImportError:
    pass

# Optional: Export functions from terrain_generator if needed
try:
    from .terrain_generator import generate_heightmap

    __all__.append("generate_heightmap")
except ImportError:
    pass

# Re-export commonly used constants if available
try:
    from .contour_map import logger

    __all__.append("logger")
except ImportError:
    pass
