"""
FreebirdCal - A comprehensive scientific calculation library

FreebirdCal provides tools for:
- Astronomy simulation and star field generation
- Physics formula calculations
- Relativistic spacetime calculations
- Equation solving
- Terrain generation and contour mapping
- Video frame interpolation
- Chemical compound management and generation
- Orbital mechanics simulation
- Game NPC management
- Fractal visualization
"""

__version__ = "0.1.0"
__author__ = "freebird"
__license__ = "GPLv3"

# Import main modules to make them available at package level
try:
    from .astro_simulator import AstronomicalSimulator
except ImportError:
    AstronomicalSimulator = None

try:
    # Import commonly used functions from formula_cal
    from .formula_cal import *
except ImportError:
    pass

try:
    from .spacetime_event import SpacetimeEvent, relativistic_velocity_addition
except ImportError:
    SpacetimeEvent = None
    relativistic_velocity_addition = None

try:
    from .equation_solver import EquationSolver
except ImportError:
    EquationSolver = None

try:
    from .number_operations import NumberOperations
except ImportError:
    NumberOperations = None

try:
    from .spacetime_coordinate import SpacetimeCoordinateSystem
except ImportError:
    SpacetimeCoordinateSystem = None

try:
    from .contour_map import VirtualContourMapGenerator
except ImportError:
    VirtualContourMapGenerator = None

try:
    from .video_interpolator import VideoInterpolator
except ImportError:
    VideoInterpolator = None

try:
    from .npc_manager import BaseNPC, Position
except ImportError:
    BaseNPC = None
    Position = None

try:
    from .orbital_dynamics import OrbitalDynamics
except ImportError:
    OrbitalDynamics = None

try:
    from .element_manager import UraniumCompoundManager
except ImportError:
    UraniumCompoundManager = None

try:
    from .element_generate import (
        ElementCompoundGenerate,
        ThreeElementCompoundGenerate,
        is_chemical_formula_valid,
        standardize_formula,
    )
except ImportError:
    ElementCompoundGenerate = None
    ThreeElementCompoundGenerate = None
    standardize_formula = None
    is_chemical_formula_valid = None

try:
    from .terrain_generator import TerrainGenerator
except ImportError:
    TerrainGenerator = None

try:
    from .zen_fractal import ZenFractal
except ImportError:
    ZenFractal = None

# Import element_data for chemical data
try:
    from . import element_data
except ImportError:
    element_data = None

# List of all available modules and functions
__all__ = [
    # Core modules
    "AstronomicalSimulator",
    "SpacetimeEvent",
    "relativistic_velocity_addition",
    "EquationSolver",
    "NumberOperations",
    "SpacetimeCoordinateSystem",
    "VirtualContourMapGenerator",
    "VideoInterpolator",
    "BaseNPC",
    "Position",
    "OrbitalDynamics",
    "UraniumCompoundManager",
    "ElementCompoundGenerate",
    "ThreeElementCompoundGenerate",
    "standardize_formula",
    "is_chemical_formula_valid",
    "TerrainGenerator",
    "ZenFractal",
    "element_data",
]

# Re-export commonly used formula_cal functions if available
try:
    from .formula_cal import (
        CONSTANTS,
        biot_savart,
        escape_velocity,
        projectile_motion,
        snells_law,
    )

    __all__.extend(
        [
            "escape_velocity",
            "biot_savart",
            "projectile_motion",
            "snells_law",
            "CONSTANTS",
        ]
    )
except ImportError:
    pass
