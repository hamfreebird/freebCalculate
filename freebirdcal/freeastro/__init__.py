"""
Freeastro - Astronomy, astrophysics and spacetime simulation module

This subpackage provides tools for:
- Astronomical simulation and star field generation
- Interactive telescope simulation
- Orbital dynamics and celestial mechanics
- Spacetime coordinate systems and relativity
- Spacetime event analysis and Lorentz transformations
"""

import logging

__version__ = "0.1.3"
__author__ = "freebird"
__license__ = "GPLv3"

logger = logging.getLogger(__name__)

# Import astro_simulator classes
try:
    from .astro_simulator import AstronomicalSimulator, FITSReader
except ImportError as e:
    AstronomicalSimulator = None
    FITSReader = None
    logger.error(f"Error: {e}")

# Import interactive_telescope classes
try:
    from .interactive_telescope import (
        InteractiveTelescopeSimulator,
        ObservationParameters,
        TelescopeParameters,
    )
except ImportError as e:
    InteractiveTelescopeSimulator = None
    TelescopeParameters = None
    ObservationParameters = None
    logger.error(f"Error: {e}")

# Import orbital_dynamics classes
try:
    from .orbital_dynamics import OrbitalDynamics
except ImportError as e:
    OrbitalDynamics = None
    logger.error(f"Error: {e}")

# Import spacetime_coordinate classes
try:
    from .spacetime_coordinate import SpacetimeCoordinateSystem
except ImportError as e:
    SpacetimeCoordinateSystem = None
    logger.error(f"Error: {e}")

# Import spacetime_event classes and functions
try:
    from .spacetime_event import SpacetimeEvent, relativistic_velocity_addition
except ImportError as e:
    SpacetimeEvent = None
    relativistic_velocity_addition = None
    logger.error(f"Error: {e}")

# List of all available modules and functions
__all__ = [
    # Core astronomy modules
    "AstronomicalSimulator",
    "FITSReader",
    # Interactive telescope
    "InteractiveTelescopeSimulator",
    "TelescopeParameters",
    "ObservationParameters",
    # Orbital mechanics
    "OrbitalDynamics",
    # Spacetime systems
    "SpacetimeCoordinateSystem",
    "SpacetimeEvent",
    "relativistic_velocity_addition",
]

# Optional: Export additional functions from astro_simulator if needed
try:
    from .astro_simulator import generate_star_field

    __all__.append("generate_star_field")
except ImportError:
    pass

# Optional: Export functions from orbital_dynamics if needed
try:
    from .orbital_dynamics import propagate_orbit

    __all__.append("propagate_orbit")
except ImportError:
    pass
