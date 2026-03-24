"""
Freeastro - Astronomy, astrophysics and spacetime simulation module

This subpackage provides tools for:
- Astronomical simulation and star field generation
- Interactive telescope simulation
- Orbital dynamics and celestial mechanics
- Spacetime coordinate systems and relativity
- Spacetime event analysis and Lorentz transformations
"""

__version__ = "0.1.2"
__author__ = "freebird"
__license__ = "GPLv3"

# Import astro_simulator classes
try:
    from .astro_simulator import AstronomicalSimulator, FITSReader
except ImportError:
    AstronomicalSimulator = None
    FITSReader = None

# Import interactive_telescope classes
try:
    from .interactive_telescope import (
        InteractiveTelescopeSimulator,
        ObservationParameters,
        TelescopeParameters,
    )
except ImportError:
    InteractiveTelescopeSimulator = None
    TelescopeParameters = None
    ObservationParameters = None

# Import orbital_dynamics classes
try:
    from .orbital_dynamics import OrbitalDynamics
except ImportError:
    OrbitalDynamics = None

# Import spacetime_coordinate classes
try:
    from .spacetime_coordinate import SpacetimeCoordinateSystem
except ImportError:
    SpacetimeCoordinateSystem = None

# Import spacetime_event classes and functions
try:
    from .spacetime_event import SpacetimeEvent, relativistic_velocity_addition
except ImportError:
    SpacetimeEvent = None
    relativistic_velocity_addition = None

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
