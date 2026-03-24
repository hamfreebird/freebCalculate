"""
Freeelement - Chemical calculation and compound management module

This subpackage provides tools for:
- Chemical element data and atomic weights
- Chemical compound generation and validation
- Uranium compound management
- Chemical formula parsing and analysis
"""

__version__ = "0.1.1"
__author__ = "freebird"
__license__ = "GPLv3"

# Import element_data module for chemical data
try:
    from . import element_data
except ImportError:
    element_data = None

# Import element_generate classes and functions
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
    is_chemical_formula_valid = None
    standardize_formula = None

# Import element_manager classes
try:
    from .element_manager import UraniumCompoundManager
except ImportError:
    UraniumCompoundManager = None

# List of all available modules and functions
__all__ = [
    # Modules
    "element_data",
    # Classes from element_generate
    "ElementCompoundGenerate",
    "ThreeElementCompoundGenerate",
    # Functions from element_generate
    "is_chemical_formula_valid",
    "standardize_formula",
    # Classes from element_manager
    "UraniumCompoundManager",
]

# Re-export commonly used element_data attributes if available
try:
    from .element_data import atomic_weights

    __all__.append("atomic_weights")
except ImportError:
    atomic_weights = None

# Optional: Export ElementCompoundManager if needed (abstract base class)
try:
    from .element_manager import ElementCompoundManager

    __all__.append("ElementCompoundManager")
except ImportError:
    ElementCompoundManager = None
