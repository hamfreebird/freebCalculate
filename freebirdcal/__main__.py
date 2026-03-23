#!/usr/bin/env python3
"""
FreebirdCal Demo - Command-line interface for freebirdcal library
"""

import argparse
import importlib
import sys
from typing import List, Optional


def demo_astronomy() -> None:
    """Demo astronomy simulation features"""
    try:
        import numpy as np

        from freebirdcal.astro_simulator import AstronomicalSimulator

        print("=== Astronomy Simulation Demo ===")
        print("Creating astronomical simulator...")

        sim = AstronomicalSimulator(
            image_size=512, pixel_scale=0.2, zeropoint=25.0, gain=2.0
        )

        stars = sim.generate_stars(num_stars=50, min_mag=18, max_mag=24)
        print(f"Generated {len(stars)} stars")

        psf = sim.generate_psf(fwhm=3.0, profile="moffat")
        print(f"Generated PSF with shape: {psf.shape}")

        image = sim.generate_image(stars, psf, sky_brightness=21.0)
        print(f"Generated image with shape: {image.shape}")

        print("Astronomy demo completed successfully!\n")

    except ImportError as e:
        print(f"Cannot run astronomy demo: {e}")
        print("Install optional astronomy dependencies:")
        print("  pip install freebirdcal[astronomy]\n")


def demo_physics() -> None:
    """Demo physics calculation features"""
    try:
        from freebirdcal.formula_cal import escape_velocity
        from freebirdcal.spacetime_event import SpacetimeEvent

        print("=== Physics Calculation Demo ===")

        # Escape velocity example
        earth_mass = 5.97237e24  # kg
        earth_radius = 6.3781e6  # m
        v_escape = escape_velocity(earth_mass, earth_radius)
        print(f"Earth escape velocity: {v_escape:.2f} m/s")

        # Spacetime event example
        print("\nSpacetime event example:")
        event1 = SpacetimeEvent(0, 0, 0)
        event2 = event1.move(1, 0, 5)  # Photon moving at light speed for 5 seconds
        print(f"Photon position after 5 seconds: {event2}")
        print(f"Interval type: {event1.interval_type(event2)}")

        print("Physics demo completed successfully!\n")

    except ImportError as e:
        print(f"Cannot run physics demo: {e}")
        print("Install required dependencies: numpy, scipy, matplotlib\n")


def demo_chemistry() -> None:
    """Demo chemistry features"""
    try:
        from freebirdcal.element_manager import UraniumCompoundManager

        print("=== Chemistry Demo ===")
        print("Uranium compound manager example...")

        uranium_mgr = UraniumCompoundManager("U")

        # Add a sample compound
        uranium_mgr.add_compound(
            name="Uranium Hexafluoride",
            formula="UF6",
            oxidation_states=[6],
            phase="Gas",
            uses=["Uranium enrichment"],
        )

        # Find the compound
        compound = uranium_mgr.find_by_formula("UF6")
        if compound:
            print(f"Found compound: {compound}")

        # Get nuclear properties
        nuclear_report = uranium_mgr.get_nuclear_properties()
        print(f"Nuclear properties report available: {len(nuclear_report) > 0}")

        print("Chemistry demo completed successfully!\n")

    except ImportError as e:
        print(f"Cannot run chemistry demo: {e}")
        print("Install optional chemistry dependencies:")
        print("  pip install freebirdcal[chemistry]\n")


def demo_terrain() -> None:
    """Demo terrain generation features"""
    try:
        from freebirdcal.terrain_generator import TerrainGenerator

        print("=== Terrain Generation Demo ===")
        print("Initializing terrain generator...")

        gen = TerrainGenerator(
            shape=(257, 257),  # Smaller resolution for demo
            dx=10.0,
            seed=42,
        )

        # Generate initial topography
        gen.initial_topography(scale=300, octaves=5)
        print("Generated initial topography")

        # Add micro details
        gen.add_micro_details(amplitude=15, scale=50)
        print("Added micro details")

        print("Terrain demo completed successfully!")
        print("Note: Full erosion simulation requires landlab package\n")

    except ImportError as e:
        print(f"Cannot run terrain demo: {e}")
        print("Install optional terrain dependencies:")
        print("  pip install freebirdcal[terrain]\n")


def demo_video() -> None:
    """Demo video processing features"""
    try:
        from freebirdcal.video_interpolator import VideoInterpolator

        print("=== Video Processing Demo ===")
        print("Video interpolator class is available")
        print("To use: from freebirdcal.video_interpolator import VideoInterpolator")
        print("\nExample usage:")
        print("interpolator = VideoInterpolator(")
        print("    input_path='input.mp4',")
        print("    output_path='output.mp4',")
        print("    interp_factor=2,")
        print("    method='optical_flow',")
        print("    use_gpu=True")
        print(")")
        print("interpolator.process()")
        print("\nInstall optional video dependencies:")
        print("  pip install freebirdcal[video]\n")

    except ImportError as e:
        print(f"Cannot run video demo: {e}")
        print("Install optional video dependencies:")
        print("  pip install freebirdcal[video]\n")


def list_modules() -> None:
    """List all available modules"""
    print("=== Available Modules ===")
    modules = [
        ("astro_simulator", "Astronomy simulation and star field generation"),
        ("formula_cal", "Physics formula calculations"),
        ("spacetime_event", "Relativistic spacetime calculations"),
        ("equation_solver", "Equation solving utilities"),
        ("number_operations", "Number and mathematical operations"),
        ("spacetime_coordinate", "4D spacetime coordinate system"),
        ("contour_map", "Contour map and terrain visualization"),
        ("video_interpolator", "Video frame interpolation"),
        ("npc_manager", "Game NPC management system"),
        ("orbital_dynamics", "Orbital mechanics simulation"),
        ("element_manager", "Chemical compound management"),
        ("element_generate", "Chemical compound generation"),
        ("terrain_generator", "Advanced terrain generation"),
    ]

    for mod_name, description in modules:
        print(f"{mod_name:<25} - {description}")


def main() -> None:
    """Main entry point for freebirdcal demo"""
    parser = argparse.ArgumentParser(
        description="FreebirdCal - Scientific calculation library demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                Run all demos
  %(prog)s --astronomy          Run astronomy demo
  %(prog)s --physics            Run physics demo
  %(prog)s --modules            List all available modules
        """,
    )

    parser.add_argument("--all", action="store_true", help="Run all available demos")
    parser.add_argument(
        "--astronomy", action="store_true", help="Run astronomy simulation demo"
    )
    parser.add_argument(
        "--physics", action="store_true", help="Run physics calculation demo"
    )
    parser.add_argument("--chemistry", action="store_true", help="Run chemistry demo")
    parser.add_argument(
        "--terrain", action="store_true", help="Run terrain generation demo"
    )
    parser.add_argument(
        "--video", action="store_true", help="Run video processing demo"
    )
    parser.add_argument(
        "--modules", action="store_true", help="List all available modules"
    )

    args = parser.parse_args()

    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        return

    print("FreebirdCal Demo\n" + "=" * 50)

    # Run selected demos
    if args.all:
        demo_astronomy()
        demo_physics()
        demo_chemistry()
        demo_terrain()
        demo_video()
        list_modules()
    else:
        if args.astronomy:
            demo_astronomy()
        if args.physics:
            demo_physics()
        if args.chemistry:
            demo_chemistry()
        if args.terrain:
            demo_terrain()
        if args.video:
            demo_video()
        if args.modules:
            list_modules()


if __name__ == "__main__":
    main()
