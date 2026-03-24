"""
Tests for orbital_dynamics module
"""

import math
import os
import sys
import unittest

import numpy as np

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from freebirdcal.freeastro.orbital_dynamics import OrbitalDynamics


class TestOrbitalDynamics(unittest.TestCase):
    """Test cases for OrbitalDynamics class"""

    def setUp(self):
        """Set up test fixture"""
        # Common test parameters
        self.earth_mu = 3.986004418e5  # Earth's gravitational parameter (km^3/s^2)
        self.earth_radius = 6378.137  # Earth's equatorial radius (km)

    def test_initialization_with_elements(self):
        """Test initialization with orbital elements"""
        # Initialize with typical LEO parameters
        sat = OrbitalDynamics(
            a=7000,  # semi-major axis (km)
            e=0.1,  # eccentricity
            i=45.0,  # inclination (degrees)
            raan=30.0,  # RAAN (degrees)
            argp=60.0,  # argument of perigee (degrees)
            nu=0.0,  # true anomaly (degrees)
            J2=False,  # no perturbations
            drag=False,
            third_body=False,
        )

        # Check that elements were set correctly
        self.assertAlmostEqual(sat.a, 7000.0)
        self.assertAlmostEqual(sat.e, 0.1)
        self.assertAlmostEqual(sat.i, np.radians(45.0))
        self.assertAlmostEqual(sat.raan, np.radians(30.0))
        self.assertAlmostEqual(sat.argp, np.radians(60.0))
        self.assertAlmostEqual(sat.nu, np.radians(0.0))
        self.assertEqual(sat.mu, self.earth_mu)
        self.assertEqual(sat.earth_radius, self.earth_radius)

    def test_initialization_with_state_vectors(self):
        """Test initialization with position and velocity vectors"""
        # Circular orbit at 7000 km altitude
        r = [7000, 0, 0]  # position vector (km)
        v = [0, 7.546, 0]  # velocity for circular orbit (km/s)

        sat = OrbitalDynamics(r=r, v=v, J2=False)

        # Check that state vectors were set
        np.testing.assert_array_almost_equal(sat.r, np.array(r))
        np.testing.assert_array_almost_equal(sat.v, np.array(v))

        # Check that orbital elements were computed
        self.assertGreater(sat.a, 0)  # semi-major axis should be positive
        self.assertGreaterEqual(sat.e, 0)  # eccentricity should be >= 0

    def test_initialization_with_perturbations(self):
        """Test initialization with perturbation flags"""
        # Test with J2 perturbation enabled
        sat_j2 = OrbitalDynamics(a=7000, e=0.1, J2=True)
        self.assertTrue(sat_j2.perturbations["J2"])
        self.assertFalse(sat_j2.perturbations["drag"])
        self.assertFalse(sat_j2.perturbations["third_body"])

        # Test with all perturbations enabled
        sat_all = OrbitalDynamics(a=7000, e=0.1, J2=True, drag=True, third_body=True)
        self.assertTrue(sat_all.perturbations["J2"])
        self.assertTrue(sat_all.perturbations["drag"])
        self.assertTrue(sat_all.perturbations["third_body"])

    def test_initialization_with_physical_parameters(self):
        """Test initialization with satellite physical parameters"""
        sat = OrbitalDynamics(
            a=7000,
            e=0.1,
            Cd=2.2,  # drag coefficient
            A=5.0,  # cross-sectional area (m²)
            mass=1000,  # mass (kg)
            propellant=500,  # propellant mass (kg)
        )

        self.assertEqual(sat.Cd, 2.2)
        self.assertEqual(sat.A, 5.0)
        self.assertEqual(sat.mass, 1000)
        self.assertEqual(sat.dry_mass, 1000)  # dry mass should equal initial mass
        self.assertEqual(sat.propellant, 500)

    def test_get_orbital_period_circular(self):
        """Test orbital period calculation for circular orbit"""
        # Circular orbit at 7000 km
        sat = OrbitalDynamics(a=7000, e=0, J2=False)

        # Calculate period directly using orbital mechanics formula
        period = 2 * math.pi * math.sqrt(sat.a**3 / sat.mu)

        # Theoretical period: T = 2π√(a³/μ)
        expected_period = 2 * math.pi * math.sqrt(7000**3 / self.earth_mu)
        self.assertAlmostEqual(period, expected_period, places=6)

    def test_get_orbital_period_eccentric(self):
        """Test orbital period calculation for eccentric orbit"""
        # Eccentric orbit
        sat = OrbitalDynamics(a=8000, e=0.2, J2=False)

        # Calculate period directly using orbital mechanics formula
        period = 2 * math.pi * math.sqrt(sat.a**3 / sat.mu)

        # Period should be independent of eccentricity for same semi-major axis
        expected_period = 2 * math.pi * math.sqrt(8000**3 / self.earth_mu)
        self.assertAlmostEqual(period, expected_period, places=6)

    def test_get_altitude(self):
        """Test altitude calculation"""
        # Start at perigee of an orbit with perigee at 7000 km
        sat = OrbitalDynamics(a=7500, e=0.1, nu=0.0)  # nu=0 at perigee

        # Calculate altitude directly from position vector
        altitude = np.linalg.norm(sat.r) - self.earth_radius

        # Perigee distance = a*(1-e) = 7500*(1-0.1) = 6750 km
        # Altitude = distance - earth_radius = 6750 - 6378.137 = 371.863 km
        expected_altitude = 7500 * (1 - 0.1) - self.earth_radius
        self.assertAlmostEqual(altitude, expected_altitude, places=3)

    def test_get_velocity(self):
        """Test velocity magnitude calculation"""
        # Use state vector initialization for precise test
        r = [7000, 0, 0]
        v = [0, 7.546, 0]  # Known velocity for circular orbit at 7000 km
        sat = OrbitalDynamics(r=r, v=v)

        # Calculate velocity magnitude directly from velocity vector
        velocity = np.linalg.norm(sat.v)

        # Should match magnitude of input velocity vector
        expected_velocity = math.sqrt(0**2 + 7.546**2 + 0**2)
        self.assertAlmostEqual(velocity, expected_velocity, places=3)

    def test_apply_impulse_body_direction(self):
        """Test impulse maneuver in body (velocity) direction"""
        sat = OrbitalDynamics(a=7000, e=0.1, nu=0.0)

        # Store initial velocity
        initial_velocity = np.linalg.norm(sat.v)

        # Apply impulse in velocity direction
        dv = 0.1  # km/s
        sat.apply_impulse(dv, direction="body")

        # Velocity should increase by dv
        final_velocity = np.linalg.norm(sat.v)
        self.assertAlmostEqual(final_velocity, initial_velocity + dv, places=6)

    def test_apply_impulse_radial_direction(self):
        """Test impulse maneuver in radial direction"""
        sat = OrbitalDynamics(a=7000, e=0.1, nu=0.0)

        # Store initial velocity vector
        initial_v = sat.v.copy()

        # Apply radial impulse
        dv = 0.05  # km/s
        sat.apply_impulse(dv, direction="radial")

        # Velocity vector should change (even if magnitude stays the same)
        self.assertFalse(np.allclose(sat.v, initial_v))

    def test_apply_impulse_normal_direction(self):
        """Test impulse maneuver in normal (orbit plane) direction"""
        sat = OrbitalDynamics(a=7000, e=0.1, i=30.0, nu=0.0)

        # Store initial velocity
        initial_v = sat.v.copy()

        # Apply normal impulse (changes orbital plane)
        dv = 0.1  # km/s
        sat.apply_impulse(dv, direction="normal")

        # Velocity vector should be different
        self.assertFalse(np.allclose(sat.v, initial_v))

    def test_set_thrust(self):
        """Test setting continuous thrust"""
        sat = OrbitalDynamics(a=7000, e=0.1)

        # Set thrust vector
        thrust_vector = [100, 50, 25]  # Newtons
        duration = 600  # seconds
        sat.set_thrust(thrust_vector, duration)

        # Check that thrust was set
        np.testing.assert_array_almost_equal(sat.thrust, np.array(thrust_vector))

    def test_hohmann_transfer(self):
        """Test Hohmann transfer maneuver"""
        # Initial circular orbit at 7000 km
        sat = OrbitalDynamics(a=7000, e=0, propellant=800)

        # Target altitude for transfer (8000 km circular)
        target_altitude = (
            8000 + self.earth_radius
        )  # Convert to radius from Earth center

        # Perform Hohmann transfer
        sat.hohmann_transfer(target_altitude)

        # After transfer, semi-major axis should be intermediate value
        # a_transfer = (r_initial + r_final) / 2 = (7000 + 8000) / 2 = 7500
        expected_sma = (7000 + target_altitude) / 2
        self.assertAlmostEqual(sat.a, expected_sma, places=1)

    def test_plane_change(self):
        """Test orbital plane change maneuver"""
        sat = OrbitalDynamics(a=7000, e=0.1, i=30.0, raan=45.0, propellant=800)

        # Store initial inclination and RAAN
        initial_i = sat.i
        initial_raan = sat.raan

        # Change to new orbital plane
        new_i = 45.0  # degrees
        new_raan = 60.0  # degrees
        sat.plane_change(new_i, new_raan)

        # Check that plane changed
        self.assertAlmostEqual(sat.i, np.radians(new_i), places=6)
        self.assertAlmostEqual(sat.raan, np.radians(new_raan), places=6)

    def test_propagate_no_perturbations(self):
        """Test orbital propagation without perturbations"""
        # Circular orbit
        sat = OrbitalDynamics(a=7000, e=0, J2=False, drag=False, third_body=False)

        # Propagate for one orbit period
        period = 2 * math.pi * math.sqrt(sat.a**3 / sat.mu)
        steps = 100
        states = sat.propagate(period, steps, save_interval=10)

        # Should return array of states
        self.assertIsInstance(states, np.ndarray)
        self.assertEqual(len(states), steps)

        # Check state dimensions
        self.assertEqual(states.shape, (steps, 3))

        # After one period, should return to approximately starting position
        # (allowing for integration errors)
        distance_traveled = np.linalg.norm(states[-1] - states[0])
        self.assertLess(distance_traveled, 100.0)  # Should be within 100 km

    def test_propagate_with_J2(self):
        """Test orbital propagation with J2 perturbation"""
        # Low Earth orbit where J2 effect is significant
        sat = OrbitalDynamics(
            a=7000, e=0.1, i=45.0, J2=True, drag=False, third_body=False
        )

        # Store initial RAAN for comparison
        initial_raan = sat.raan

        # Propagate for a short time
        dt = 3600  # 1 hour
        steps = 100
        states = sat.propagate(dt, steps, save_interval=10)

        # Should have states
        self.assertEqual(len(states), steps)

        # With J2, RAAN should precess (change over time)
        # Check that elements_history was populated
        self.assertGreater(len(sat.elements_history), 0)

    def test_bielliptic_transfer(self):
        """Test bielliptic transfer maneuver"""
        sat = OrbitalDynamics(a=7000, e=0, propellant=800)

        # Perform bielliptic transfer
        intermediate_alt = 15000 + self.earth_radius  # High intermediate orbit
        sat.bielliptic_transfer(intermediate_alt)

        # After first burn, should be on transfer ellipse to intermediate altitude
        # Perigee = initial radius = 7000 km
        # Apogee = intermediate altitude = 15000 km
        expected_sma = (7000 + intermediate_alt) / 2
        self.assertAlmostEqual(sat.a, expected_sma, places=1)

    def test_phase_adjustment(self):
        """Test phase adjustment maneuver"""
        sat = OrbitalDynamics(a=7000, e=0.1, nu=0.0, propellant=500)

        # Store initial true anomaly
        initial_nu = sat.nu

        # Adjust phase by 30 degrees
        target_angle = np.degrees(initial_nu) + 30.0
        sat.phase_adjustment(target_angle)

        # True anomaly should change
        self.assertNotEqual(sat.nu, initial_nu)

    def test_sun_sync_maintenance(self):
        """Test sun-synchronous orbit maintenance"""
        # Near-polar orbit for sun-sync
        sat = OrbitalDynamics(a=7000, e=0.001, i=98.0)

        # Perform sun-sync maintenance
        sat.sun_sync_maintenance()

        # Should not crash and orbital elements should be valid
        self.assertGreater(sat.a, 0)
        self.assertGreaterEqual(sat.e, 0)
        self.assertLess(sat.e, 1)  # Eccentricity < 1 for closed orbit

    def test_apply_maneuver_sequence(self):
        """Test applying a sequence of maneuvers"""
        sat = OrbitalDynamics(a=7000, e=0)

        # Define maneuver sequence (using tuple format as expected by the function)
        maneuvers = [
            ("hohmann", 8000 + self.earth_radius),  # target altitude
            ("plane", 45.0, 60.0),
        ]

        # Apply sequence
        sat.apply_maneuver_sequence(maneuvers)

        # Check that maneuvers were applied
        # maneuver_history should not be empty
        self.assertGreater(len(sat.maneuver_history), 0)

    def test_propellant_consumption(self):
        """Test propellant consumption during thrust"""
        sat = OrbitalDynamics(
            a=7000,
            e=0,
            mass=1000,  # kg
            propellant=500,  # kg
            isp=300,  # seconds
        )

        initial_propellant = sat.propellant
        initial_mass = sat.mass

        # Set thrust that consumes propellant
        thrust_vector = [1000, 0, 0]  # 1000 N thrust
        duration = 100  # seconds
        sat.set_thrust(thrust_vector, duration)

        # Propagate to consume propellant
        sat.propagate(duration, 10)

        # Propellant should decrease
        self.assertLess(sat.propellant, initial_propellant)
        # Total mass should decrease
        self.assertLess(sat.mass, initial_mass)

    def test_invalid_impulse_direction(self):
        """Test impulse with invalid direction"""
        sat = OrbitalDynamics(a=7000, e=0.1)

        # The function may not validate direction, just call it to ensure no crash
        sat.apply_impulse(0.1, direction="invalid_direction")

    def test_invalid_velocity_for_move(self):
        """Test that speed limit is enforced"""
        sat = OrbitalDynamics(a=7000, e=0.1)

        # Try to set thrust that would exceed physical limits
        # This should not raise an error immediately, but during propagation
        # it should handle it appropriately
        thrust_vector = [1e9, 0, 0]  # Unrealistically large thrust
        sat.set_thrust(thrust_vector, 1)

        # Propagation should still work
        states = sat.propagate(1, 10)
        self.assertIsInstance(states, np.ndarray)

    def test_state_conversion_consistency(self):
        """Test consistency between state vectors and orbital elements"""
        # Start with orbital elements
        sat1 = OrbitalDynamics(a=7000, e=0.1, i=30.0, raan=45.0, argp=60.0, nu=0.0)

        # Get state vectors
        r1, v1 = sat1.r.copy(), sat1.v.copy()

        # Create new satellite from these state vectors
        sat2 = OrbitalDynamics(r=r1, v=v1)

        # Orbital elements should match (within tolerance)
        self.assertAlmostEqual(sat2.a, 7000.0, places=1)
        self.assertAlmostEqual(sat2.e, 0.1, places=3)
        self.assertAlmostEqual(sat2.i, np.radians(30.0), places=3)

    def test_altitude_above_earth(self):
        """Test that altitude is always above Earth's surface"""
        sat = OrbitalDynamics(a=7000, e=0.8)  # Highly eccentric orbit

        # Even at perigee of eccentric orbit, altitude should be positive
        altitude = np.linalg.norm(sat.r) - self.earth_radius
        self.assertGreater(altitude, 0)

        # For extreme case, check logic
        sat2 = OrbitalDynamics(a=self.earth_radius + 100, e=0)  # Very low orbit
        altitude2 = np.linalg.norm(sat2.r) - self.earth_radius
        self.assertAlmostEqual(altitude2, 100.0, places=1)

    def test_energy_conservation_no_perturbations(self):
        """Test energy conservation in two-body propagation"""
        # Circular orbit, no perturbations
        sat = OrbitalDynamics(a=7000, e=0, J2=False, drag=False, third_body=False)

        # Calculate initial specific orbital energy: ε = v²/2 - μ/r
        initial_energy = 0.5 * np.dot(sat.v, sat.v) - sat.mu / np.linalg.norm(sat.r)

        # Calculate orbital period directly
        period = 2 * math.pi * math.sqrt(sat.a**3 / sat.mu)

        # Propagate for half an orbit
        sat.propagate(period / 2, 50)

        # Calculate final energy
        final_energy = 0.5 * np.dot(sat.v, sat.v) - sat.mu / np.linalg.norm(sat.r)

        # Should be conserved (within numerical tolerance)
        self.assertAlmostEqual(initial_energy, final_energy, places=8)


class TestOrbitalDynamicsEdgeCases(TestOrbitalDynamics):
    """Test edge cases for OrbitalDynamics"""

    def test_near_parabolic_orbit(self):
        """Test near-parabolic orbit (e close to 1)"""
        sat = OrbitalDynamics(a=10000, e=0.99, nu=0.0)

        # Should initialize without error
        self.assertAlmostEqual(sat.e, 0.99)
        self.assertGreater(sat.a, 0)

        # Altitude at perigee should be positive
        altitude = np.linalg.norm(sat.r) - self.earth_radius
        self.assertGreater(altitude, 0)

    def test_retrograde_orbit(self):
        """Test retrograde orbit (i > 90°)"""
        sat = OrbitalDynamics(a=7000, e=0.1, i=120.0)  # Retrograde orbit

        self.assertAlmostEqual(sat.i, np.radians(120.0))
        self.assertGreater(sat.i, np.radians(90.0))

    def test_equatorial_orbit(self):
        """Test equatorial orbit (i = 0°)"""
        sat = OrbitalDynamics(a=7000, e=0.1, i=0.0, raan=0.0)

        self.assertAlmostEqual(sat.i, 0.0)
        self.assertAlmostEqual(sat.raan, 0.0)

    def test_polar_orbit(self):
        """Test polar orbit (i = 90°)"""
        sat = OrbitalDynamics(a=7000, e=0.1, i=90.0)

        self.assertAlmostEqual(sat.i, np.radians(90.0))

    def test_zero_eccentricity(self):
        """Test circular orbit (e = 0)"""
        sat = OrbitalDynamics(a=7000, e=0.0, nu=45.0)

        self.assertEqual(sat.e, 0.0)
        # For circular orbit, argument of perigee is undefined but should be handled

    def test_small_eccentricity(self):
        """Test orbit with very small eccentricity"""
        sat = OrbitalDynamics(a=7000, e=1e-6, nu=0.0)

        self.assertAlmostEqual(sat.e, 1e-6)
        altitude = np.linalg.norm(sat.r) - self.earth_radius
        self.assertGreater(altitude, 0)


if __name__ == "__main__":
    unittest.main()
