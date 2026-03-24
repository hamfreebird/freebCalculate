import math
import os
import sys
import unittest

import numpy as np

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Try to import formula_cal, but skip tests if not available
try:
    import freebirdcal.formula_cal as fc

    FORMULA_CAL_AVAILABLE = True
except ImportError as e:
    FORMULA_CAL_AVAILABLE = False
    print(f"Warning: formula_cal module not available: {e}")


@unittest.skipIf(not FORMULA_CAL_AVAILABLE, "formula_cal module not available")
class TestFormulaCal(unittest.TestCase):
    """Test cases for formula_cal module"""

    def test_constants(self):
        """Test that physical constants are defined"""
        self.assertIn("G", fc.CONSTANTS)
        self.assertIn("c", fc.CONSTANTS)
        self.assertIn("sigma", fc.CONSTANTS)
        self.assertIn("M_sun", fc.CONSTANTS)
        self.assertIn("R_sun", fc.CONSTANTS)

        # Check some constant values
        self.assertAlmostEqual(fc.CONSTANTS["c"], 299792458.0)
        self.assertAlmostEqual(fc.CONSTANTS["G"], 6.67430e-11)
        self.assertAlmostEqual(fc.CONSTANTS["g"], 9.80665)

    def test_newton_gravitation(self):
        """Test Newton's law of universal gravitation"""
        # Earth-Moon system
        M_earth = 5.972e24  # kg
        M_moon = 7.348e22  # kg
        r = 3.844e8  # m (average distance)

        F = fc.newton_gravitation(M_earth, M_moon, r)

        # Calculate expected value using formula
        expected = fc.CONSTANTS["G"] * M_earth * M_moon / r**2

        self.assertAlmostEqual(F, expected, delta=expected * 1e-10)

        # Test with scalar values
        self.assertIsInstance(F, float)

        # Test with numpy arrays
        masses1 = np.array([M_earth, M_earth])
        masses2 = np.array([M_moon, M_moon])
        distances = np.array([r, r * 2])

        forces = fc.newton_gravitation(masses1, masses2, distances)
        self.assertEqual(len(forces), 2)
        self.assertIsInstance(forces, np.ndarray)

    def test_kepler_third_law(self):
        """Test Kepler's third law"""
        # Earth around Sun
        M_central = fc.CONSTANTS["M_sun"]
        a = 1.496e11  # m (1 AU)

        period = fc.kepler_third_law(M_central, a)

        # Calculate expected period (in seconds)
        expected = 2 * np.pi * np.sqrt(a**3 / (fc.CONSTANTS["G"] * M_central))

        self.assertAlmostEqual(period, expected, delta=expected * 1e-10)

        # Should be about 1 year
        self.assertAlmostEqual(period / (365.25 * 24 * 3600), 1.0, delta=0.01)

    def test_escape_velocity(self):
        """Test escape velocity calculation"""
        # Earth
        M_earth = 5.972e24  # kg
        R_earth = 6.371e6  # m

        v_esc = fc.escape_velocity(M_earth, R_earth)

        # Expected value: sqrt(2GM/R)
        expected = np.sqrt(2 * fc.CONSTANTS["G"] * M_earth / R_earth)

        self.assertAlmostEqual(v_esc, expected, delta=expected * 1e-10)

        # Should be about 11.2 km/s
        self.assertAlmostEqual(v_esc / 1000, 11.2, delta=0.2)

    def test_coulomb_force(self):
        """Test Coulomb's law"""
        q1 = 1.602e-19  # C (proton charge)
        q2 = -1.602e-19  # C (electron charge)
        r = 5.29e-11  # m (Bohr radius)

        F = fc.coulomb_force(q1, q2, r)

        # Expected value using Coulomb's constant
        expected = fc.CONSTANTS["ke"] * abs(q1) * abs(q2) / r**2

        self.assertAlmostEqual(F, expected, delta=abs(expected) * 1e-10)

        # Force should be attractive (negative)
        self.assertLess(F, 0)

    def test_electric_field(self):
        """Test electric field calculation"""
        q = 1.602e-19  # C
        r = np.array([0.1, 0, 0])  # 10 cm along x-axis

        E = fc.electric_field(q, r)

        # Magnitude should be k|q|/r^2
        r_mag = np.linalg.norm(r)
        expected_magnitude = fc.CONSTANTS["ke"] * abs(q) / r_mag**2

        # Field direction should be radial
        E_mag = np.linalg.norm(E)
        self.assertAlmostEqual(
            E_mag, expected_magnitude, delta=expected_magnitude * 1e-10
        )

        # For positive charge, field points away
        E_direction = E / E_mag
        expected_direction = r / r_mag
        np.testing.assert_array_almost_equal(E_direction, expected_direction)

    def test_biot_savart(self):
        """Test Biot-Savart law"""
        I = 1.0  # A
        dl = np.array([0, 0, 0.001])  # 1mm segment along z-axis
        r = np.array([0.1, 0, 0])  # 10 cm along x-axis

        B = fc.biot_savart(I, dl, r)

        # For a short straight segment, approximate formula
        # dB = (μ₀/4π) * I * (dl × r) / |r|^3
        mu0 = fc.CONSTANTS["μ0"]
        expected = (mu0 / (4 * np.pi)) * I * np.cross(dl, r) / (np.linalg.norm(r) ** 3)

        np.testing.assert_array_almost_equal(B, expected, decimal=12)

        # Should be perpendicular to both dl and r
        self.assertAlmostEqual(np.dot(B, dl), 0, delta=1e-12)
        self.assertAlmostEqual(np.dot(B, r), 0, delta=1e-12)

    def test_projectile_motion(self):
        """Test projectile motion calculation"""
        v0 = 50  # m/s
        theta = 45  # degrees
        t = np.linspace(0, 3, 10)  # 0 to 3 seconds

        x, y = fc.projectile_motion(v0, theta, t)

        # Convert angle to radians
        theta_rad = np.radians(theta)

        # Expected values
        expected_x = v0 * np.cos(theta_rad) * t
        expected_y = v0 * np.sin(theta_rad) * t - 0.5 * fc.CONSTANTS["g"] * t**2

        np.testing.assert_array_almost_equal(x, expected_x)
        np.testing.assert_array_almost_equal(y, expected_y)

        # Test with single time value
        t_single = 1.0
        x_single, y_single = fc.projectile_motion(v0, theta, t_single)
        self.assertIsInstance(x_single, (float, np.float64))
        self.assertIsInstance(y_single, (float, np.float64))

    def test_snells_law(self):
        """Test Snell's law"""
        n1 = 1.0  # air
        n2 = 1.33  # water
        theta1 = 30  # degrees

        theta2 = fc.snells_law(n1, n2, theta1)

        # Expected value
        expected = np.degrees(np.arcsin(n1 * np.sin(np.radians(theta1)) / n2))

        self.assertAlmostEqual(theta2, expected, delta=1e-10)

        # Test total internal reflection case
        # Light from water to air at angle > critical angle
        n1 = 1.33
        n2 = 1.0
        theta1 = 90  # degrees (should result in NaN or special value)

        theta2 = fc.snells_law(n1, n2, theta1)

        # Should be NaN or indicate total internal reflection
        self.assertTrue(np.isnan(theta2) or theta2 > 90)

    def test_reynolds_number(self):
        """Test Reynolds number calculation"""
        rho = 1000  # kg/m³ (water)
        v = 1.0  # m/s
        L = 0.1  # m
        mu = 0.001  # Pa·s

        Re = fc.reynolds_number(rho, v, L, mu)

        # Expected value
        expected = rho * v * L / mu

        self.assertAlmostEqual(Re, expected, delta=expected * 1e-10)

        # Test with dictionary from FLUID_PROPS
        if hasattr(fc, "FLUID_PROPS") and "water" in fc.FLUID_PROPS:
            water_props = fc.FLUID_PROPS["water"]
            Re_water = fc.reynolds_number(water_props["rho"], v, L, water_props["mu"])
            self.assertGreater(Re_water, 0)

    def test_bernoulli_total_pressure(self):
        """Test Bernoulli total pressure"""
        p_static = 101325  # Pa (atmospheric pressure)
        rho = 1.225  # kg/m³ (air)
        v = 10  # m/s

        p_total = fc.bernoulli_total_pressure(p_static, rho, v)

        # Expected value
        expected = p_static + 0.5 * rho * v**2

        self.assertAlmostEqual(p_total, expected, delta=expected * 1e-10)

    def test_kinetic_energy(self):
        """Test kinetic energy calculation"""
        m = 10  # kg
        v = 5  # m/s

        KE = fc.kinetic_energy(m, v)

        # Expected value
        expected = 0.5 * m * v**2

        self.assertAlmostEqual(KE, expected, delta=expected * 1e-10)

        # Test with numpy array velocities
        v_array = np.array([1, 2, 3, 4, 5])
        KE_array = fc.kinetic_energy(m, v_array)
        expected_array = 0.5 * m * v_array**2
        np.testing.assert_array_almost_equal(KE_array, expected_array)

    def test_simple_harmonic(self):
        """Test simple harmonic motion"""
        A = 0.1  # m amplitude
        omega = 2 * np.pi  # rad/s (1 Hz)
        t = np.array([0, 0.25, 0.5, 0.75, 1.0])  # seconds

        x = fc.simple_harmonic(A, omega, t)

        # Expected value
        expected = A * np.sin(omega * t)

        np.testing.assert_array_almost_equal(x, expected)

        # Test with phase shift
        phi = np.pi / 2
        x_phase = fc.simple_harmonic(A, omega, t, phi)
        expected_phase = A * np.sin(omega * t + phi)
        np.testing.assert_array_almost_equal(x_phase, expected_phase)

    def test_planck_law(self):
        """Test Planck's law for blackbody radiation"""
        # Test at room temperature for visible wavelength
        T = 300  # K
        lambda_ = 500e-9  # m (green light)

        B_lambda = fc.planck_law(T, lambda_)

        # Should be a positive number (very small at room temperature)
        self.assertGreater(B_lambda, 0)

        # Test at Sun's temperature for visible wavelength
        T_sun = 5778  # K
        B_sun = fc.planck_law(T_sun, lambda_)

        # Should be much larger than at room temperature
        self.assertGreater(B_sun, B_lambda * 1e6)

        # Test with numpy array wavelengths
        lambdas = np.array([400e-9, 500e-9, 600e-9, 700e-9])
        B_array = fc.planck_law(T_sun, lambdas)
        self.assertEqual(len(B_array), 4)
        self.assertTrue(np.all(B_array > 0))

    def test_mach_number(self):
        """Test Mach number calculation"""
        v = 300  # m/s
        # Speed of sound in air at 15°C
        c = 340.29  # m/s

        M = fc.mach_number(v, c)

        # Expected value
        expected = v / c

        self.assertAlmostEqual(M, expected, delta=expected * 1e-10)

        # Test subsonic, transonic, supersonic
        self.assertLess(fc.mach_number(100, c), 1)  # Subsonic
        self.assertAlmostEqual(fc.mach_number(c, c), 1, delta=1e-10)  # Sonic
        self.assertGreater(fc.mach_number(500, c), 1)  # Supersonic

    def test_linear_motion(self):
        """Test linear motion equations"""
        x0 = 0  # m
        v0 = 10  # m/s
        a = 2  # m/s²
        t = 5  # s

        x = fc.linear_motion(x0, v0, a, t)

        # Expected value
        expected = x0 + v0 * t + 0.5 * a * t**2

        self.assertAlmostEqual(x, expected, delta=expected * 1e-10)

        # Test with numpy array times
        t_array = np.array([0, 1, 2, 3, 4, 5])
        x_array = fc.linear_motion(x0, v0, a, t_array)
        expected_array = x0 + v0 * t_array + 0.5 * a * t_array**2
        np.testing.assert_array_almost_equal(x_array, expected_array)

    def test_momentum_conservation(self):
        """Test momentum conservation in 1D collision"""
        m1 = 2  # kg
        v1i = 3  # m/s
        m2 = 1  # kg
        v2i = 0  # m/s (stationary)

        # Perfectly elastic collision
        v1f, v2f = fc.momentum_conservation(m1, v1i, m2, v2i, e=1.0)

        # Check momentum conservation
        p_initial = m1 * v1i + m2 * v2i
        p_final = m1 * v1f + m2 * v2f
        self.assertAlmostEqual(p_final, p_initial, delta=abs(p_initial) * 1e-10)

        # Check kinetic energy conservation for elastic collision
        KE_initial = 0.5 * m1 * v1i**2 + 0.5 * m2 * v2i**2
        KE_final = 0.5 * m1 * v1f**2 + 0.5 * m2 * v2f**2
        self.assertAlmostEqual(KE_final, KE_initial, delta=abs(KE_initial) * 1e-10)

        # Perfectly inelastic collision
        v1f_inel, v2f_inel = fc.momentum_conservation(m1, v1i, m2, v2i, e=0.0)

        # In inelastic collision, final velocities should be equal
        self.assertAlmostEqual(v1f_inel, v2f_inel, delta=1e-10)

        # Momentum should still be conserved
        p_final_inel = m1 * v1f_inel + m2 * v2f_inel
        self.assertAlmostEqual(p_final_inel, p_initial, delta=abs(p_initial) * 1e-10)

    def test_young_interference_intensity(self):
        """Test Young's double-slit interference intensity"""
        I0 = 1.0  # Maximum intensity
        d = 1e-3  # Slit separation (m)
        lambda_ = 500e-9  # Wavelength (m)
        L = 1.0  # Distance to screen (m)
        y = np.linspace(-0.01, 0.01, 5)  # Positions on screen (m)

        I = fc.young_interference_intensity(I0, d, lambda_, L, y)

        # Should have same length as y
        self.assertEqual(len(I), len(y))

        # All intensities should be between 0 and I0
        self.assertTrue(np.all(I >= 0))
        self.assertTrue(np.all(I <= I0))

        # Pattern should be symmetric
        self.assertAlmostEqual(I[0], I[-1], delta=1e-10)
        self.assertAlmostEqual(I[1], I[-2], delta=1e-10)

    def test_stefan_boltzmann_luminosity(self):
        """Test Stefan-Boltzmann luminosity"""
        R = fc.CONSTANTS["R_sun"]  # Solar radius
        T = 5778  # Solar temperature (K)

        L = fc.stefan_boltzmann_luminosity(R, T)

        # Expected value
        expected = 4 * np.pi * R**2 * fc.CONSTANTS["sigma"] * T**4

        self.assertAlmostEqual(L, expected, delta=expected * 1e-10)

        # Solar luminosity should be about 3.828e26 W
        self.assertAlmostEqual(L / 1e26, 3.828, delta=0.5)

    def test_schwarzschild_radius(self):
        """Test Schwarzschild radius calculation"""
        M = fc.CONSTANTS["M_sun"]  # Solar mass

        r_s = fc.schwarzschild_radius(M)

        # Expected value
        expected = 2 * fc.CONSTANTS["G"] * M / fc.CONSTANTS["c"] ** 2

        self.assertAlmostEqual(r_s, expected, delta=expected * 1e-10)

        # For Sun, should be about 2.95 km
        self.assertAlmostEqual(r_s / 1000, 2.95, delta=0.1)


if __name__ == "__main__":
    unittest.main()
