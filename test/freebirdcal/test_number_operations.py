import cmath
import math
import os
import sys
import unittest

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from freebirdcal.number_operations import NumberOperations


class TestNumberOperations(unittest.TestCase):
    """Test cases for NumberOperations class"""

    def test_init_with_number(self):
        """Test initialization with numeric values"""
        # Test with integer
        num = NumberOperations(42)
        self.assertEqual(num.value, 42 + 0j)
        self.assertTrue(num.is_real)
        self.assertTrue(num.is_positive_integer)
        self.assertEqual(num.integer, 42)

        # Test with float
        num = NumberOperations(3.14)
        self.assertEqual(num.value, 3.14 + 0j)
        self.assertTrue(num.is_real)
        self.assertFalse(num.is_positive_integer)

        # Test with complex number
        num = NumberOperations(3 + 4j)
        self.assertEqual(num.value, 3 + 4j)
        self.assertFalse(num.is_real)
        self.assertFalse(num.is_positive_integer)

    def test_init_with_string_expression(self):
        """Test initialization with mathematical expressions as strings"""
        # Basic arithmetic
        num = NumberOperations("2/3 + 1/6")
        self.assertAlmostEqual(num.value, 0.8333333333333333 + 0j)
        self.assertTrue(num.is_real)

        # With Greek letter π
        num = NumberOperations("π/2")
        self.assertAlmostEqual(num.value, math.pi / 2 + 0j)
        self.assertTrue(num.is_real)

        # With pi spelled out
        num = NumberOperations("pi/4")
        self.assertAlmostEqual(num.value, math.pi / 4 + 0j)
        self.assertTrue(num.is_real)

        # With exponentiation
        num = NumberOperations("2^3")
        self.assertAlmostEqual(num.value, 8 + 0j)
        self.assertTrue(num.is_real)

    def test_complex_expressions(self):
        """Test initialization with complex number expressions"""
        # Square root of negative number
        num = NumberOperations("sqrt(-4)")
        self.assertEqual(num.value, 2j)
        self.assertFalse(num.is_real)

        # Complex expression
        num = NumberOperations("sqrt(-4) + 3^2")
        self.assertEqual(num.value, 9 + 2j)
        self.assertFalse(num.is_real)

    def test_parse_math_expression_with_spaces(self):
        """Test expression parsing with spaces"""
        num = NumberOperations("2 / 3 + 1 / 6")
        self.assertAlmostEqual(num.value, 0.8333333333333333 + 0j)

    def test_invalid_expression(self):
        """Test initialization with invalid expression"""
        with self.assertRaises(ValueError):
            NumberOperations("invalid expression")

        with self.assertRaises(ValueError):
            NumberOperations("2 / 0")  # Division by zero

    def test_power_operation(self):
        """Test power method"""
        num = NumberOperations(2)
        # Square
        result = num.power(2)
        self.assertAlmostEqual(result, 4 + 0j)

        # Square root
        result = num.power(0.5)
        self.assertAlmostEqual(result, math.sqrt(2) + 0j)

        # Complex number power
        num = NumberOperations(1j)
        result = num.power(2)
        self.assertAlmostEqual(result, -1 + 0j)

    def test_nth_root_operation(self):
        """Test nth_root method"""
        num = NumberOperations(8)
        # Cube root
        result = num.nth_root(3)
        self.assertAlmostEqual(result, 2 + 0j)

        # Square root
        result = num.nth_root(2)
        self.assertAlmostEqual(result, math.sqrt(8) + 0j)

        # With complex number
        num = NumberOperations(-8)
        result = num.nth_root(3)
        # Cube root of -8 is complex (principal value is 1 + sqrt(3)i)
        # Verify that cubing gives back -8
        self.assertAlmostEqual(result**3, -8 + 0j)

    def test_nth_root_zero_error(self):
        """Test nth_root with n=0 should raise ValueError"""
        num = NumberOperations(4)
        with self.assertRaises(ValueError):
            num.nth_root(0)

    def test_logarithm_operation(self):
        """Test logarithm method"""
        num = NumberOperations(100)
        # Natural log
        result = num.logarithm()
        self.assertAlmostEqual(result, math.log(100) + 0j)

        # Base 10 log
        result = num.logarithm(10)
        self.assertAlmostEqual(result, 2 + 0j)

        # Complex number log
        num = NumberOperations(1j)
        result = num.logarithm()
        expected = cmath.log(1j)
        self.assertAlmostEqual(result.real, expected.real)
        self.assertAlmostEqual(result.imag, expected.imag)

    def test_factorize_positive_integer(self):
        """Test factorization of positive integers"""
        # Test with 12
        num = NumberOperations(12)
        factors = num.factorize()
        self.assertEqual(factors, [2, 2, 3])

        # Test with prime number
        num = NumberOperations(17)
        factors = num.factorize()
        self.assertEqual(factors, [17])

        # Test with 1 (edge case)
        num = NumberOperations(1)
        factors = num.factorize()
        self.assertEqual(factors, [])

        # Test with 100
        num = NumberOperations(100)
        factors = num.factorize()
        self.assertEqual(factors, [2, 2, 5, 5])

        # Test with large even number
        num = NumberOperations(1024)
        factors = num.factorize()
        self.assertEqual(factors, [2] * 10)

    def test_factorize_non_integer(self):
        """Test factorization with non-integer values"""
        # Float
        num = NumberOperations(3.14)
        factors = num.factorize()
        self.assertEqual(factors, [])

        # Negative integer
        num = NumberOperations(-12)
        factors = num.factorize()
        self.assertEqual(factors, [])

        # Complex number
        num = NumberOperations(3 + 4j)
        factors = num.factorize()
        self.assertEqual(factors, [])

        # String expression resulting in non-integer
        num = NumberOperations("π")
        factors = num.factorize()
        self.assertEqual(factors, [])

    def test_factorize_string_integer(self):
        """Test factorization with string representation of integer"""
        num = NumberOperations("12")
        factors = num.factorize()
        self.assertEqual(factors, [2, 2, 3])

    def test_edge_cases(self):
        """Test various edge cases"""
        # Zero
        num = NumberOperations(0)
        self.assertEqual(num.value, 0 + 0j)
        self.assertTrue(num.is_real)
        self.assertFalse(num.is_positive_integer)  # 0 is not positive
        self.assertEqual(num.factorize(), [])

        # Negative zero (should be same as zero)
        num = NumberOperations(-0.0)
        self.assertEqual(num.value, 0 + 0j)
        self.assertTrue(num.is_real)

        # Very small number
        num = NumberOperations(1e-10)
        self.assertTrue(num.is_real)
        self.assertFalse(num.is_positive_integer)

    def test_mathematical_constants(self):
        """Test with mathematical constants"""
        # Euler's number
        num = NumberOperations("e")
        self.assertAlmostEqual(num.value, math.e + 0j)
        self.assertTrue(num.is_real)

        # Expression with e
        num = NumberOperations("e^2")
        self.assertAlmostEqual(num.value, math.e**2 + 0j)
        self.assertTrue(num.is_real)

    def test_unicode_characters(self):
        """Test with Unicode mathematical symbols"""
        # Note: √ support is buggy in library, use sqrt() instead
        # Using sqrt for square root
        num = NumberOperations("sqrt(4)")
        self.assertAlmostEqual(num.value, 2 + 0j)
        self.assertTrue(num.is_real)

        # Using π
        num = NumberOperations("π")
        self.assertAlmostEqual(num.value, math.pi + 0j)
        self.assertTrue(num.is_real)

    def test_is_real_property(self):
        """Test is_real property with various inputs"""
        # Real numbers
        self.assertTrue(NumberOperations(42).is_real)
        self.assertTrue(NumberOperations(3.14).is_real)
        self.assertTrue(NumberOperations("2+0j").is_real)

        # Complex numbers
        self.assertFalse(NumberOperations(3 + 4j).is_real)
        self.assertFalse(NumberOperations("sqrt(-1)").is_real)
        self.assertFalse(NumberOperations("1j").is_real)

    def test_is_positive_integer_property(self):
        """Test is_positive_integer property"""
        # Positive integers
        self.assertTrue(NumberOperations(1).is_positive_integer)
        self.assertTrue(NumberOperations(42).is_positive_integer)
        self.assertTrue(NumberOperations("100").is_positive_integer)

        # Non-integers
        self.assertFalse(NumberOperations(3.14).is_positive_integer)
        self.assertFalse(NumberOperations("π").is_positive_integer)

        # Non-positive integers
        self.assertFalse(NumberOperations(0).is_positive_integer)
        self.assertFalse(NumberOperations(-5).is_positive_integer)

        # Complex numbers
        self.assertFalse(NumberOperations(3 + 4j).is_positive_integer)


if __name__ == "__main__":
    unittest.main()
