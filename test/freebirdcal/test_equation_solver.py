import cmath
import os
import sys
import unittest

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from freebirdcal.equation_solver import EquationSolver


class TestEquationSolver(unittest.TestCase):
    """Test cases for EquationSolver class"""

    def setUp(self):
        """Set up test fixture"""
        self.solver = EquationSolver()

    def test_solve_linear_1v_basic(self):
        """Test basic linear equation solving"""
        # 2x - 4 = 0 -> x = 2
        result = self.solver.solve_linear_1v(2, -4)
        self.assertAlmostEqual(result, 2.0)

    def test_solve_linear_1v_zero_a_nonzero_b(self):
        """Test linear equation with a=0 and b≠0 (no solution)"""
        # 0x + 5 = 0 -> no solution
        result = self.solver.solve_linear_1v(0, 5)
        self.assertEqual(result, "无解")

    def test_solve_linear_1v_zero_a_zero_b(self):
        """Test linear equation with a=0 and b=0 (infinite solutions)"""
        # 0x + 0 = 0 -> infinite solutions
        result = self.solver.solve_linear_1v(0, 0)
        self.assertEqual(result, "无穷多解")

    def test_solve_linear_1v_fractional(self):
        """Test linear equation with fractional solution"""
        # 3x + 2 = 0 -> x = -2/3
        result = self.solver.solve_linear_1v(3, 2)
        self.assertAlmostEqual(result, -2 / 3)

    def test_solve_quadratic_1v_real_roots(self):
        """Test quadratic equation with real roots"""
        # x² - 3x + 2 = 0 -> x=1, x=2
        result = self.solver.solve_quadratic_1v(1, -3, 2)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0], 2.0)
        self.assertAlmostEqual(result[1], 1.0)

    def test_solve_quadratic_1v_complex_roots(self):
        """Test quadratic equation with complex roots"""
        # x² + 1 = 0 -> x = i, x = -i
        result = self.solver.solve_quadratic_1v(1, 0, 1)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        # Check that roots are complex
        self.assertIsInstance(result[0], complex)
        self.assertIsInstance(result[1], complex)
        # Check values (should be i and -i)
        self.assertAlmostEqual(result[0], 1j)
        self.assertAlmostEqual(result[1], -1j)

    def test_solve_quadratic_1v_double_root(self):
        """Test quadratic equation with double root"""
        # x² - 2x + 1 = 0 -> x=1 (double root)
        result = self.solver.solve_quadratic_1v(1, -2, 1)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0], 1.0)
        self.assertAlmostEqual(result[1], 1.0)

    def test_solve_quadratic_1v_linear_case(self):
        """Test quadratic equation with a=0 (degenerates to linear)"""
        # 0x² + 2x - 4 = 0 -> 2x - 4 = 0 -> x=2
        result = self.solver.solve_quadratic_1v(0, 2, -4)
        self.assertAlmostEqual(result, 2.0)

    def test_solve_linear_2v_unique_solution(self):
        """Test 2x2 linear system with unique solution"""
        # 3x + 2y = 7
        # 2x - y = 4
        # Solution: x=15/7, y=2/7
        result = self.solver.solve_linear_2v([[3, 2], [2, -1]], [7, 4])
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0], 15 / 7)
        self.assertAlmostEqual(result[1], 2 / 7)

    def test_solve_linear_2v_no_solution(self):
        """Test 2x2 linear system with no solution"""
        # x + y = 1
        # x + y = 2
        result = self.solver.solve_linear_2v([[1, 1], [1, 1]], [1, 2])
        self.assertEqual(result, "无解")

    def test_solve_linear_2v_infinite_solutions(self):
        """Test 2x2 linear system with infinite solutions"""
        # x + y = 1
        # 2x + 2y = 2 (same equation multiplied by 2)
        result = self.solver.solve_linear_2v([[1, 1], [2, 2]], [1, 2])
        self.assertEqual(result, "无穷多解")

    def test_solve_linear_2v_fractional_solution(self):
        """Test 2x2 linear system with fractional solution"""
        # x + 2y = 1
        # 3x + 4y = 2
        # Solution: x=0, y=0.5
        result = self.solver.solve_linear_2v([[1, 2], [3, 4]], [1, 2])
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 0.5)

    def test_solve_linear_3v_unique_solution(self):
        """Test 3x3 linear system with unique solution"""
        # 2x + y + z = 4
        # x + 3y + 2z = 5
        # x + 0y + z = 1
        # Solution: x=1.5, y=1.5, z=-0.5
        result = self.solver.solve_linear_3v(
            [[2, 1, 1], [1, 3, 2], [1, 0, 1]], [4, 5, 1]
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 1.5)
        self.assertAlmostEqual(result[1], 1.5)
        self.assertAlmostEqual(result[2], -0.5)

    def test_solve_linear_3v_simple_case(self):
        """Test simple 3x3 linear system"""
        # x + 0y + 0z = 1
        # 0x + y + 0z = 2
        # 0x + 0y + z = 3
        result = self.solver.solve_linear_3v(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [1, 2, 3]
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 1.0)
        self.assertAlmostEqual(result[1], 2.0)
        self.assertAlmostEqual(result[2], 3.0)

    def test_solve_linear_3v_zero_determinant(self):
        """Test 3x3 linear system with zero determinant"""
        # x + y + z = 1
        # 2x + 2y + 2z = 2
        # 3x + 3y + 3z = 3
        result = self.solver.solve_linear_3v(
            [[1, 1, 1], [2, 2, 2], [3, 3, 3]], [1, 2, 3]
        )
        self.assertEqual(result, "无解或无穷多解")

    def test_solve_quadratic_2v_real_solutions(self):
        """Test system of linear + quadratic equations with real solutions"""
        # x + y = 3
        # x² + y² = 9
        # Solutions: (3, 0) and (0, 3)
        result = self.solver.solve_quadratic_2v((1, 1, 3), (1, 1, 0, 0, 0, -9))
        self.assertIsInstance(result, list)
        # Should have 2 solutions
        self.assertEqual(len(result), 2)

        # Check both solutions (order may vary)
        solutions_found = set()
        for sol in result:
            x, y = sol
            # Check if it's either (3, 0) or (0, 3)
            self.assertTrue(
                (abs(x - 3) < 1e-10 and abs(y - 0) < 1e-10)
                or (abs(x - 0) < 1e-10 and abs(y - 3) < 1e-10)
            )
            solutions_found.add((round(x.real, 10), round(y.real, 10)))

        # Both solutions should be present
        self.assertIn((3.0, 0.0), solutions_found)
        self.assertIn((0.0, 3.0), solutions_found)

    def test_solve_quadratic_2v_complex_solutions(self):
        """Test system of linear + quadratic equations with complex solutions"""
        # x + y = 0
        # x² + y² = -1
        # Solutions: (i/√2, -i/√2) and (-i/√2, i/√2)
        result = self.solver.solve_quadratic_2v((1, 1, 0), (1, 1, 0, 0, 0, 1))
        self.assertIsInstance(result, list)
        # Should have 2 solutions
        self.assertEqual(len(result), 2)

        for sol in result:
            x, y = sol
            self.assertIsInstance(x, complex)
            self.assertIsInstance(y, complex)

    def test_solve_quadratic_2v_invalid_linear(self):
        """Test system with invalid linear equation (A=0, B=0)"""
        # 0x + 0y = 3 (invalid)
        result = self.solver.solve_quadratic_2v((0, 0, 3), (1, 1, 0, 0, 0, 0))
        self.assertEqual(result, "无效方程")

    def test_solve_quadratic_2v_single_solution(self):
        """Test system with single solution (tangent case)"""
        # x + y = 2
        # x² + y² = 2
        # Solution: (1, 1) only (circle tangent to line)
        result = self.solver.solve_quadratic_2v((1, 1, 2), (1, 1, 0, 0, 0, -2))
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)  # Still returns 2 (double root)

        # Check that both solutions are (approximately) (1, 1)
        for x, y in result:
            self.assertAlmostEqual(x, 1.0)
            self.assertAlmostEqual(y, 1.0)


if __name__ == "__main__":
    unittest.main()
