"""
Tests for spacetime_coordinate module
"""

import math
import os
import sys
import unittest

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from freebirdcal.freeastro.spacetime_coordinate import (
    SpacetimeCoordinateSystem,
    compute_velocity_angles,
    decompose_velocity,
)


class TestSpacetimeCoordinateSystem(unittest.TestCase):
    """Test cases for SpacetimeCoordinateSystem class"""

    def setUp(self):
        """Set up test fixture"""
        self.system = SpacetimeCoordinateSystem()

    def test_initialization(self):
        """Test system initialization"""
        # Test that origin is at (0,0,0,0)
        self.assertEqual(self.system.origin, (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(self.system.points, {})
        self.assertEqual(self.system.next_id, 1)

    def test_add_point(self):
        """Test adding points to the system"""
        # Add a point at relative coordinates (2, 3, 1, 0)
        point_id = self.system.add_point(2, 3, 1, 0)

        # Check that ID was returned
        self.assertEqual(point_id, 1)

        # Check that point was added with correct absolute coordinates
        # Origin is (0,0,0,0), so absolute = relative
        self.assertIn(point_id, self.system.points)
        self.assertEqual(self.system.points[point_id], (2.0, 3.0, 1.0, 0.0))

        # Check that next_id was incremented
        self.assertEqual(self.system.next_id, 2)

    def test_add_multiple_points(self):
        """Test adding multiple points"""
        point1 = self.system.add_point(1, 2, 3, 4)
        point2 = self.system.add_point(5, 6, 7, 8)

        self.assertEqual(point1, 1)
        self.assertEqual(point2, 2)
        self.assertEqual(self.system.points[point1], (1.0, 2.0, 3.0, 4.0))
        self.assertEqual(self.system.points[point2], (5.0, 6.0, 7.0, 8.0))

    def test_remove_point(self):
        """Test removing points"""
        point_id = self.system.add_point(1, 2, 3, 4)
        self.assertIn(point_id, self.system.points)

        self.system.remove_point(point_id)
        self.assertNotIn(point_id, self.system.points)

    def test_remove_nonexistent_point(self):
        """Test removing a point that doesn't exist (should not raise error)"""
        # This should not raise an error
        self.system.remove_point(999)

    def test_get_coordinate(self):
        """Test getting relative coordinates of a point"""
        # Add point at (2, 3, 1, 4) relative to origin
        point_id = self.system.add_point(2, 3, 1, 4)

        # Get relative coordinates
        coords = self.system.get_coordinate(point_id)
        self.assertEqual(coords, (2.0, 3.0, 1.0, 4.0))

    def test_get_coordinate_nonexistent(self):
        """Test getting coordinates of a non-existent point"""
        coords = self.system.get_coordinate(999)
        self.assertIsNone(coords)

    def test_change_origin(self):
        """Test changing the origin of the coordinate system"""
        # Add two points
        point1 = self.system.add_point(2, 3, 1, 0)
        point2 = self.system.add_point(5, 7, 2, 4)

        # Change origin to point2
        self.system.change_origin(point2)

        # Check that origin is now at point2's absolute coordinates
        self.assertEqual(self.system.origin, (5.0, 7.0, 2.0, 4.0))

        # Check that point1's relative coordinates are now correct
        # Relative to new origin: (2-5, 3-7, 1-2, 0-4) = (-3, -4, -1, -4)
        coords = self.system.get_coordinate(point1)
        self.assertEqual(coords, (-3.0, -4.0, -1.0, -4.0))

    def test_change_origin_nonexistent(self):
        """Test changing origin to non-existent point (should not change origin)"""
        original_origin = self.system.origin

        # This should not raise an error or change the origin
        self.system.change_origin(999)

        # Origin should remain unchanged
        self.assertEqual(self.system.origin, original_origin)

    def test_calculate_space_distance(self):
        """Test calculating 3D Euclidean distance between points"""
        point1 = self.system.add_point(2, 3, 1, 0)
        point2 = self.system.add_point(5, 7, 2, 4)

        # Distance = sqrt((5-2)^2 + (7-3)^2 + (2-1)^2) = sqrt(9 + 16 + 1) = sqrt(26) ≈ 5.099
        distance = self.system.calculate_space_distance(point1, point2)
        self.assertAlmostEqual(distance, math.sqrt(26))

        # Distance should be symmetric
        distance_reverse = self.system.calculate_space_distance(point2, point1)
        self.assertAlmostEqual(distance_reverse, math.sqrt(26))

        # Test with same point (distance should be 0)
        distance_same = self.system.calculate_space_distance(point1, point1)
        self.assertAlmostEqual(distance_same, 0.0)

    def test_calculate_space_distance_nonexistent(self):
        """Test calculating distance with non-existent points"""
        point_id = self.system.add_point(1, 2, 3, 4)

        # One point doesn't exist
        distance = self.system.calculate_space_distance(point_id, 999)
        self.assertIsNone(distance)

        # Both points don't exist
        distance = self.system.calculate_space_distance(998, 999)
        self.assertIsNone(distance)

    def test_calculate_time_distance(self):
        """Test calculating absolute time difference between points"""
        point1 = self.system.add_point(2, 3, 1, 0)
        point2 = self.system.add_point(5, 7, 2, 4)

        # Time distance = |4 - 0| = 4
        time_diff = self.system.calculate_time_distance(point1, point2)
        self.assertAlmostEqual(time_diff, 4.0)

        # Time distance should be symmetric
        time_diff_reverse = self.system.calculate_time_distance(point2, point1)
        self.assertAlmostEqual(time_diff_reverse, 4.0)

        # Test with same point (time distance should be 0)
        time_diff_same = self.system.calculate_time_distance(point1, point1)
        self.assertAlmostEqual(time_diff_same, 0.0)

    def test_calculate_time_distance_nonexistent(self):
        """Test calculating time distance with non-existent points"""
        point_id = self.system.add_point(1, 2, 3, 4)

        # One point doesn't exist
        time_diff = self.system.calculate_time_distance(point_id, 999)
        self.assertIsNone(time_diff)

        # Both points don't exist
        time_diff = self.system.calculate_time_distance(998, 999)
        self.assertIsNone(time_diff)

    def test_move_point(self):
        """Test calculating new coordinates after moving a point"""
        point_id = self.system.add_point(2, 3, 1, 0)

        # Move point with velocity (1, 2, 3) for 5 time units
        new_coords = self.system.move_point(point_id, 1, 2, 3, 5)

        # New coordinates: (2+1*5, 3+2*5, 1+3*5, 0+5) = (7, 13, 16, 5)
        expected = (7.0, 13.0, 16.0, 5.0)
        self.assertEqual(new_coords, expected)

        # Original point should not be modified
        coords = self.system.get_coordinate(point_id)
        self.assertEqual(coords, (2.0, 3.0, 1.0, 0.0))

    def test_move_point_nonexistent(self):
        """Test moving a non-existent point"""
        new_coords = self.system.move_point(999, 1, 2, 3, 5)
        self.assertIsNone(new_coords)

    def test_move_point_with_negative_time(self):
        """Test moving point with negative time (moving backward)"""
        point_id = self.system.add_point(2, 3, 1, 0)

        # Move point with velocity (1, 2, 3) for -2 time units
        new_coords = self.system.move_point(point_id, 1, 2, 3, -2)

        # New coordinates: (2+1*(-2), 3+2*(-2), 1+3*(-2), 0+(-2)) = (0, -1, -5, -2)
        expected = (0.0, -1.0, -5.0, -2.0)
        self.assertEqual(new_coords, expected)

    def test_move_point_with_zero_time(self):
        """Test moving point with zero time (should return same position)"""
        point_id = self.system.add_point(2, 3, 1, 0)

        new_coords = self.system.move_point(point_id, 1, 2, 3, 0)
        expected = (2.0, 3.0, 1.0, 0.0)
        self.assertEqual(new_coords, expected)


class TestVelocityFunctions(unittest.TestCase):
    """Test cases for decompose_velocity and compute_velocity_angles functions"""

    def test_decompose_velocity(self):
        """Test decomposing velocity into components"""
        # Test with 30° azimuth, 45° elevation, speed 10
        vx, vy, vz = decompose_velocity(
            speed=10, azimuth=math.radians(30), elevation=math.radians(45)
        )

        # Expected values:
        # cos(45°) ≈ 0.7071, sin(45°) ≈ 0.7071
        # cos(30°) ≈ 0.8660, sin(30°) = 0.5
        # vx = 10 * cos(45°) * cos(30°) ≈ 10 * 0.7071 * 0.8660 ≈ 6.1237
        # vy = 10 * cos(45°) * sin(30°) ≈ 10 * 0.7071 * 0.5 ≈ 3.5355
        # vz = 10 * sin(45°) ≈ 10 * 0.7071 ≈ 7.0711
        self.assertAlmostEqual(vx, 6.123724356957945)
        self.assertAlmostEqual(vy, 3.5355339059327378)
        self.assertAlmostEqual(vz, 7.0710678118654755)

    def test_decompose_velocity_zero_speed(self):
        """Test decomposing zero speed"""
        vx, vy, vz = decompose_velocity(0, math.radians(30), math.radians(45))
        self.assertEqual(vx, 0.0)
        self.assertEqual(vy, 0.0)
        self.assertEqual(vz, 0.0)

    def test_decompose_velocity_horizontal(self):
        """Test decomposing horizontal velocity (elevation = 0)"""
        vx, vy, vz = decompose_velocity(
            speed=5,
            azimuth=math.radians(60),  # 60 degrees
            elevation=0,  # Horizontal
        )

        # vz should be 0 for horizontal motion
        self.assertAlmostEqual(vz, 0.0)

        # vx = 5 * cos(0) * cos(60°) = 5 * 1 * 0.5 = 2.5
        # vy = 5 * cos(0) * sin(60°) = 5 * 1 * √3/2 ≈ 4.3301
        self.assertAlmostEqual(vx, 2.5)
        self.assertAlmostEqual(vy, 4.330127018922194)

    def test_decompose_velocity_vertical(self):
        """Test decomposing vertical velocity (azimuth irrelevant)"""
        vx, vy, vz = decompose_velocity(
            speed=8,
            azimuth=math.radians(90),  # 90 degrees (but should be irrelevant)
            elevation=math.radians(90),  # Straight up
        )

        # For straight up: vx = 0, vy = 0, vz = speed
        self.assertAlmostEqual(vx, 0.0)
        self.assertAlmostEqual(vy, 0.0)
        self.assertAlmostEqual(vz, 8.0)

    def test_compute_velocity_angles(self):
        """Test computing speed and angles from components"""
        # Test with components from previous test
        vx, vy, vz = 6.123724356957945, 3.5355339059327378, 7.0710678118654755

        speed, azimuth, elevation = compute_velocity_angles(vx, vy, vz)

        # Speed should be 10
        self.assertAlmostEqual(speed, 10.0)

        # Azimuth should be 30° (in radians)
        self.assertAlmostEqual(azimuth, math.radians(30))

        # Elevation should be 45° (in radians)
        self.assertAlmostEqual(elevation, math.radians(45))

    def test_compute_velocity_angles_zero_velocity(self):
        """Test computing angles for zero velocity"""
        speed, azimuth, elevation = compute_velocity_angles(0, 0, 0)
        self.assertEqual(speed, 0.0)
        self.assertEqual(azimuth, 0.0)
        self.assertEqual(elevation, 0.0)

    def test_compute_velocity_angles_horizontal(self):
        """Test computing angles for horizontal velocity"""
        # Horizontal velocity at 60° azimuth, speed 5
        vx, vy = 2.5, 4.330127018922194  # From previous test
        vz = 0.0

        speed, azimuth, elevation = compute_velocity_angles(vx, vy, vz)

        self.assertAlmostEqual(speed, 5.0)
        self.assertAlmostEqual(azimuth, math.radians(60))
        self.assertAlmostEqual(elevation, 0.0)

    def test_compute_velocity_angles_vertical(self):
        """Test computing angles for vertical velocity"""
        # Straight up velocity
        vx, vy, vz = 0.0, 0.0, 8.0

        speed, azimuth, elevation = compute_velocity_angles(vx, vy, vz)

        self.assertAlmostEqual(speed, 8.0)
        # Azimuth is undefined for pure vertical, function returns 0
        self.assertEqual(azimuth, 0.0)
        self.assertAlmostEqual(elevation, math.radians(90))

    def test_compute_velocity_angles_pure_x_direction(self):
        """Test computing angles for velocity purely in x direction"""
        vx, vy, vz = 7.0, 0.0, 0.0

        speed, azimuth, elevation = compute_velocity_angles(vx, vy, vz)

        self.assertAlmostEqual(speed, 7.0)
        self.assertAlmostEqual(azimuth, 0.0)  # Along x-axis
        self.assertAlmostEqual(elevation, 0.0)

    def test_compute_velocity_angles_pure_y_direction(self):
        """Test computing angles for velocity purely in y direction"""
        vx, vy, vz = 0.0, 5.0, 0.0

        speed, azimuth, elevation = compute_velocity_angles(vx, vy, vz)

        self.assertAlmostEqual(speed, 5.0)
        self.assertAlmostEqual(azimuth, math.radians(90))  # Along y-axis
        self.assertAlmostEqual(elevation, 0.0)

    def test_compute_velocity_angles_pure_z_direction(self):
        """Test computing angles for velocity purely in z direction"""
        vx, vy, vz = 0.0, 0.0, 3.0

        speed, azimuth, elevation = compute_velocity_angles(vx, vy, vz)

        self.assertAlmostEqual(speed, 3.0)
        self.assertEqual(azimuth, 0.0)  # Returns 0 for pure vertical
        self.assertAlmostEqual(elevation, math.radians(90))

    def test_round_trip_decomposition(self):
        """Test that decompose and compute are inverse operations"""
        # Original values
        original_speed = 12.5
        original_azimuth = math.radians(42)
        original_elevation = math.radians(17)

        # Decompose
        vx, vy, vz = decompose_velocity(
            original_speed, original_azimuth, original_elevation
        )

        # Recompute
        computed_speed, computed_azimuth, computed_elevation = compute_velocity_angles(
            vx, vy, vz
        )

        # Check round-trip
        self.assertAlmostEqual(computed_speed, original_speed)
        self.assertAlmostEqual(computed_azimuth, original_azimuth)
        self.assertAlmostEqual(computed_elevation, original_elevation)

    def test_round_trip_composition(self):
        """Test that compute and decompose are inverse operations"""
        # Original components
        original_vx = 3.7
        original_vy = -2.1
        original_vz = 5.8

        # Compute angles
        speed, azimuth, elevation = compute_velocity_angles(
            original_vx, original_vy, original_vz
        )

        # Decompose
        vx, vy, vz = decompose_velocity(speed, azimuth, elevation)

        # Check round-trip (allow for small floating point differences)
        self.assertAlmostEqual(vx, original_vx)
        self.assertAlmostEqual(vy, original_vy)
        self.assertAlmostEqual(vz, original_vz)


if __name__ == "__main__":
    unittest.main()
