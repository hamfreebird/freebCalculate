"""
Tests for spacetime_event module
"""

import math
import os
import sys
import unittest

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from freebirdcal.freeastro.spacetime_event import (
    SpacetimeEvent,
    decompose_velocity,
    relativistic_velocity_addition,
)


class TestDecomposeVelocity(unittest.TestCase):
    """Test cases for decompose_velocity function"""

    def test_decompose_velocity_basic(self):
        """Test basic velocity decomposition"""
        vx, vy = decompose_velocity(speed=10, angle_deg=30)

        # Expected: vx = 10 * cos(30°) = 10 * 0.8660254 ≈ 8.660254
        # vy = 10 * sin(30°) = 10 * 0.5 = 5
        self.assertAlmostEqual(vx, 8.660254037844386)
        self.assertAlmostEqual(vy, 5.0)

    def test_decompose_velocity_zero_angle(self):
        """Test decomposition with zero angle (purely x direction)"""
        vx, vy = decompose_velocity(speed=5, angle_deg=0)
        self.assertAlmostEqual(vx, 5.0)
        self.assertAlmostEqual(vy, 0.0)

    def test_decompose_velocity_90_angle(self):
        """Test decomposition with 90° angle (purely y direction)"""
        vx, vy = decompose_velocity(speed=3, angle_deg=90)
        self.assertAlmostEqual(vx, 0.0)
        self.assertAlmostEqual(vy, 3.0)

    def test_decompose_velocity_negative_angle(self):
        """Test decomposition with negative angle"""
        vx, vy = decompose_velocity(speed=4, angle_deg=-45)

        # cos(-45°) = cos(45°) = √2/2 ≈ 0.7071
        # sin(-45°) = -sin(45°) = -√2/2 ≈ -0.7071
        self.assertAlmostEqual(vx, 2.8284271247461903)  # 4 * √2/2
        self.assertAlmostEqual(vy, -2.8284271247461903)  # 4 * -√2/2

    def test_decompose_velocity_zero_speed(self):
        """Test decomposition with zero speed"""
        vx, vy = decompose_velocity(speed=0, angle_deg=30)
        self.assertAlmostEqual(vx, 0.0)
        self.assertAlmostEqual(vy, 0.0)


class TestRelativisticVelocityAddition(unittest.TestCase):
    """Test cases for relativistic_velocity_addition function"""

    def test_relativistic_velocity_addition_simple(self):
        """Test simple relativistic velocity addition along x axis"""
        # v = 0.6c along x, u' = 0.5c along x in moving frame
        wx, wy = relativistic_velocity_addition(vx=0.6, vy=0, ux=0.5, uy=0, c=1.0)

        # Expected: w = (0.6 + 0.5) / (1 + 0.6*0.5) = 1.1 / 1.3 ≈ 0.84615
        self.assertAlmostEqual(wx, 0.8461538461538463)
        self.assertAlmostEqual(wy, 0.0)

    def test_relativistic_velocity_addition_perpendicular(self):
        """Test velocity addition with perpendicular components"""
        # v = 0.6c along x, u' = 0.5c along y in moving frame
        wx, wy = relativistic_velocity_addition(vx=0.6, vy=0, ux=0, uy=0.5, c=1.0)

        # Expected: wx = 0.6, wy = 0.5 * sqrt(1-0.6^2) / (1+0) = 0.5 * 0.8 = 0.4
        self.assertAlmostEqual(wx, 0.6)
        self.assertAlmostEqual(wy, 0.4)

    def test_relativistic_velocity_addition_combined(self):
        """Test velocity addition with both x and y components"""
        wx, wy = relativistic_velocity_addition(vx=0.3, vy=0.4, ux=0.1, uy=0.2, c=1.0)

        # Verify magnitude is less than c
        speed = math.hypot(wx, wy)
        self.assertLess(speed, 1.0)

    def test_relativistic_velocity_addition_speed_limit(self):
        """Test that resulting speed cannot exceed c"""
        # Even adding 0.9c + 0.9c should give < c
        wx, wy = relativistic_velocity_addition(vx=0.9, vy=0, ux=0.9, uy=0, c=1.0)
        speed = math.hypot(wx, wy)
        self.assertLess(speed, 1.0)
        self.assertGreater(speed, 0.0)

    def test_relativistic_velocity_addition_zero_velocity(self):
        """Test adding zero velocity"""
        wx, wy = relativistic_velocity_addition(vx=0.6, vy=0, ux=0, uy=0, c=1.0)
        self.assertAlmostEqual(wx, 0.6)
        self.assertAlmostEqual(wy, 0.0)

    def test_relativistic_velocity_addition_same_frame(self):
        """Test velocity addition in same frame (v=0)"""
        wx, wy = relativistic_velocity_addition(vx=0, vy=0, ux=0.7, uy=0, c=1.0)
        self.assertAlmostEqual(wx, 0.7)
        self.assertAlmostEqual(wy, 0.0)


class TestSpacetimeEvent(unittest.TestCase):
    """Test cases for SpacetimeEvent class"""

    def test_initialization(self):
        """Test SpacetimeEvent initialization"""
        event = SpacetimeEvent(x=1.0, y=2.0, t=3.0, c=1.0)
        self.assertEqual(event.x, 1.0)
        self.assertEqual(event.y, 2.0)
        self.assertEqual(event.t, 3.0)
        self.assertEqual(event.c, 1.0)

    def test_initialization_default_c(self):
        """Test initialization with default c=1"""
        event = SpacetimeEvent(x=1.0, y=2.0, t=3.0)
        self.assertEqual(event.c, 1.0)

    def test_interval_to_timelike(self):
        """Test interval calculation for timelike separation"""
        event1 = SpacetimeEvent(0, 0, 0, c=1)
        event2 = SpacetimeEvent(1, 1, 2, c=1)  # 1²+1²-2² = -2 < 0

        interval = event1.interval_to(event2)
        self.assertAlmostEqual(interval, -2.0)

    def test_interval_to_lightlike(self):
        """Test interval calculation for lightlike separation"""
        event1 = SpacetimeEvent(0, 0, 0, c=1)
        event2 = SpacetimeEvent(3, 4, 5, c=1)  # 3²+4²-5² = 0

        interval = event1.interval_to(event2)
        self.assertAlmostEqual(interval, 0.0)

    def test_interval_to_spacelike(self):
        """Test interval calculation for spacelike separation"""
        event1 = SpacetimeEvent(0, 0, 0, c=1)
        event2 = SpacetimeEvent(3, 4, 2, c=1)  # 3²+4²-2² = 21 > 0

        interval = event1.interval_to(event2)
        self.assertAlmostEqual(interval, 21.0)

    def test_interval_type_timelike(self):
        """Test interval type classification for timelike"""
        event1 = SpacetimeEvent(0, 0, 0, c=1)
        event2 = SpacetimeEvent(1, 1, 2, c=1)

        interval_type = event1.interval_type(event2)
        self.assertEqual(interval_type, "timelike")

    def test_interval_type_lightlike(self):
        """Test interval type classification for lightlike"""
        event1 = SpacetimeEvent(0, 0, 0, c=1)
        event2 = SpacetimeEvent(3, 4, 5, c=1)

        interval_type = event1.interval_type(event2)
        self.assertEqual(interval_type, "lightlike")

    def test_interval_type_spacelike(self):
        """Test interval type classification for spacelike"""
        event1 = SpacetimeEvent(0, 0, 0, c=1)
        event2 = SpacetimeEvent(3, 4, 2, c=1)

        interval_type = event1.interval_type(event2)
        self.assertEqual(interval_type, "spacelike")

    def test_is_in_future_of_true(self):
        """Test is_in_future_of when event is in future light cone"""
        event1 = SpacetimeEvent(0, 0, 0, c=1)
        event2 = SpacetimeEvent(1, 1, 2, c=1)  # Timelike and t2 > t1

        self.assertTrue(event2.is_in_future_of(event1))
        self.assertFalse(event1.is_in_future_of(event2))

    def test_is_in_future_of_false_spacelike(self):
        """Test is_in_future_of when events are spacelike separated"""
        event1 = SpacetimeEvent(0, 0, 0, c=1)
        event2 = SpacetimeEvent(3, 4, 2, c=1)  # Spacelike

        self.assertFalse(event2.is_in_future_of(event1))

    def test_is_in_future_of_false_past(self):
        """Test is_in_future_of when event is in past light cone"""
        event1 = SpacetimeEvent(0, 0, 0, c=1)
        event2 = SpacetimeEvent(1, 1, 2, c=1)  # Timelike and t2 > t1

        # event1 is in past of event2
        self.assertTrue(event1.is_in_past_of(event2))
        self.assertFalse(event2.is_in_past_of(event1))

    def test_is_in_past_of_true(self):
        """Test is_in_past_of when event is in past light cone"""
        event1 = SpacetimeEvent(0, 0, 0, c=1)
        event2 = SpacetimeEvent(1, 1, 2, c=1)  # Timelike and t2 > t1

        self.assertTrue(event1.is_in_past_of(event2))

    def test_lorentz_transform_basic(self):
        """Test basic Lorentz transformation along x axis"""
        event = SpacetimeEvent(x=2, y=0, t=3, c=1)
        transformed = event.lorentz_transform(v=0.8)

        # gamma = 1/sqrt(1-0.8²) = 1/0.6 ≈ 1.6667
        # x' = gamma*(x - v*t) = 1.6667*(2 - 0.8*3) = 1.6667*(-0.4) ≈ -0.6667
        # t' = gamma*(t - v*x) = 1.6667*(3 - 0.8*2) = 1.6667*(1.4) ≈ 2.3333
        # y unchanged
        self.assertAlmostEqual(transformed.x, -0.6666666666666667, places=10)
        self.assertAlmostEqual(transformed.y, 0.0)
        self.assertAlmostEqual(transformed.t, 2.3333333333333335, places=10)
        self.assertEqual(transformed.c, 1.0)

    def test_lorentz_transform_speed_limit(self):
        """Test Lorentz transform with speed >= c should raise ValueError"""
        event = SpacetimeEvent(x=2, y=0, t=3, c=1)

        with self.assertRaises(ValueError):
            event.lorentz_transform(v=1.0)  # Speed = c

        with self.assertRaises(ValueError):
            event.lorentz_transform(v=1.2)  # Speed > c

    def test_lorentz_transform_y(self):
        """Test Lorentz transformation along y axis"""
        event = SpacetimeEvent(x=0, y=2, t=3, c=1)
        transformed = event.lorentz_transform_y(v=0.6)

        # gamma = 1/sqrt(1-0.6²) = 1.25
        # y' = gamma*(y - v*t) = 1.25*(2 - 0.6*3) = 1.25*0.2 = 0.25
        # t' = gamma*(t - v*y) = 1.25*(3 - 0.6*2) = 1.25*1.8 = 2.25
        # x unchanged
        self.assertAlmostEqual(transformed.x, 0.0)
        self.assertAlmostEqual(transformed.y, 0.25)
        self.assertAlmostEqual(transformed.t, 2.25)
        self.assertEqual(transformed.c, 1.0)

    def test_lorentz_transform_xy(self):
        """Test Lorentz transformation with 2D velocity"""
        event = SpacetimeEvent(x=2, y=3, t=5, c=1)

        # Use moderate speed to avoid numerical issues
        vx, vy = 0.3, 0.4  # Speed = 0.5 < c
        transformed = event.lorentz_transform_xy(vx, vy)

        # Check that transformed object is SpacetimeEvent
        self.assertIsInstance(transformed, SpacetimeEvent)
        self.assertEqual(transformed.c, 1.0)

    def test_lorentz_transform_xy_speed_limit(self):
        """Test Lorentz transform with speed >= c should raise ValueError"""
        event = SpacetimeEvent(x=2, y=3, t=5, c=1)

        # v = (0.6, 0.8) has magnitude 1.0 = c
        with self.assertRaises(ValueError):
            event.lorentz_transform_xy(0.6, 0.8)

    def test_lorentz_transform_invariant(self):
        """Test that spacetime interval is invariant under Lorentz transform"""
        event1 = SpacetimeEvent(0, 0, 0, c=1)
        event2 = SpacetimeEvent(1, 2, 3, c=1)

        # Original interval
        original_interval = event1.interval_to(event2)

        # Transform both events
        v = 0.7
        event1_transformed = event1.lorentz_transform(v)
        event2_transformed = event2.lorentz_transform(v)

        # Transformed interval
        transformed_interval = event1_transformed.interval_to(event2_transformed)

        # Interval should be invariant
        self.assertAlmostEqual(original_interval, transformed_interval)

    def test_move_basic(self):
        """Test basic movement in spacetime"""
        event = SpacetimeEvent(x=0, y=0, t=0, c=1)
        moved = event.move(vx=0.6, vy=0, duration=3)

        # x = 0 + 0.6*3 = 1.8, y = 0, t = 0 + 3 = 3
        self.assertAlmostEqual(moved.x, 1.8)
        self.assertAlmostEqual(moved.y, 0.0)
        self.assertAlmostEqual(moved.t, 3.0)
        self.assertEqual(moved.c, 1.0)

    def test_move_speed_limit(self):
        """Test move with speed >= c should raise ValueError"""
        event = SpacetimeEvent(x=0, y=0, t=0, c=1)

        with self.assertRaises(ValueError):
            event.move(vx=1.0, vy=0, duration=1)  # Speed = c

        with self.assertRaises(ValueError):
            event.move(vx=0.8, vy=0.6, duration=1)  # Speed = 1.0 > c

    def test_move_with_velocity_components(self):
        """Test move with both x and y velocity components"""
        event = SpacetimeEvent(x=1, y=2, t=0, c=1)
        moved = event.move(vx=0.3, vy=0.4, duration=2)

        # x = 1 + 0.3*2 = 1.6, y = 2 + 0.4*2 = 2.8, t = 0 + 2 = 2
        self.assertAlmostEqual(moved.x, 1.6)
        self.assertAlmostEqual(moved.y, 2.8)
        self.assertAlmostEqual(moved.t, 2.0)

    def test_boost_and_move_basic(self):
        """Test boost_and_move method"""
        event = SpacetimeEvent(x=0, y=0, t=0, c=1)
        final = event.boost_and_move(boost_vx=0.6, boost_vy=0, local_duration=2)

        # Should get time dilation: t' = gamma * local_duration = 1.25 * 2 = 2.5
        # But also spatial displacement from boost
        self.assertAlmostEqual(final.t, 2.5, places=10)

    def test_boost_and_move_time_dilation(self):
        """Test time dilation effect in boost_and_move"""
        # For v=0.8c, gamma = 1/√(1-0.64) = 1/0.6 ≈ 1.6667
        # Local duration 1 second should give ~1.6667 seconds in original frame
        event = SpacetimeEvent(x=0, y=0, t=0, c=1)
        final = event.boost_and_move(boost_vx=0.8, boost_vy=0, local_duration=1)

        self.assertAlmostEqual(final.t, 1.6666666666666667, places=10)

    def test_boost_and_move_composition(self):
        """Test that boost_and_move composes correctly"""
        event = SpacetimeEvent(x=1, y=2, t=3, c=1)

        # Do boost_and_move
        final = event.boost_and_move(boost_vx=0.5, boost_vy=0, local_duration=2)

        # Should be different from original (y may stay the same for x-only boost)
        self.assertNotEqual(final.x, event.x)
        self.assertNotEqual(final.t, event.t)
        # y may remain unchanged for boost in x direction only

    def test_repr(self):
        """Test string representation"""
        event = SpacetimeEvent(x=1.5, y=2.5, t=3.5, c=1.0)
        repr_str = repr(event)

        self.assertIn("SpacetimeEvent", repr_str)
        self.assertIn("x=1.5", repr_str)
        self.assertIn("y=2.5", repr_str)
        self.assertIn("t=3.5", repr_str)
        self.assertIn("c=1.0", repr_str)


class TestSpacetimeEventIntegration(unittest.TestCase):
    """Integration tests for SpacetimeEvent"""

    def test_causality_preservation(self):
        """Test that causality is preserved under Lorentz transformation"""
        # Event A and B with A causally before B
        event_A = SpacetimeEvent(0, 0, 0, c=1)
        event_B = SpacetimeEvent(1, 1, 2, c=1)  # Timelike, B in A's future

        # Transform to moving frame
        v = 0.7
        event_A_transformed = event_A.lorentz_transform(v)
        event_B_transformed = event_B.lorentz_transform(v)

        # Causality should be preserved
        self.assertTrue(event_B_transformed.is_in_future_of(event_A_transformed))

    def test_light_cone_invariance(self):
        """Test that events on light cone remain on light cone after transform"""
        # Event on light cone: x² + y² = t²
        event = SpacetimeEvent(3, 4, 5, c=1)

        # Transform
        transformed = event.lorentz_transform(v=0.6)

        # Check it's still lightlike relative to origin
        origin = SpacetimeEvent(0, 0, 0, c=1)
        origin_transformed = origin.lorentz_transform(v=0.6)

        interval = origin_transformed.interval_to(transformed)
        self.assertAlmostEqual(interval, 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
