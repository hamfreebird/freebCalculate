import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Try to import npc_manager, but skip tests if not available
try:
    from freebirdcal.npc_manager import BaseNPC, NPCState, Position

    NPC_MANAGER_AVAILABLE = True
except ImportError as e:
    NPC_MANAGER_AVAILABLE = False
    print(f"Warning: npc_manager module not available: {e}")


@unittest.skipIf(not NPC_MANAGER_AVAILABLE, "npc_manager module not available")
class TestPosition(unittest.TestCase):
    """Test cases for Position class"""

    def test_position_initialization_default(self):
        """Test Position initialization with default values"""
        pos = Position()
        self.assertEqual(pos.x, 0)
        self.assertEqual(pos.y, 0)
        self.assertEqual(pos.t, 0)

    def test_position_initialization_custom(self):
        """Test Position initialization with custom values"""
        pos = Position(x=10, y=20, t=2023)
        self.assertEqual(pos.x, 10)
        self.assertEqual(pos.y, 20)
        self.assertEqual(pos.t, 2023)

    def test_position_update_all_values(self):
        """Test updating all position values"""
        pos = Position(x=1, y=2, t=3)
        pos.update(x=10, y=20, t=30)
        self.assertEqual(pos.x, 10)
        self.assertEqual(pos.y, 20)
        self.assertEqual(pos.t, 30)

    def test_position_update_partial(self):
        """Test updating partial position values"""
        pos = Position(x=1, y=2, t=3)
        pos.update(x=10)  # Only update x
        self.assertEqual(pos.x, 10)
        self.assertEqual(pos.y, 2)  # Should remain unchanged
        self.assertEqual(pos.t, 3)  # Should remain unchanged

        pos.update(y=20, t=30)  # Update y and t
        self.assertEqual(pos.x, 10)
        self.assertEqual(pos.y, 20)
        self.assertEqual(pos.t, 30)

    def test_position_get_coords(self):
        """Test get_coords method"""
        pos = Position(x=100, y=200, t=2023)
        coords = pos.get_coords()
        self.assertIsInstance(coords, tuple)
        self.assertEqual(len(coords), 3)
        self.assertEqual(coords, (100, 200, 2023))

    def test_position_str_representation(self):
        """Test string representation of Position"""
        pos = Position(x=123, y=456, t=2023)
        str_repr = str(pos)
        self.assertIsInstance(str_repr, str)
        # Check that all values are in the string representation
        self.assertIn("123", str_repr)
        self.assertIn("456", str_repr)
        self.assertIn("2023", str_repr)


@unittest.skipIf(not NPC_MANAGER_AVAILABLE, "npc_manager module not available")
class TestBaseNPC(unittest.TestCase):
    """Test cases for BaseNPC class"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_quotes = [
            "Hello, traveler!",
            "The weather is nice today.",
            "Beware of dragons in the forest.",
        ]

    def test_npc_initialization_minimal(self):
        """Test BaseNPC initialization with minimal parameters"""
        npc = BaseNPC(identifier=1001, name="TestNPC")

        self.assertEqual(npc._identifier, 1001)
        self.assertEqual(npc._name, "TestNPC")
        self.assertIsInstance(npc.position, Position)
        self.assertEqual(npc.position.x, 0)
        self.assertEqual(npc.position.y, 0)
        self.assertEqual(npc.position.t, 0)
        self.assertEqual(npc.state, NPCState.IDLE)
        self.assertIsNone(npc._nickname)
        self.assertEqual(npc._age, 0)
        self.assertIsNone(npc._image_path)
        self.assertEqual(npc._faction, "中立")
        self.assertEqual(npc._quotes, [])

    def test_npc_initialization_all_parameters(self):
        """Test BaseNPC initialization with all parameters"""
        position = Position(x=100, y=200, t=2023)
        npc = BaseNPC(
            identifier=1002,
            name="TestNPC",
            position=position,
            state=NPCState.MOVING,
            nickname="Testy",
            age=30,
            image_path="path/to/image.png",
            faction="Alliance",
            quotes=self.test_quotes,
        )

        self.assertEqual(npc._identifier, 1002)
        self.assertEqual(npc._name, "TestNPC")
        self.assertEqual(npc.position, position)
        self.assertEqual(npc.state, NPCState.MOVING)
        self.assertEqual(npc._nickname, "Testy")
        self.assertEqual(npc._age, 30)
        self.assertEqual(npc._image_path, "path/to/image.png")
        self.assertEqual(npc._faction, "Alliance")
        self.assertEqual(npc._quotes, self.test_quotes)

    def test_npc_move_to(self):
        """Test move_to method"""
        npc = BaseNPC(identifier=1001, name="TestNPC")
        initial_position = npc.position

        # Move to new coordinates
        npc.move_to(150, 250)

        # Check that position was updated
        self.assertEqual(npc.position.x, 150)
        self.assertEqual(npc.position.y, 250)
        # Time should remain unchanged
        self.assertEqual(npc.position.t, initial_position.t)
        # State should be MOVING
        self.assertEqual(npc.state, NPCState.MOVING)

    def test_npc_time_travel(self):
        """Test time_travel method"""
        npc = BaseNPC(identifier=1001, name="TestNPC")
        initial_x = npc.position.x
        initial_y = npc.position.y

        # Travel to new time
        npc.time_travel(2050)

        # Check that time was updated
        self.assertEqual(npc.position.t, 2050)
        # Space coordinates should remain unchanged
        self.assertEqual(npc.position.x, initial_x)
        self.assertEqual(npc.position.y, initial_y)

    def test_npc_speak_with_quotes(self):
        """Test speak method when NPC has quotes"""
        npc = BaseNPC(identifier=1001, name="TestNPC", quotes=self.test_quotes)

        # Mock random.choice to return a specific quote
        with patch("random.choice") as mock_choice:
            mock_choice.return_value = self.test_quotes[0]
            quote = npc.speak()

            # Check that random.choice was called with the quotes
            mock_choice.assert_called_with(self.test_quotes)
            self.assertEqual(quote, self.test_quotes[0])

    def test_npc_speak_no_quotes(self):
        """Test speak method when NPC has no quotes"""
        npc = BaseNPC(identifier=1001, name="TestNPC", quotes=[])

        # Should return a default message
        quote = npc.speak()
        self.assertIsInstance(quote, str)
        self.assertIn(npc._name, quote)

    def test_npc_get_info(self):
        """Test get_info method"""
        npc = BaseNPC(
            identifier=1001,
            name="TestNPC",
            nickname="Testy",
            age=25,
            faction="TestFaction",
        )

        info = npc.get_info()

        self.assertIsInstance(info, dict)
        # Check that all expected keys are present
        expected_keys = [
            "identifier",
            "name",
            "position",
            "state",
            "nickname",
            "age",
            "faction",
        ]
        for key in expected_keys:
            self.assertIn(key, info)

        # Check specific values
        self.assertEqual(info["identifier"], 1001)
        self.assertEqual(info["name"], "TestNPC")
        self.assertEqual(info["nickname"], "Testy")
        self.assertEqual(info["age"], 25)
        self.assertEqual(info["faction"], "TestFaction")
        self.assertIsInstance(info["position"], Position)
        self.assertEqual(info["state"], NPCState.IDLE)

    def test_npc_state_transition(self):
        """Test NPC state transitions"""
        npc = BaseNPC(identifier=1001, name="TestNPC")

        # Initial state should be IDLE
        self.assertEqual(npc.state, NPCState.IDLE)

        # Change state
        npc.state = NPCState.COMBAT
        self.assertEqual(npc.state, NPCState.COMBAT)

        # Change to another state
        npc.state = NPCState.INTERACTING
        self.assertEqual(npc.state, NPCState.INTERACTING)

    def test_npc_identifier_property(self):
        """Test that identifier is accessible"""
        npc = BaseNPC(identifier=9999, name="TestNPC")
        self.assertEqual(npc._identifier, 9999)

    def test_npc_name_property(self):
        """Test that name is accessible"""
        npc = BaseNPC(identifier=1001, name="UniqueName")
        self.assertEqual(npc._name, "UniqueName")

    def test_npc_with_nickname(self):
        """Test NPC with nickname"""
        npc = BaseNPC(identifier=1001, name="TestNPC", nickname="ShortName")
        self.assertEqual(npc._nickname, "ShortName")

        # Test get_info includes nickname
        info = npc.get_info()
        self.assertEqual(info["nickname"], "ShortName")

    def test_npc_without_nickname(self):
        """Test NPC without nickname"""
        npc = BaseNPC(identifier=1001, name="TestNPC")
        self.assertIsNone(npc._nickname)

        # Test get_info handles None nickname
        info = npc.get_info()
        self.assertIsNone(info["nickname"])

    def test_npc_age_property(self):
        """Test age property"""
        npc = BaseNPC(identifier=1001, name="TestNPC", age=42)
        self.assertEqual(npc._age, 42)

    def test_npc_faction_property(self):
        """Test faction property"""
        npc = BaseNPC(identifier=1001, name="TestNPC", faction="Horde")
        self.assertEqual(npc._faction, "Horde")

    def test_npc_state_enum_values(self):
        """Test that NPCState enum has expected values"""
        self.assertEqual(NPCState.IDLE.value, "空闲")
        self.assertEqual(NPCState.MOVING.value, "移动中")
        self.assertEqual(NPCState.INTERACTING.value, "交互中")
        self.assertEqual(NPCState.COMBAT.value, "战斗状态")

    def test_npc_repr_representation(self):
        """Test string representation of BaseNPC"""
        npc = BaseNPC(identifier=1001, name="TestNPC")
        repr_str = repr(npc)
        self.assertIsInstance(repr_str, str)
        # Should contain the class name and identifier
        self.assertIn("BaseNPC", repr_str)
        self.assertIn("1001", repr_str)


if __name__ == "__main__":
    unittest.main()
