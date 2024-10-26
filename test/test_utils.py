import unittest
import json
from src.utils import StoryContext


class TestStoryContext(unittest.TestCase):
    def setUp(self):
        self.story_context = StoryContext(
            concept="A world where dreams come true",
            plot="A young girl discovers she can control her dreams",
            characters="Emma, the dreamer; Mr. Sandman, the dream guide",
            storyline="Emma learns to navigate the dream world and faces challenges",
            world="A mix of reality and dreamscapes",
            themes="Self-discovery, imagination, overcoming fears"
        )

    def test_initialization(self):
        self.assertEqual(self.story_context.concept, "A world where dreams come true")
        self.assertEqual(self.story_context.plot, "A young girl discovers she can control her dreams")
        self.assertEqual(self.story_context.characters, "Emma, the dreamer; Mr. Sandman, the dream guide")
        self.assertEqual(self.story_context.storyline, "Emma learns to navigate the dream world and faces challenges")
        self.assertEqual(self.story_context.world, "A mix of reality and dreamscapes")
        self.assertEqual(self.story_context.themes, "Self-discovery, imagination, overcoming fears")

    def test_str_representation(self):
        expected_str = (
            "StoryContext:\n"
            "\tconcept: A world where dreams come true\n"
            "\tplot: A young girl discovers she can control her dreams\n"
            "\tthemes: Self-discovery, imagination, overcoming fears\n"
            "\tcharacters: Emma, the dreamer; Mr. Sandman, the dream guide\n"
            "\tworld: A mix of reality and dreamscapes\n"
            "\tstoryline: Emma learns to navigate the dream world and faces challenges"
        )
        self.assertEqual(str(self.story_context), expected_str)

    def test_marshall(self):
        marshalled = self.story_context.marshall()
        self.assertIsInstance(marshalled, str)

        # Verify that the marshalled string can be parsed as JSON
        parsed_json = json.loads(marshalled)
        self.assertIsInstance(parsed_json, dict)

        # Check if all attributes are present in the JSON
        for attr in ["concept", "plot", "characters", "storyline", "world", "themes"]:
            self.assertIn(attr, parsed_json)
            self.assertEqual(parsed_json[attr], getattr(self.story_context, attr))

    def test_unmarshall(self):
        json_str = json.dumps({
            "concept": "A dystopian future",
            "plot": "Survivors fight for resources",
            "characters": "Alex, the leader; Sam, the strategist",
            "storyline": "Group journeys through dangerous wastelands",
            "world": "Post-apocalyptic Earth",
            "themes": "Survival, hope, human nature"
        })

        unmarshalled = StoryContext().unmarshall(json_str)
        self.assertIsInstance(unmarshalled, StoryContext)
        self.assertEqual(unmarshalled.concept, "A dystopian future")
        self.assertEqual(unmarshalled.plot, "Survivors fight for resources")
        self.assertEqual(unmarshalled.characters, "Alex, the leader; Sam, the strategist")
        self.assertEqual(unmarshalled.storyline, "Group journeys through dangerous wastelands")
        self.assertEqual(unmarshalled.world, "Post-apocalyptic Earth")
        self.assertEqual(unmarshalled.themes, "Survival, hope, human nature")

    def test_empty_context(self):
        empty_context = StoryContext()
        self.assertIsNone(empty_context.concept)
        self.assertIsNone(empty_context.plot)
        self.assertIsNone(empty_context.characters)
        self.assertIsNone(empty_context.storyline)
        self.assertIsNone(empty_context.world)
        self.assertIsNone(empty_context.themes)
