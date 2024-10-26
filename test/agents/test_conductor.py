import unittest
from unittest.mock import Mock, patch

from pykka import ActorProxy

from src.conductor import Conductor


class TestConductor(unittest.TestCase):

    def setUp(self):
        self.conductor = Conductor()

    def tearDown(self):
        self.conductor.stop()

    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('json.dumps')
    def test_start_method(self, mock_json_dumps, mock_open):
        # Mock the behavior of all the actor proxies
        self.conductor.human.get_starter.return_value.get.return_value = {
            "genre": "romance",
            "num_concepts": 2,
            "idea": "two puppies in love"
        }
        self.conductor.author.ideate.return_value.get.return_value = ["Idea 1", "Idea 2"]
        self.conductor.human.select_idea.return_value.get.return_value = "Idea 1"
        self.conductor.author.develop_plot.return_value.get.return_value = "Plot"
        self.conductor.author.develop_themes.return_value.get.return_value = "Themes"
        self.conductor.author.develop_characters.return_value.get.return_value = "Characters"
        self.conductor.author.develop_world.return_value.get.return_value = "World"
        self.conductor.author.develop_storyline.return_value.get.return_value = "Storyline"
        self.conductor.critic.critique_concept.return_value.get.return_value = "Critique"
        self.conductor.author.summarize_concept.return_value.get.return_value = "Summary"

        # Call the method under test
        self.conductor.start()

        # Assert that all expected methods were called
        self.conductor.human.get_starter.assert_called_once()
        self.conductor.author.ideate.assert_called_once_with("romance", 2, "two puppies in love")
        self.conductor.human.select_idea.assert_called_once()
        self.conductor.author.develop_plot.assert_called()
        self.conductor.author.develop_themes.assert_called_once()
        self.conductor.author.develop_characters.assert_called()
        self.conductor.author.develop_world.assert_called()
        self.conductor.author.develop_storyline.assert_called()
        self.conductor.critic.critique_concept.assert_called_once()
        self.conductor.author.summarize_concept.assert_called_once()

        # Assert that the file writing operations were called
        self.assertEqual(mock_open.call_count, 2)
        mock_json_dumps.assert_called_once()

        # Assert that all actors were stopped
        self.conductor.author.stop.assert_called_once()
        self.conductor.critic.stop.assert_called_once()
        self.conductor.editor.stop.assert_called_once()
        self.conductor.human.stop.assert_called_once()

    def test_conductor_initialization(self):
        self.assertIsInstance(self.conductor.author, ActorProxy)
        self.assertIsInstance(self.conductor.editor, ActorProxy)
        self.assertIsInstance(self.conductor.critic, ActorProxy)
        self.assertIsInstance(self.conductor.human, ActorProxy)