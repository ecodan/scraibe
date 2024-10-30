import unittest
from pathlib import Path
from unittest.mock import patch
import sys
import io

from src.agents.actor import CreativeMode
from src.agents.human import Human
from src.prompt_manager import PromptManager


class TestHuman(unittest.TestCase):
    def setUp(self):
        fdir = Path(__file__).parent
        prompt_file_path: Path = fdir.parent.parent / "src" / "prompts" / "prompts.toml"
        assert prompt_file_path.is_file()
        self.prompt_manager: PromptManager = PromptManager(prompt_file_path)
        """Set up test cases"""
        self.human = Human(prompt_manager=self.prompt_manager, creative_mode=CreativeMode.AUTHOR_MODE)

    @patch('builtins.input', return_value='test input')
    def test_prompt_user(self, mock_input):
        """Test prompt_user method"""
        result = self.human.prompt_user("Enter something: ")

        # Verify input was called with correct prompt
        mock_input.assert_called_once_with("Enter something: ")
        # Verify return value
        self.assertEqual(result, 'test input')

    @patch('builtins.input', return_value='2')
    def test_prompt_user_select(self, mock_input):
        """Test prompt_user_select method"""
        options = ['Option A', 'Option B', 'Option C']
        expected_prompt = "Choose one of the following options:\n1: Option A\n2: Option B\n3: Option C\n"

        selection, chosen_option = self.human.prompt_user_select(options)

        # Verify input was called with correct prompt
        mock_input.assert_called_once_with(expected_prompt)
        # Verify return values
        self.assertEqual(selection, 2)
        self.assertEqual(chosen_option, 'Option B')

    @patch('builtins.input', side_effect=['fantasy', 'dragon story', '3'])
    def test_get_starter(self, mock_input):
        """Test get_starter method"""
        result = self.human.get_starter()

        # Verify input was called three times
        self.assertEqual(mock_input.call_count, 3)
        # Verify the returned dictionary
        expected_result = {
            "genre": "fantasy",
            "idea": "dragon story",
            "num_concepts": 3
        }
        self.assertEqual(result, expected_result)

    @patch('builtins.input', return_value='2')
    def test_select_idea(self, mock_input):
        """Test select_idea method"""
        # Redirect stdout to capture output
        captured_output = io.StringIO()
        sys.stdout = captured_output

        ideas = ['Idea 1', 'Idea 2', 'Idea 3']
        result = self.human.select_idea(ideas)

        # Restore stdout
        sys.stdout = sys.__stdout__

        # Verify the output format
        expected_output = "select from one of the following:\n1: Idea 1\n2: Idea 2\n3: Idea 3\n"
        self.assertEqual(captured_output.getvalue(), expected_output)

        # Verify the selected idea
        self.assertEqual(result, 'Idea 2')

    # Error cases
    @patch('builtins.input', return_value='invalid')
    def test_prompt_user_select_invalid_input(self, mock_input):
        """Test prompt_user_select with invalid input"""
        options = ['Option A', 'Option B']

        with self.assertRaises(ValueError):
            self.human.prompt_user_select(options)

    @patch('builtins.input', return_value='4')
    def test_prompt_user_select_out_of_range(self, mock_input):
        """Test prompt_user_select with out of range input"""
        options = ['Option A', 'Option B']

        with self.assertRaises(IndexError):
            self.human.prompt_user_select(options)

    @patch('builtins.input', side_effect=['fantasy', 'dragon story', 'invalid'])
    def test_get_starter_invalid_num_concepts(self, mock_input):
        """Test get_starter with invalid number of concepts"""
        with self.assertRaises(ValueError):
            self.human.get_starter()

    @patch('builtins.input', return_value='invalid')
    def test_select_idea_invalid_input(self, mock_input):
        """Test select_idea with invalid input"""
        ideas = ['Idea 1', 'Idea 2']

        with self.assertRaises(ValueError):
            self.human.select_idea(ideas)

