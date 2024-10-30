import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama

from src import prompt_manager
from src.agents.actor import CreativeMode
from src.agents.author import Author
from src.prompt_manager import PromptManager
from src.utils import StoryContext


class TestAuthor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.mock_llm = Mock(spec=ChatOllama)
        self.mock_prompt_manager = Mock(spec=PromptManager)

        # Create Author instance with mocked dependencies
        self.author = Author(llm=self.mock_llm, prompt_manager=self.mock_prompt_manager, creative_mode=CreativeMode.AUTHOR_MODE)
        self.author.llm = self.mock_llm

        # Setup common test data
        self.test_context = StoryContext(
            concept="A story about a magical library",
            plot="The library contains books that come to life",
            themes="Magic, Knowledge, Adventure",
            characters="Librarian Sarah, Living Books",
            world="Modern day with magical elements",
            storyline="Sarah discovers the library's secret"
        )

    def test_ideate(self):
        """Test ideate method"""
        # Setup mock responses
        mock_json_response = '["idea1", "idea2", "idea3"]'
        self.mock_llm.invoke.return_value = AIMessage(content=mock_json_response)
        self.mock_prompt_manager.get_prompt.return_value = "test prompt"

        # Test the method
        result = self.author.ideate("fantasy", "magical library")

        # Verify the results
        self.assertEqual(result, mock_json_response)
        self.mock_prompt_manager.get_prompt.assert_called_with([self.author.creative_mode, "IDEATE"])
        self.mock_llm.invoke.assert_called_once()

    def test_develop_plot_without_critique(self):
        """Test develop_plot method without critique"""
        # Setup mock response
        mock_plot = "Detailed plot description..."
        self.mock_llm.invoke.return_value = AIMessage(content=mock_plot)
        self.mock_prompt_manager.get_prompt.return_value = "test prompt"

        # Test the method
        result = self.author.develop_plot(self.test_context)

        # Verify results
        self.assertEqual(result, mock_plot)
        self.mock_prompt_manager.get_prompt.assert_any_call([self.author.creative_mode, "DEVELOP_PLOT", "BASE"])
        self.mock_prompt_manager.get_prompt.assert_any_call([self.author.creative_mode, "DEVELOP_PLOT", "UNASSISTED"])

    def test_develop_plot_with_critique(self):
        """Test develop_plot method with critique"""
        # Setup mock response
        mock_plot = "Revised plot description..."
        self.mock_llm.invoke.return_value = AIMessage(content=mock_plot)
        self.mock_prompt_manager.get_prompt.return_value = "test prompt"

        # Test the method
        result = self.author.develop_plot(self.test_context, critique="Need more conflict")

        # Verify results
        self.assertEqual(result, mock_plot)
        self.mock_prompt_manager.get_prompt.assert_any_call([self.author.creative_mode, "DEVELOP_PLOT", "BASE"])
        self.mock_prompt_manager.get_prompt.assert_any_call(
            [self.author.creative_mode, "DEVELOP_PLOT", "WITH_FEEDBACK"])

    def test_develop_themes(self):
        """Test develop_themes method"""
        # Setup mock response
        mock_themes = "Theme analysis..."
        self.mock_llm.invoke.return_value = AIMessage(content=mock_themes)
        self.mock_prompt_manager.get_prompt.return_value = "test prompt"

        # Test the method
        result = self.author.develop_themes(self.test_context)

        # Verify results
        self.assertEqual(result, mock_themes)
        self.mock_prompt_manager.get_prompt.assert_called_with(
            [self.author.creative_mode, "DEVELOP_THEME", "UNASSISTED"])

    def test_develop_characters_without_critique(self):
        """Test develop_characters method without critique"""
        # Setup mock response
        mock_characters = "Character descriptions..."
        self.mock_llm.invoke.return_value = AIMessage(content=mock_characters)
        self.mock_prompt_manager.get_prompt.return_value = "test prompt"

        # Test the method
        result = self.author.develop_characters(self.test_context)

        # Verify results
        self.assertEqual(result, mock_characters)
        self.mock_prompt_manager.get_prompt.assert_any_call([self.author.creative_mode, "DEVELOP_CHARACTERS", "BASE"])
        self.mock_prompt_manager.get_prompt.assert_any_call(
            [self.author.creative_mode, "DEVELOP_CHARACTERS", "UNASSISTED"])

    def test_develop_world_without_critique(self):
        """Test develop_world method without critique"""
        # Setup mock response
        mock_world = "World building description..."
        self.mock_llm.invoke.return_value = AIMessage(content=mock_world)
        self.mock_prompt_manager.get_prompt.return_value = "test prompt"

        # Test the method
        result = self.author.develop_world(self.test_context)

        # Verify results
        self.assertEqual(result, mock_world)
        self.mock_prompt_manager.get_prompt.assert_any_call([self.author.creative_mode, "DEVELOP_WORLD", "BASE"])
        self.mock_prompt_manager.get_prompt.assert_any_call([self.author.creative_mode, "DEVELOP_WORLD", "UNASSISTED"])

    def test_develop_storyline_without_critique(self):
        """Test develop_storyline method without critique"""
        # Setup mock response
        mock_storyline = "Detailed storyline..."
        self.mock_llm.invoke.return_value = AIMessage(content=mock_storyline)
        self.mock_prompt_manager.get_prompt.return_value = "test prompt"

        # Test the method
        result = self.author.develop_storyline(self.test_context)

        # Verify results
        self.assertEqual(result, mock_storyline)
        self.mock_prompt_manager.get_prompt.assert_any_call([self.author.creative_mode, "DEVELOP_STORYLINE", "BASE"])
        self.mock_prompt_manager.get_prompt.assert_any_call(
            [self.author.creative_mode, "DEVELOP_STORYLINE", "UNASSISTED"])

    def test_summarize_concept(self):
        """Test summarize_concept method"""
        # Setup mock response
        mock_summary = "Concept summary..."
        self.mock_llm.invoke.return_value = AIMessage(content=mock_summary)
        self.mock_prompt_manager.get_prompt.return_value = "test prompt"

        # Test the method
        result = self.author.summarize_concept(self.test_context)

        # Verify results
        self.assertEqual(result, mock_summary)
        self.mock_prompt_manager.get_prompt.assert_called_with([self.author.creative_mode, "SUMMARIZE_CONCEPT"])

    def test_write_section(self):
        """Test write_section method"""
        # Setup mock response
        mock_section = "Written section content..."
        self.mock_llm.invoke.return_value = AIMessage(content=mock_section)
        self.mock_prompt_manager.get_prompt.return_value = "test prompt"

        # Test the method
        result = self.author.write_section(
            context=self.test_context,
            num_words=1000,
            section_number=1,
            total_sections=3,
            preceding_sections="Previous content...",
            extended_context="Additional context..."
        )

        # Verify results
        self.assertEqual(result, mock_section)
        self.mock_prompt_manager.get_prompt.assert_called_with([self.author.creative_mode, "DRAFT", "SECTION"])

