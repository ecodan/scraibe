import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock, DEFAULT
from pathlib import Path
import json

from langchain_aws import ChatBedrock
from langchain_ollama import ChatOllama

from src.agents.actor import CreativeMode
from src.agents.author import Author
from src.agents.critic import Critic
from src.agents.editor import Editor
from src.agents.human import Human
from src.conductor import PaperbackWriter, Conductor
from src.utils import StoryContext


class FauxConductor(Conductor):

    def _post_init(self, **kwargs):
        self.creative_mode = "TEST"
        self.author = MagicMock(Author)
        self.editor = MagicMock(Editor)
        self.critic = MagicMock(Critic)
        self.human = MagicMock(Human)

    def _do_develop_concept(self, concept_dir: Path, **kwargs):
        pass

    def _do_draft_narrative(self, concept_dir: Path, **kwargs):
        pass


class TestConductor(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_working_dir")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        # Clean up test directory
        if self.test_dir.exists():
            for file in self.test_dir.glob("**/*"):
                if file.is_file():
                    file.unlink()
            self.test_dir.rmdir()

    @patch('src.prompt_manager.PromptManager')
    def test_conductor_initialization(self, mock_prompt_manager):
        """Test basic conductor initialization"""

        conductor = FauxConductor(
            working_dir=self.test_dir,
        )

        self.assertIsNotNone(conductor.prompt_manager)
        self.assertEqual(conductor.working_dir, self.test_dir)
        self.assertIsInstance(conductor.author, MagicMock)
        self.assertEqual(conductor.author.__class__, Author)
        self.assertIsInstance(conductor.editor, MagicMock)
        self.assertEqual(conductor.editor.__class__, Editor)
        self.assertIsInstance(conductor.critic, MagicMock)
        self.assertEqual(conductor.critic.__class__, Critic)
        self.assertIsInstance(conductor.human, MagicMock)
        self.assertEqual(conductor.human.__class__, Human)

    def test_conductor_stop(self):
        """Test that stop method calls stop on all agents"""

        conductor = FauxConductor(
            working_dir=self.test_dir,
        )

        conductor._stop()

        conductor.author.stop.assert_called_once()
        conductor.author.stop.assert_called_once()
        conductor.author.stop.assert_called_once()
        conductor.author.stop.assert_called_once()


class TestPaperbackWriter(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test_paperback_writer_local_init(self):
        """Test PaperbackWriter initialization with local environment"""
        with tempfile.TemporaryDirectory() as working_dir:
            writer = PaperbackWriter(working_dir=working_dir, env='local')

            self.assertEqual(writer.creative_mode, CreativeMode.AUTHOR_MODE)
            self.assertIsInstance(writer.author, Author)
            self.assertIsInstance(writer.editor, Editor)
            self.assertIsInstance(writer.critic, Critic)
            self.assertIsInstance(writer.human, Human)
            self.assertIsInstance(writer.author.llm, ChatOllama)
            self.assertIsInstance(writer.critic.llm, ChatOllama)
            self.assertIsInstance(writer.editor.llm, ChatOllama)

    @patch('langchain_aws.ChatBedrock')
    @patch('src.prompt_manager.PromptManager')
    def test_paperback_writer_bedrock_init(self, mock_prompt_manager, mock_chat_bedrock):
        with tempfile.TemporaryDirectory() as working_dir:
            work_dir_path = Path(working_dir)
            """Test PaperbackWriter initialization with bedrock environment"""
            writer = PaperbackWriter(working_dir=working_dir, env='bedrock')

            self.assertEqual(writer.creative_mode, CreativeMode.AUTHOR_MODE)
            self.assertEqual(writer.author.llm.__class__, ChatBedrock)

    def test_write_chapter(self):
        with tempfile.TemporaryDirectory() as working_dir:
            """Test chapter writing functionality"""
            writer = PaperbackWriter(working_dir=working_dir)

            mock_author_instance = MagicMock(spec=Author)
            mock_author_instance.write_section.return_value = "Test page content"
            writer.author = mock_author_instance

            context = StoryContext()
            context.concept = "Test concept"

            chapter = writer._write_chapter(
                context,
                pages_per_chapter=2,
                words_per_page=100,
                previous_chapter_summaries=["Chapter 1 summary"]
            )

            self.assertEqual("Test page content Test page content", chapter)
            self.assertEqual(mock_author_instance.write_section.call_count, 2)

    def test_do_develop_concept(self):
        with tempfile.TemporaryDirectory() as working_dir:
            work_dir_path = Path(working_dir)

            """Test concept development process"""
            writer = PaperbackWriter(working_dir=working_dir)
            mock_human_instance = MagicMock(spec=Human)
            mock_author_instance = MagicMock(spec=Author)
            mock_critic_instance = MagicMock(spec=Critic)

            writer.author = mock_author_instance
            writer.critic = mock_critic_instance
            writer.human = mock_human_instance

            mock_human_instance.prompt_user.side_effect = ["fantasy", "wizard story", "3"]
            mock_human_instance.prompt_user_select.return_value = (0, "selected idea")
            mock_author_instance.ideate.return_value = "test idea"
            mock_author_instance.develop_plot.return_value = "test plot"
            mock_author_instance.develop_themes.return_value = "test themes"
            mock_author_instance.develop_characters.return_value = "test characters"
            mock_author_instance.develop_world.return_value = "test world"
            mock_author_instance.develop_storyline.return_value = "test storyline"
            mock_author_instance.summarize_concept.return_value = "summary of concept"
            mock_critic_instance.critique_concept.return_value = "test critique"

            concept_dir = work_dir_path / "concepts" / "test_time"
            concept_dir.mkdir(parents=True)
            writer._do_develop_concept(concept_dir)

            mock_author_instance.ideate.assert_called()
            mock_author_instance.develop_plot.assert_called()
            mock_author_instance.develop_themes.assert_called()
            mock_author_instance.develop_characters.assert_called()
            mock_author_instance.develop_world.assert_called()
            mock_author_instance.develop_storyline.assert_called()
            mock_critic_instance.critique_concept.assert_called()
