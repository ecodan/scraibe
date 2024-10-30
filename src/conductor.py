import logging
import os
from abc import abstractmethod, ABCMeta
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path

from langchain_aws import ChatBedrock
from langchain_ollama import ChatOllama

from src.agents.actor import CreativeMode
from src.agents.author import Author
from src.agents.critic import Critic
from src.agents.editor import Editor
from src.agents.human import Human
from src.prompt_manager import PromptManager
from src.utils import StoryContext, utc_as_string

logger: Logger = logging.getLogger("scrAIbe")


@dataclass
class Conductor(metaclass=ABCMeta):
    """
    Base class for orchestrating operations between author, editor and critic agents and humans.
    Provides facilities for prompts and a working dir.
    Hands off initialization of agents to the child class to allow for LLM selection.

    Child classes must override:
    - _post_init() - initialize the agents
    - _do_develop_concept() - create the overall concept (plot, storyline, characters, etc.) and puts artifacts in a working dir.
    - _do_develop_narrative() - take the concept and actually write the narrative.

    All output for a specific project should be written into the project working directory.

    """
    working_dir: str
    env: str = field(default="local")
    working_dir_path: Path = field(init=False)
    author: Author = field(init=False)
    editor: Editor = field(init=False)
    critic: Critic = field(init=False)
    human: Human = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()

        # load prompts
        src_path: Path = Path(os.path.dirname(__file__))
        prompt_file_path: Path = src_path / "prompts" / "prompts.toml"
        assert prompt_file_path.is_file()
        self.prompt_manager: PromptManager = PromptManager(prompt_file_path)

        # validate working dir
        self.working_dir_path = Path(self.working_dir)
        assert self.working_dir_path.is_dir(), f"{self.working_dir_path} is not a valid directory"

        # hand off to child class to finish init
        self._post_init()

    def develop_concept(self, **kwargs) -> Path:
        try:
            out_dir: Path = self.working_dir_path / 'concepts' / f"{utc_as_string()}"
            out_dir.mkdir(parents=True, exist_ok=False)

            return self._do_develop_concept(out_dir, **kwargs)
        except Exception as e:
            logger.exception(e)
        finally:
            self._stop()
            logger.info("done!")

    def draft_narrative(self, concept_dir_path: Path, **kwargs):
        try:
            self._do_draft_narrative(concept_dir_path, **kwargs)
        except Exception as e:
            logger.exception(e)
        finally:
            self._stop()
            logger.info("done!")

    def _stop(self):
        logger.info("stopping. shutting down agents...")
        if self.author:
            self.author.stop()
        if self.critic:
            self.critic.stop()
        if self.editor:
            self.editor.stop()
        if self.human:
            self.human.stop()

    @abstractmethod
    def _post_init(self):
        """
        Subclasses should initialize LLMs and perform any other sub-class unique initalizations here.
        """
        pass

    @abstractmethod
    def _do_develop_concept(self, concept_dir: Path, **kwargs):
        """
        Override to go from rought idea to fully baked concept.
        """
        pass

    @abstractmethod
    def _do_draft_narrative(self, concept_dir: Path, **kwargs):
        """
        Override to take the concept to a full narrative.
        """
        pass


class PaperbackWriter(Conductor):
    """
    Implementation of conductor optimized for producing long-form fiction.
    """

    def _post_init(self):

        # this is used by the prompt system to find the right root node
        self.creative_mode = CreativeMode.AUTHOR_MODE

        # create LLM
        logger.info(f"running creative mode: {self.creative_mode} and env: {self.env}")
        if self.env == 'local':
            llm: ChatOllama = ChatOllama(
                model="llama3.2",
                temperature=0.8,
                num_predict=256,
            )
            llm2: ChatOllama = ChatOllama(
                model="llama3.2",
                temperature=0.8,
                num_predict=256,
            )
        elif self.env == 'bedrock':
            llm: ChatBedrock = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0")
            llm2: ChatBedrock = ChatBedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
        else:
            raise ValueError(f"invalid environment {self.env}")

        # start agents
        self.author = Author(
            llm=llm,
            prompt_manager=self.prompt_manager,
            creative_mode=self.creative_mode,
            identity_prompt_preamble="You are a thoughtful and skilled fiction writer."
        )

        self.critic = Critic(
            llm=llm2,
            prompt_manager=self.prompt_manager,
            creative_mode=self.creative_mode,
            identity_prompt_preamble="You are a thoughtful and skilled literary critic who likes to help writers improve."
        )

        self.editor = Editor(
            llm=llm2,
            prompt_manager=self.prompt_manager,
            creative_mode=self.creative_mode,
            identity_prompt_preamble="You are a skilled editor who helps writers refine their work."
        )

        self.human = Human(
            prompt_manager=self.prompt_manager,
            creative_mode=self.creative_mode
        )

    def _do_develop_concept(self, concept_dir: Path, **kwargs):
        """
        Order of operations
        - Get starter idea from human
        - Generate concepts
        - Human chooses winner or none
        - Generate:
            - Plot outline
            - Characters
            - World
            - Themes
            - Storyline
        - Review with critic
        - Update all of the above
        """

        # get seed ideas and genre from the human
        genre: str = self.human.prompt_user(
            self.prompt_manager.get_prompt([self.creative_mode.value, "HUMAN", "GENRE"]))
        starter: str = self.human.prompt_user(
            self.prompt_manager.get_prompt([self.creative_mode.value, "HUMAN", "STARTER"]))
        num_concepts: int = int(self.human.prompt_user(
            self.prompt_manager.get_prompt([self.creative_mode.value, "HUMAN", "NUM_IDEAS"])))

        # generate ideas
        futures: list[Future] = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            for i in range(num_concepts):
                futures.append(executor.submit(self.author.ideate, genre, starter))
        ideas: list = [f.result() for f in futures]

        # human selects idea to work with
        idx, selected_idea = self.human.prompt_user_select(ideas)

        # generate all of the elements of the story
        context: StoryContext = StoryContext()
        context.concept = selected_idea
        context.plot = self.author.develop_plot(context)
        context.themes = self.author.develop_themes(context)
        context.characters = self.author.develop_characters(context)
        context.world = self.author.develop_world(context)
        context.storyline = self.author.develop_storyline(context)

        # output context (in progress)
        with open(concept_dir / "concept.json", "w") as f:
            f.write(context.marshall())

        # critique the first pass
        critique: str = self.critic.critique_concept(context)

        # update the story elements based on the critique
        context.plot = self.author.develop_plot(context, critique=critique)
        context.characters = self.author.develop_characters(context, critique=critique)
        context.world = self.author.develop_world(context, critique=critique)
        context.storyline = self.author.develop_storyline(context, critique=critique)

        # output context (final)
        with open(concept_dir / "context.json", "w") as f:
            f.write(context.marshall())

        # generate a markdown summary
        summary: str = self.author.summarize_concept(context)
        with open(concept_dir / "summary.md", "w") as f:
            f.write(summary)

    def _write_chapter(self, context: StoryContext, pages_per_chapter: int, words_per_page: int,
                       previous_chapter_summaries: list) -> str:
        """
        Experimental; writes the next section of the doc.
        """
        book_summary = "".join(
            [f"Chapter {idx + 1}: {chapter}\n" for idx, chapter in enumerate(previous_chapter_summaries)])
        # first pass
        pages: list = []
        for page in range(1, pages_per_chapter + 1):
            chapter_so_far: str = "".join([f"{p}\n" for idx, p in enumerate(pages)])
            pages.append(self.author.write_section(context, words_per_page, page, pages_per_chapter, chapter_so_far,
                                                   book_summary))
        chapter: str = " ".join(pages)
        return chapter

    def _do_draft_narrative(self, concept_dir: Path, **kwargs):
        """
        Experimental; turns the concept into a full narrative. Works well for a single chapter, but struggling to
        keep continuity and flow across sections and chapters.
        """

        logger.info(f"draft narrative for {concept_dir}")
        num_pages: int = 240
        num_chapters: int = 12
        words_per_page: int = 250
        pages_per_chapter: int = num_pages // num_chapters

        with open(concept_dir / "context.json", "r") as f:
            context = StoryContext.unmarshall(f.read())

        chapter_summaries: list = []
        chapters: list = []
        for chapter in range(1, num_chapters + 1):
            content: str = self._write_chapter(context, pages_per_chapter, words_per_page, chapter_summaries)
            chapters.append(content)
            with open(concept_dir / f"chapter_{chapter}.txt", "w") as f:
                f.write(content)

        with open(concept_dir / f"full_narrative.txt", "w") as f:
            f.write("\n\n".join(chapters))


class HistoryPodcaster(Conductor):
    """
    Experimental.
    """

    def _post_init(self):

        self.creative_mode = CreativeMode.PODCAST_MODE

        # create LLM
        logger.info(f"running creative mode: {self.creative_mode} and env: {self.env}")
        if self.env == 'local':
            llm: ChatOllama = ChatOllama(
                model="llama3.2",
                temperature=0.8,
                num_predict=256,
            )
            llm2: ChatOllama = ChatOllama(
                model="llama3.2",
                temperature=0.8,
                num_predict=256,
            )
        elif self.env == 'bedrock':
            llm: ChatBedrock = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0")
            llm2: ChatBedrock = ChatBedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
        else:
            raise ValueError(f"invalid environment {self.env}")

        # start agents
        self.author = Author(
            llm=llm,
            prompt_manager=self.prompt_manager,
            creative_mode=self.creative_mode,
            identity_prompt_preamble="You are the assistant to a creative podcast producer."
        )

        self.critic = Critic(
            llm=llm2,
            prompt_manager=self.prompt_manager,
            creative_mode=self.creative_mode,
            identity_prompt_preamble="You are a thoughtful and skilled critic on podcasts."
        )

        self.editor = Editor(
            llm=llm2,
            prompt_manager=self.prompt_manager,
            creative_mode=self.creative_mode,
            identity_prompt_preamble="You are the assistant to a creative podcast producer."
        )

        self.human = Human(
            prompt_manager=self.prompt_manager,
            creative_mode=self.creative_mode,
        )

    def _do_develop_concept(self, concept_dir: Path, **kwargs):
        genre: str = self.human.prompt_user(self.prompt_manager.get_prompt([self.creative_mode.value, "HUMAN", "GENRE"])).get()
        # genre: str = "historical battles"
        starter: str = self.human.prompt_user(
            self.prompt_manager.get_prompt([self.creative_mode.value, "HUMAN", "STARTER"]))
        num_concepts: int = int(self.human.prompt_user(
            self.prompt_manager.get_prompt([self.creative_mode.value, "HUMAN", "NUM_IDEAS"])))

        futures: list = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            for i in range(num_concepts):
                futures.append(executor.submit(self.author.ideate, genre, starter))
        ideas: list = [f.get() for f in futures]
        idx, selected_idea = self.human.prompt_user_select(ideas)

        context: StoryContext = StoryContext()
        context.concept = selected_idea
        context.plot = self.author.develop_plot(context)
        context.themes = self.author.develop_themes(context)
        context.characters = self.author.develop_characters(context)
        # context.world = self.author.develop_world(context)
        context.storyline = self.author.develop_storyline(context)

        # output context
        with open(concept_dir / "concept.json", "w") as f:
            f.write(context.marshall())

        critique: str = self.critic.critique_concept(context)
        context.plot = self.author.develop_plot(context, critique=critique)
        context.characters = self.author.develop_characters(context, critique=critique)
        # context.world = self.author.develop_world(context, critique=critique)
        context.storyline = self.author.develop_storyline(context, critique=critique)

        # output context
        with open(concept_dir / "context.json", "w") as f:
            f.write(context.marshall())

        summary: str = self.author.summarize_concept(context)

        # output summary
        with open(concept_dir / "summary.md", "w") as f:
            f.write(summary)

    def _write_segment(self, context: StoryContext, num_words: int, previous_segment_summaries: list) -> str:
        full_summary = "".join(
            [f"Segment {idx + 1}: {segment}\n" for idx, segment in enumerate(previous_segment_summaries)])
        # first pass
        content = self.author.write_section(context, num_words, 1, 1, "", full_summary)
        return content

    def _do_draft_narrative(self, concept_dir: Path, **kwargs):
        logger.info(f"draft narrative for {concept_dir}")
        num_segments: int = 4
        words_per_segment: int = 1000

        with open(concept_dir / "context.json", "r") as f:
            context = StoryContext.unmarshall(f.read())

        segment_summaries: list = []
        segments: list = []
        for chapter in range(1, num_segments + 1):
            content: str = self._write_segment(context, words_per_segment, segment_summaries)
            segments.append(content)
            with open(concept_dir / f"segment_{chapter}.txt", "w") as f:
                f.write(content)

        with open(concept_dir / f"podcast.txt", "w") as f:
            f.write("\n\n".join(segments))
