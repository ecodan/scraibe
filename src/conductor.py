import os
from abc import abstractmethod, ABCMeta
from dataclasses import dataclass
from pathlib import Path

from langchain.chat_models import ChatOllama
from langchain_aws import ChatBedrock
from pykka import ActorProxy

from src.agents.actor import CreativeMode
from src.logutils import create_logger

from src.agents.author import Author
from src.agents.critic import Critic
from src.agents.editor import Editor
from src.agents.human import Human
from src.prompt_manager import PromptManager
from src.utils import StoryContext, utc_as_string

logger = create_logger(__name__)


@dataclass
class Conductor(metaclass=ABCMeta):

    author: ActorProxy[Author]
    editor: ActorProxy[Editor]
    critic: ActorProxy[Critic]
    human: ActorProxy[Human]

    def __init__(self, working_dir: str, **kwargs) -> None:
        super().__init__()

        # load prompts
        src_path: Path = Path(os.path.dirname(__file__))
        prompt_file_path: Path = src_path / "prompts" / "prompts.toml"
        assert prompt_file_path.is_file()
        self.prompt_manager: PromptManager = PromptManager(prompt_file_path)

        # validate working dir
        self.working_dir: Path = Path(working_dir)
        assert self.working_dir.is_dir(), f"{self.working_dir} is not a valid directory"

        # hand off to child class to finish init
        self._post_init(**kwargs)

    def develop_concept(self, **kwargs) -> Path:
        try:
            out_dir: Path = self.working_dir / 'concepts' / f"{utc_as_string()}"
            out_dir.mkdir(parents=True, exist_ok=False)

            self._do_develop_concept(out_dir, **kwargs)
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
    def _post_init(self, **kwargs):
        pass

    @abstractmethod
    def _do_develop_concept(self, concept_dir: Path, **kwargs):
        pass

    @abstractmethod
    def _do_draft_narrative(self, concept_dir: Path, **kwargs):
        pass


class PaperbackWriter(Conductor):

    def _post_init(self, **kwargs):

        self.creative_mode = CreativeMode.AUTHOR_MODE

        # create LLM
        logger.info(f"running creative mode: {self.creative_mode} and env: {kwargs.get('env', 'local')}")
        if kwargs.get('env', 'local') == 'local':
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
        elif kwargs.get('env') == 'bedrock':
            llm: ChatBedrock = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0")
            llm2: ChatBedrock = ChatBedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
        else:
            raise ValueError(f"invalid environment {kwargs.get('env')}")

        # start agents
        self.author = Author.start(
            llm=llm,
            prompt_manager=self.prompt_manager,
            creative_mode=self.creative_mode,
            identity_prompt_preamble="You are a thoughtful and skilled expert in literary fiction."
        ).proxy()

        self.critic = Critic.start(
            llm=llm2,
            prompt_manager=self.prompt_manager,
            creative_mode=self.creative_mode,
            identity_prompt_preamble = "You are a thoughtful and skilled literary critic who likes to help writers improve."
        ).proxy()

        self.editor = Editor.start(
            llm=llm2,
            prompt_manager=self.prompt_manager,
            creative_mode=self.creative_mode,
            identity_prompt_preamble="You are a skilled editor who helps writers refine their work."
        ).proxy()

        self.human = Human.start().proxy()

    def _do_develop_concept(self, concept_dir: Path,**kwargs):
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
        - Review with critic; repeat or continue
        - Write the first chapter
        - Review with editor; repeat or continue
        - Repeat
        - Review with critic
        - Publish
        """

        genre: str = self.human.prompt_user(self.prompt_manager.get_prompt([self.creative_mode.value, "HUMAN", "GENRE"])).get()
        starter: str = self.human.prompt_user(self.prompt_manager.get_prompt([self.creative_mode.value, "HUMAN", "STARTER"])).get()
        num_concepts: int = int(self.human.prompt_user(self.prompt_manager.get_prompt([self.creative_mode.value, "HUMAN", "NUM_IDEAS"])).get())

        futures: list = []
        for i in range(num_concepts):
            futures.append(self.author.ideate(genre, starter))
        ideas: list = [f.get() for f in futures]
        idx, selected_idea = self.human.prompt_user_select(ideas).get()

        context: StoryContext = StoryContext()
        context.concept = selected_idea
        context.plot = self.author.develop_plot(context).get()
        context.themes = self.author.develop_themes(context).get()
        context.characters = self.author.develop_characters(context).get()
        context.world = self.author.develop_world(context).get()
        context.storyline = self.author.develop_storyline(context).get()

        # output context
        with open(concept_dir / "concept.json", "w") as f:
            f.write(context.marshall())

        critique: str = self.critic.critique_concept(context).get()
        context.plot = self.author.develop_plot(context, critique=critique).get()
        context.characters = self.author.develop_characters(context, critique=critique).get()
        context.world = self.author.develop_world(context, critique=critique).get()
        context.storyline = self.author.develop_storyline(context, critique=critique).get()

        # output context
        with open(concept_dir / "context.json", "w") as f:
            f.write(context.marshall())

        summary: str = self.author.summarize_concept(context).get()

        # output summary
        with open(concept_dir / "summary.md", "w") as f:
            f.write(summary)

    def _write_chapter(self, context: StoryContext, pages_per_chapter: int, words_per_page: int, previous_chapter_summaries: list) -> str:
        book_summary = "".join([f"Chapter {idx+1}: {chapter}\n" for idx, chapter in enumerate(previous_chapter_summaries)])
        # first pass
        pages: list = []
        for page in range(1, pages_per_chapter + 1):
            chapter_so_far: str = "".join([f"{p}\n" for idx, p in enumerate(pages)])
            pages.append(self.author.write_section(context, words_per_page, page, pages_per_chapter, chapter_so_far, book_summary).get())
        chapter: str = " ".join(pages)
        return chapter

    def _do_draft_narrative(self, concept_dir: Path, **kwargs):
        logger.info(f"draft narrative for {concept_dir}")
        num_pages: int = 240
        num_chapters: int = 12
        words_per_page: int = 250
        pages_per_chapter: int = num_pages // num_chapters

        with open(concept_dir / "context.json", "r") as f:
            context = StoryContext.unmarshall(f.read())

        chapter_summaries: list = []
        chapters: list = []
        # for chapter in range(1, num_chapters + 1):
        for chapter in range(1, 1 + 1):
            content: str = self._write_chapter(context, pages_per_chapter, words_per_page, chapter_summaries)
            chapters.append(content)
            with open(concept_dir / f"chapter_{chapter}", "w") as f:
                f.write(content)


class HistoryPodcaster(Conductor):

    def _post_init(self, **kwargs):

        self.creative_mode = CreativeMode.PODCAST_MODE

        # create LLM
        logger.info(f"running creative mode: {self.creative_mode} and env: {kwargs.get('env', 'local')}")
        if kwargs.get('env', 'local') == 'local':
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
        elif kwargs.get('env') == 'bedrock':
            llm: ChatBedrock = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0")
            llm2: ChatBedrock = ChatBedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
        else:
            raise ValueError(f"invalid environment {kwargs.get('env')}")

        # start agents
        self.author = Author.start(
            llm=llm,
            prompt_manager=self.prompt_manager,
            creative_mode=self.creative_mode,
            identity_prompt_preamble="You are the assistant to a creative podcast producer."
        ).proxy()

        self.critic = Critic.start(
            llm=llm2,
            prompt_manager=self.prompt_manager,
            creative_mode=self.creative_mode,
            identity_prompt_preamble="You are a thoughtful and skilled critic on podcasts."
        ).proxy()

        self.editor = Editor.start(
            llm=llm2,
            prompt_manager=self.prompt_manager,
            creative_mode=self.creative_mode,
            identity_prompt_preamble="You are the assistant to a creative podcast producer."
        ).proxy()

        self.human = Human.start().proxy()

    def _do_develop_concept(self, concept_dir: Path, **kwargs):
        # genre: str = self.human.prompt_user(self.prompt_manager.get_prompt([self.creative_mode.value, "HUMAN", "GENRE"])).get()
        genre: str = "historical battles"
        starter: str = self.human.prompt_user(self.prompt_manager.get_prompt([self.creative_mode.value, "HUMAN", "STARTER"])).get()
        num_concepts: int = int(self.human.prompt_user(self.prompt_manager.get_prompt([self.creative_mode.value, "HUMAN", "NUM_IDEAS"])).get())

        futures: list = []
        for i in range(num_concepts):
            futures.append(self.author.ideate(genre, starter))
        ideas: list = [f.get() for f in futures]
        idx, selected_idea = self.human.prompt_user_select(ideas).get()

        context: StoryContext = StoryContext()
        context.concept = selected_idea
        context.plot = self.author.develop_plot(context).get()
        context.themes = self.author.develop_themes(context).get()
        context.characters = self.author.develop_characters(context).get()
        # context.world = self.author.develop_world(context).get()
        context.storyline = self.author.develop_storyline(context).get()

        # output context
        with open(concept_dir / "concept.json", "w") as f:
            f.write(context.marshall())

        critique: str = self.critic.critique_concept(context).get()
        context.plot = self.author.develop_plot(context, critique=critique).get()
        context.characters = self.author.develop_characters(context, critique=critique).get()
        # context.world = self.author.develop_world(context, critique=critique).get()
        context.storyline = self.author.develop_storyline(context, critique=critique).get()

        # output context
        with open(concept_dir / "context.json", "w") as f:
            f.write(context.marshall())

        summary: str = self.author.summarize_concept(context).get()

        # output summary
        with open(concept_dir / "summary.md", "w") as f:
            f.write(summary)


    def _write_segment(self, context: StoryContext, num_words: int, previous_segment_summaries: list) -> str:
        full_summary = "".join([f"Segment {idx+1}: {segment}\n" for idx, segment in enumerate(previous_segment_summaries)])
        # first pass
        content = self.author.write_section(context, num_words, 1, 1, "", full_summary).get()
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


if __name__ == '__main__':
    # conductor: Conductor = PaperbackWriter(working_dir="/Users/dan/dev/code/projects/python/scraibe/working", env="bedrock")
    # # concept_dir: Path = conductor.develop_concept()
    # conductor.draft_narrative(Path("/Users/dan/dev/code/projects/python/scraibe/working/concepts/20241026_200601"))

    conductor: Conductor = HistoryPodcaster(working_dir="/Users/dan/dev/code/projects/python/scraibe/working", env="local")
    # concept_dir: Path = conductor.develop_concept()
    conductor.draft_narrative(Path("/Users/dan/dev/code/projects/python/scraibe/working/concepts/20241028_122939"))