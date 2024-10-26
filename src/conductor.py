from dataclasses import dataclass
from pathlib import Path

from langchain.chat_models import ChatOllama
from langchain_aws import ChatBedrock
from pykka import ActorProxy

from src.logutils import create_logger

from src.agents.author import Author
from src.agents.critic import Critic
from src.agents.editor import Editor
from src.agents.human import Human
from src.prompt_manager import PromptManager
from src.utils import StoryContext, utc_as_string

logger = create_logger(__name__)


@dataclass
class Conductor:
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
    author: ActorProxy[Author]
    editor: ActorProxy[Editor]
    critic: ActorProxy[Critic]
    human: ActorProxy[Human]

    def __init__(self, config_dir: str) -> None:
        super().__init__()
        config_dir_path: Path = Path(config_dir)
        assert config_dir_path.is_dir()

        # load prompts
        prompt_file_path: Path = config_dir_path / 'prompts.toml'
        assert prompt_file_path.is_file()
        prompt_manager: PromptManager = PromptManager(prompt_file_path)

        # create LLM
        # llm: ChatBedrock = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0")
        # llm2: ChatBedrock = ChatBedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
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

        # start agents
        self.author = Author.start(
            llm=llm,
            prompt_manager=prompt_manager,
            identity_prompt_preamble="You are a thoughtful and skilled fiction writer."
        ).proxy()
        self.critic = Critic.start(
            llm=llm2,
            prompt_manager=prompt_manager,
            identity_prompt_preamble = "You are a thoughtful and skilled literary critic who likes to help writers improve."
        ).proxy()
        self.editor = Editor.start(
            llm=llm2,
            prompt_manager=prompt_manager,
            identity_prompt_preamble="You are a skilled editor who helps writers refine their work."
        ).proxy()
        self.human = Human.start().proxy()


    def start(self):
        try:
            starter: dict = self.human.get_starter().get()
            # starter = {"genre": "romance", "num_concepts": 2, "idea": "two puppies in love"}
            ideas: list = self.author.ideate(starter['genre'], starter['num_concepts'], starter['idea']).get()
            selected_idea: str = self.human.select_idea(ideas).get()
            context: StoryContext = StoryContext()
            context.concept = selected_idea
            plot = self.author.develop_plot(context).get()
            context.plot = plot
            themes = self.author.develop_themes(context).get()
            context.themes = themes
            characters = self.author.develop_characters(context).get()
            context.characters = characters
            world = self.author.develop_world(context).get()
            context.world = world
            storyline = self.author.develop_storyline(context).get()
            context.storyline = storyline

            critique: str = self.critic.critique_concept(context).get()
            updated_plot: str = self.author.develop_plot(context, critique=critique).get()
            context.plot = updated_plot
            updated_characters = self.author.develop_characters(context, critique=critique).get()
            context.characters = updated_characters
            updated_world = self.author.develop_world(context, critique=critique).get()
            context.world = updated_world
            updated_storyline = self.author.develop_storyline(context, critique=critique).get()
            context.storyline = updated_storyline
            summary: str = self.author.summarize_concept(context).get()

            # output context
            fname: str = f"{utc_as_string()}-{starter['genre']}".replace(" ", "_")
            with open(f"/Users/dan/dev/code/projects/python/scraibe/working/concepts/{fname}.json", "w") as f:
                f.write(context.marshall())

            # output summary
            with open(f"/Users/dan/dev/code/projects/python/scraibe/working/concepts/{fname}-summary.md", "w") as f:
                f.write(summary)

        except Exception as e:
            logger.exception(e)
        finally:
            self.author.stop()
            self.critic.stop()
            self.editor.stop()
            self.human.stop()
            logger.info("done!")

    def stop(self):
        logger.info("stopping. shutting down agents...")
        self.author.stop()
        self.critic.stop()
        self.editor.stop()
        self.human.stop()


if __name__ == '__main__':
    conductor: Conductor = Conductor("/Users/dan/dev/code/projects/python/scraibe/config")
    conductor.start()
