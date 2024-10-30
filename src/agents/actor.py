from abc import ABCMeta
from enum import Enum

from langchain_core.language_models import BaseChatModel

from src.prompt_manager import PromptManager


class CreativeMode(Enum):
    AUTHOR_MODE = "AUTHOR"
    PODCAST_MODE = "PODCAST"


class Actor(metaclass=ABCMeta):

    def __init__(self, prompt_manager: PromptManager, creative_mode: CreativeMode):
        super().__init__()
        self.prompt_manager: PromptManager = prompt_manager
        self.creative_mode: str = creative_mode.value

    def stop(self):
        """
        Placeholder for subclasses that implement threading
        """
        pass

class LLMActor(Actor):

    def __init__(self, llm: BaseChatModel, prompt_manager: PromptManager, creative_mode: CreativeMode, identity_prompt_preamble: str = "You are a helpful bot."):
        super().__init__(prompt_manager, creative_mode)
        self.llm: BaseChatModel = llm
        self.identity_prompt_preamble: str = identity_prompt_preamble

