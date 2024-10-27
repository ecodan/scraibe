from enum import Enum

from langchain_core.language_models import BaseChatModel
from pykka import ThreadingActor

from src.prompt_manager import PromptManager


class CreativeMode(Enum):
    AUTHOR_MODE = "AUTHOR"
    PODCAST_MODE = "PODCAST"


class LLMActor(ThreadingActor):

    def __init__(self, llm: BaseChatModel, prompt_manager: PromptManager, creative_mode: CreativeMode, identity_prompt_preamble: str = "You are a helpful bot."):
        super().__init__()
        self.llm: BaseChatModel = llm
        self.prompt_manager: PromptManager = prompt_manager
        self.creative_mode: str = creative_mode.value
        self.identity_prompt_preamble: str = identity_prompt_preamble
