from langchain_core.language_models import BaseChatModel
from pykka import ThreadingActor

from src.prompt_manager import PromptManager


class LLMActor(ThreadingActor):

    def __init__(self, llm: BaseChatModel, prompt_manager: PromptManager, identity_prompt_preamble: str = "You are a helpful bot."):
        super().__init__()
        self.llm: BaseChatModel = llm
        self.prompt_manager: PromptManager = prompt_manager
        self.identity_prompt_preamble: str = identity_prompt_preamble
