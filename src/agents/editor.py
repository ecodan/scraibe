from typing import Dict, Any

from src.agents.actor import LLMActor


class Editor(LLMActor):

    def review_outline(self, context: Dict[str, Any]) -> str:
        pass

    def review_section(self, context: Dict[str, Any], section: str) -> str:
        pass
