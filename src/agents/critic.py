from typing import Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from src.agents.actor import LLMActor
from src.logutils import logio
from src.utils import StoryContext


class Critic(LLMActor):
    @logio(truncate_at=-1)
    def critique_concept(self, context: StoryContext) -> str:
        tplt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
            [
                ("system", "You're a gifted editor and literary critic.\n"
                           "A client of yours is developing a concept for a story based on the following idea:\n"
                           "IDEA: {concept}\n"
                           "Here's the plot that builds on the idea:\n"
                           "PLOT: {plot}\n"
                           "The story will also examining the following literary themes:\n"
                           "THEMES: {themes}\n"
                           "Here are the characters used in the story and their definitions:\n"
                           "CHARACTERS: {characters}\n"
                           "Here's a description of the world the characters inhabit:\n"
                           "WORLD: {world}\n"
                           "Step back and think about this concept. Is the plot compelling? Does it have novel elements that will hold the reader's interest? Are the characters deep and interesting? Does the world suspend disbelief? \n"
                           "Provide suggestions on how your client can improve concept. If you think it is perfect as is, respond with 'NO CHANGES NEEDED'.\n"
                           "ANSWER: "),
            ]
        )
        prompt: str = tplt.format(
            concept=context.concept,
            plot=context.plot,
            themes=context.themes,
            characters=context.characters,
            world=context.world
        )
        res: BaseMessage = self.llm.invoke(prompt)
        return res.content

    def critique_characters(self, concept: str, characters: Dict[str, Any]) -> str:
        pass

    def critique_storyline(self, concept: str, characters: Dict[str, Any], storyline: str) -> str:
        pass

    def critique_world(self, concept: str, plot: str, storyline: str, world: str) -> str:
        pass

    def critique_themes(self, concept: str, plot: str, ) -> str:
        pass

    def critique_writing(self, concept: str, text: str) -> str:
        pass
