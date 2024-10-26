import json
from typing import List, Dict, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import JsonOutputParser, CommaSeparatedListOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.agents.actor import LLMActor
from src.logutils import logio
from src.prompt_manager import PromptManager
from src.utils import StoryContext


class Author(LLMActor):
    class JsonListOutputParser(JsonOutputParser):
        def parse(self, text: str) -> List[str]:
            json_object = json.loads(text)
            if isinstance(json_object, list):
                return json_object
            else:
                raise ValueError("Parsed JSON is not a list")

        def get_format_instructions(self) -> str:
            return """Your response should be a JSON list of strings. For example:
            
            ["item 1", "item 2", "item 3"]
         
            Wrap each idea in double quotes. Separated each idea with a comma.
            Put a single open square bracket at the start and a single close square bracket at the end. 
            Respond only with valid JSON and no extra characters.
            """

    @logio()
    def ideate(self, genre: str, num_concepts: int, starter_idea: str) -> List[str]:
        output_parser = Author.JsonListOutputParser()
        tplt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
            [
                ("system",
                 self.identity_prompt_preamble + "\n" +
                 self.prompt_manager.get_prompt("AUTHOR.IDEATE")
                 ),
            ]
        )
        prompt: str = tplt.format(
            genre=genre,
            num_plots=num_concepts,
            starter=starter_idea,
            format_instructions=output_parser.get_format_instructions()
        )
        res: BaseMessage = self.llm.invoke(prompt)
        return output_parser.parse(res.content)

    @logio(truncate_at=-1)
    def develop_plot(self, context: StoryContext, critique: str = None) -> str:
        if critique is None:
            tplt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        self.identity_prompt_preamble + "\n" +
                        self.prompt_manager.get_prompt("AUTHOR.DEVELOP_PLOT.BASE") + "\n" +
                        self.prompt_manager.get_prompt("AUTHOR.DEVELOP_PLOT.UNASSISTED")
                     ),
                ]
            )
        else:
            tplt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        self.identity_prompt_preamble + "\n" +
                        self.prompt_manager.get_prompt("AUTHOR.DEVELOP_PLOT.BASE") + "\n" +
                        self.prompt_manager.get_prompt("AUTHOR.DEVELOP_PLOT.WITH_FEEDBACK")
                    )
                ]
            )
        prompt: str = tplt.format(
            concept=context.concept,
            plot=context.plot,
            feedback=critique
        )
        res: BaseMessage = self.llm.invoke(prompt)
        return res.content

    @logio(truncate_at=-1)
    def develop_themes(self, context: StoryContext) -> str:
        tplt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
            [
                ("system",
                 self.identity_prompt_preamble + "\n" +
                 self.prompt_manager.get_prompt("AUTHOR.DEVELOP_THEME.UNASSISTED")
                 ),
            ]
        )
        prompt: str = tplt.format(
            concept=context.concept,
            plot=context.plot
        )
        res: BaseMessage = self.llm.invoke(prompt)
        return res.content

    @logio(truncate_at=-1)
    def develop_characters(self, context: StoryContext, critique: str = None) -> str:
        if critique is None:
            tplt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
                [
                    ("system",
                     self.identity_prompt_preamble + "\n" +
                     self.prompt_manager.get_prompt("AUTHOR.DEVELOP_CHARACTERS.BASE") + "\n" +
                     self.prompt_manager.get_prompt("AUTHOR.DEVELOP_CHARACTERS.UNASSISTED")
                     ),
                ]
            )
        else:
            tplt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
                [
                    ("system",
                     self.identity_prompt_preamble + "\n" +
                     self.prompt_manager.get_prompt("AUTHOR.DEVELOP_CHARACTERS.BASE") + "\n" +
                     self.prompt_manager.get_prompt("AUTHOR.DEVELOP_CHARACTERS.WITH_FEEDBACK")
                     ),
                ]
            )

        prompt: str = tplt.format(
            concept=context.concept,
            plot=context.plot,
            themes=context.themes,
            characters=context.characters,
            feedback=critique
        )
        res: BaseMessage = self.llm.invoke(prompt)
        return res.content

    @logio(truncate_at=-1)
    def develop_world(self, context: StoryContext, critique: str = None) -> str:
        if critique is None:
            tplt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
                [
                    ("system",
                     self.identity_prompt_preamble + "\n" +
                     self.prompt_manager.get_prompt("AUTHOR.DEVELOP_WORLD.BASE") + "\n" +
                     self.prompt_manager.get_prompt("AUTHOR.DEVELOP_WORLD.UNASSISTED")
                     ),
                ]
            )
        else:
            tplt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
                [
                    ("system",
                     self.identity_prompt_preamble + "\n" +
                     self.prompt_manager.get_prompt("AUTHOR.DEVELOP_WORLD.BASE") + "\n" +
                     self.prompt_manager.get_prompt("AUTHOR.DEVELOP_WORLD.WITH_FEEDBACK")
                     ),
                ]
            )
        prompt: str = tplt.format(
            concept=context.concept,
            plot=context.plot,
            world=context.world,
            feedback=critique
        )
        res: BaseMessage = self.llm.invoke(prompt)
        return res.content

    @logio()
    def develop_storyline(self, context: StoryContext, critique: str = None) -> str:
        if critique is None:
            tplt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
                [
                    ("system",
                     self.identity_prompt_preamble + "\n" +
                     self.prompt_manager.get_prompt("AUTHOR.DEVELOP_STORYLINE.BASE") + "\n" +
                     self.prompt_manager.get_prompt("AUTHOR.DEVELOP_STORYLINE.UNASSISTED")
                     ),
                ]
            )
        else:
            tplt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
                [
                    ("system",
                     self.identity_prompt_preamble + "\n" +
                     self.prompt_manager.get_prompt("AUTHOR.DEVELOP_STORYLINE.BASE") + "\n" +
                     self.prompt_manager.get_prompt("AUTHOR.DEVELOP_STORYLINE.WITH_FEEDBACK")
                     ),
                ]
            )
        prompt: str = tplt.format(
            concept=context.concept,
            plot=context.plot,
            themes=context.themes,
            characters=context.characters,
            storyline=context.storyline,
            world=context.world,
            feedback=critique
        )
        res: BaseMessage = self.llm.invoke(prompt)
        return res.content

    @logio()
    def summarize_concept(self, context: StoryContext) -> str:
        tplt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
            [
                ("system",
                 self.identity_prompt_preamble + "\n" +
                 self.prompt_manager.get_prompt("AUTHOR.SUMMARIZE_CONCEPT")
                 ),
            ]
        )
        prompt: str = tplt.format(
            concept=context.concept,
            plot=context.plot,
            themes=context.themes,
            characters=context.characters,
            storyline=context.storyline,
            world=context.world
        )
        res: BaseMessage = self.llm.invoke(prompt)
        return res.content

    def write_section(self, context: Dict[str, Any]) -> str:
        pass

