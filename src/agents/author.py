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
    def ideate(self, genre: str, starter_idea: str) -> str:
        output_parser = Author.JsonListOutputParser()
        tplt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
            [
                ("system",
                 self.identity_prompt_preamble + "\n" +
                 self.prompt_manager.get_prompt([self.creative_mode, "IDEATE"])
                 ),
            ]
        )
        prompt: str = tplt.format(
            genre=genre,
            starter=starter_idea,
            format_instructions=output_parser.get_format_instructions()
        )
        res: BaseMessage = self.llm.invoke(prompt)
        return res.content

    @logio(truncate_at=-1)
    def develop_plot(self, context: StoryContext, critique: str = None) -> str:
        if critique is None:
            tplt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        self.identity_prompt_preamble + "\n" +
                        self.prompt_manager.get_prompt([self.creative_mode, "DEVELOP_PLOT", "BASE"]) + "\n" +
                        self.prompt_manager.get_prompt([self.creative_mode, "DEVELOP_PLOT", "UNASSISTED"])
                     ),
                ]
            )
        else:
            tplt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        self.identity_prompt_preamble + "\n" +
                        self.prompt_manager.get_prompt([self.creative_mode, "DEVELOP_PLOT", "BASE"]) + "\n" +
                        self.prompt_manager.get_prompt([self.creative_mode, "DEVELOP_PLOT", "WITH_FEEDBACK"])
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
                 self.prompt_manager.get_prompt([self.creative_mode, "DEVELOP_THEME", "UNASSISTED"])
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
                     self.prompt_manager.get_prompt([self.creative_mode, "DEVELOP_CHARACTERS", "BASE"]) + "\n" +
                     self.prompt_manager.get_prompt([self.creative_mode, "DEVELOP_CHARACTERS", "UNASSISTED"])
                     ),
                ]
            )
        else:
            tplt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
                [
                    ("system",
                     self.identity_prompt_preamble + "\n" +
                     self.prompt_manager.get_prompt([self.creative_mode, "DEVELOP_CHARACTERS", "BASE"]) + "\n" +
                     self.prompt_manager.get_prompt([self.creative_mode, "DEVELOP_CHARACTERS", "WITH_FEEDBACK"])
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
                     self.prompt_manager.get_prompt([self.creative_mode, "DEVELOP_WORLD", "BASE"]) + "\n" +
                     self.prompt_manager.get_prompt([self.creative_mode, "DEVELOP_WORLD", "UNASSISTED"])
                     ),
                ]
            )
        else:
            tplt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
                [
                    ("system",
                     self.identity_prompt_preamble + "\n" +
                     self.prompt_manager.get_prompt([self.creative_mode, "DEVELOP_WORLD", "BASE"]) + "\n" +
                     self.prompt_manager.get_prompt([self.creative_mode, "DEVELOP_WORLD", "WITH_FEEDBACK"])
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
                     self.prompt_manager.get_prompt([self.creative_mode, "DEVELOP_STORYLINE", "BASE"]) + "\n" +
                     self.prompt_manager.get_prompt([self.creative_mode, "DEVELOP_STORYLINE", "UNASSISTED"])
                     ),
                ]
            )
        else:
            tplt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
                [
                    ("system",
                     self.identity_prompt_preamble + "\n" +
                     self.prompt_manager.get_prompt([self.creative_mode, "DEVELOP_STORYLINE", "BASE"]) + "\n" +
                     self.prompt_manager.get_prompt([self.creative_mode, "DEVELOP_STORYLINE", "WITH_FEEDBACK"])
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
                 self.prompt_manager.get_prompt([self.creative_mode, "SUMMARIZE_CONCEPT"])
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

    @logio()
    def write_section(self, context: StoryContext, num_words, section_number, total_sections, preceding_sections, extended_context) -> str:
        tplt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
            [
                ("system",
                 self.identity_prompt_preamble + "\n" +
                 """
                 You're helping the author write a story based on the following context:\n
                    IDEA: {concept}\n
                    PLOT: {plot}\n
                    THEMES: {themes}\n
                    CHARACTERS: {characters}\n
                    WORLD: {world}\n
                    STORYLINE: {storyline}\n
                
                The story so far can be summarized as:\n
                ========\n
                {extended_context}\n
                ========\n
                
                The current section so far is:\n
                ========\n
                {preceding_sections}\n
                ========\n

                 Provide suggested content for the next tranche of the current section using approximately {num_words} 
                 words. This will be number {section_number} of {total_sections} total sections in this chapter. 
                 If this is one of the last tranches, prepare to wrap up the chapter. If this is the last tranches, 
                 then end the section cleanly.\n
                 
                 Don't provide a preamble; only respond with the content.\n
                 
                 ANSWER: Here is a suggestion for the next tranche of the current section:\n\n
                 """
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
            extended_context=extended_context,
            preceding_sections=preceding_sections,
            num_words=num_words,
            section_number=section_number,
            total_sections=total_sections
        )
        res: BaseMessage = self.llm.invoke(prompt)
        return res.content

