import sys
from typing import Dict, List

from src.agents.actor import Actor
from src.logutils import logio


class Human(Actor):

    @logio()
    def prompt_user(self, prompt: str) -> str:
        return input(prompt)

    @logio()
    def prompt_user_select(self, options: List[str]) -> (int, str):
        prompt: str = "".join([f"{idx + 1}: {option}\n" for idx, option in enumerate(options)])
        prompt = "Choose one of the following options:\n" + prompt
        selection: int = int(self.prompt_user(prompt))
        return selection, options[selection - 1]

    @logio()
    def get_starter(self) -> Dict[str, str | int]:
        genre: str = input("which genre? > ")
        idea: str = input("what is your idea? > ")
        num_concepts: int = int(input("how many concepts should I generate? > "))
        return {"genre": genre, "idea": idea, "num_concepts": num_concepts}

    @logio()
    def select_idea(self, ideas: List[str]) -> str:
        sys.stdout.write("select from one of the following:\n")
        for idx, idea in enumerate(ideas):
            sys.stdout.write(f"{idx + 1}: {idea}\n")
        concept_num: int = int(input("which concept should I build on? > "))
        return ideas[concept_num - 1]
