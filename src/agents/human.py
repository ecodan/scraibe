import sys
from typing import Dict, List

from pykka import ThreadingActor

from src.logutils import logio


class Human(ThreadingActor):

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
        return ideas[concept_num-1]
