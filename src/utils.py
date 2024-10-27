import json
from dataclasses import dataclass
from datetime import datetime


@dataclass
class StoryContext:
    concept: str = None
    plot: str = None
    characters: str = None
    storyline: str = None
    world: str = None
    themes: str = None

    def __str__(self):
        return (
            f"StoryContext:\n"
            f"\tconcept: {self.concept}\n"
            f"\tplot: {self.plot}\n"
            f"\tthemes: {self.themes}\n"
            f"\tcharacters: {self.characters}\n"
            f"\tworld: {self.world}\n"
            f"\tstoryline: {self.storyline}"
        )

    def marshall(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__)

    @classmethod
    def unmarshall(cls, json_str: str) -> "StoryContext":
        context: dict = json.loads(json_str)
        return StoryContext(
            concept=context["concept"],
            plot=context["plot"],
            characters=context["characters"],
            storyline=context["storyline"],
            world=context["world"],
            themes=context["themes"]
        )


def utc_as_string(dt: datetime = datetime.now(), compact: bool = True, sub_second: bool = False) -> str:
    if compact:
        if sub_second:
            return dt.strftime('%Y%m%d_%H%M%S.%f')
        else:
            return dt.strftime('%Y%m%d_%H%M%S')
    else:
        if sub_second:
            return dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        else:
            return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
