import tomllib
from typing import Union, Dict, Any


class PromptManager:
    def __init__(self, toml_file_path: str):
        self.prompts = self._load_toml(toml_file_path)

    def _load_toml(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, 'rb') as file:
            return tomllib.load(file)

    def get_prompt(self, prompt_path: Union[str, list], variant: str = "DEFAULT") -> str:
        if isinstance(prompt_path, str):
            prompt_path = prompt_path.split('.')

        current = self.prompts
        for key in prompt_path:
            if key in current:
                current = current[key]
            else:
                raise KeyError(f"No prompt found for path: {'.'.join(prompt_path)}")

        if isinstance(current, dict):
            if variant in current:
                return current[variant]
            elif variant is None:
                return current['DEFAULT']
            else:
                raise KeyError(
                    f"No prompt found for prompt {'.'.join(prompt_path)} variant: '{variant}' and no default value available")
        else:
            raise KeyError(f"Prompt path must end with a Dict node: {'.'.join(prompt_path)}")
