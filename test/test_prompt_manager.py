import unittest
import tempfile
import os
from src.prompt_manager import PromptManager


class TestPromptManager(unittest.TestCase):
    def setUp(self):
        self.test_toml_content = """
        [category1]
        subcat1.prompt1.PODCAST = "Podcast prompt 1"
        subcat1.prompt1.DEFAULT = "Default prompt for subcategory "

        [category2]
        subcat2.prompt2.DEFAULT = "Specific prompt 2"

        [category3]
        subcat3.prompt3.VIDEO = "prompt for category3"
        """

        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.toml')
        self.temp_file.write(self.test_toml_content)
        self.temp_file.close()

        self.manager = PromptManager(self.temp_file.name)

    def tearDown(self):
        os.unlink(self.temp_file.name)

    def test_get_specific_prompt(self):
        self.assertEqual(self.manager.get_prompt("category1.subcat1.prompt1"), "Default prompt for subcategory ")
        self.assertEqual(self.manager.get_prompt("category1.subcat1.prompt1", variant='PODCAST'), "Podcast prompt 1")
        self.assertEqual(self.manager.get_prompt("category1.subcat1.prompt1", variant=None), "Default prompt for subcategory ")
        self.assertEqual(self.manager.get_prompt(["category2", "subcat2", "prompt2"]), "Specific prompt 2")
        with self.assertRaises(KeyError):
            self.assertEqual(self.manager.get_prompt(["category2", "subcat2", "prompt2"], variant="blah"), "Specific prompt 2")

    def test_prompt_not_found(self):
        with self.assertRaises(KeyError):
            self.manager.get_prompt("nonexistent.category")
        with self.assertRaises(KeyError):
            self.manager.get_prompt("category3.subcat3.prompt3")
        with self.assertRaises(KeyError):
            self.manager.get_prompt("subcat3.prompt3")

    def test_string_and_list_input(self):
        string_result = self.manager.get_prompt("category2.subcat2.prompt2")
        list_result = self.manager.get_prompt(["category2", "subcat2", "prompt2"])
        self.assertEqual(string_result, list_result)

    def test_empty_path(self):
        with self.assertRaises(KeyError):
            self.manager.get_prompt("")
        with self.assertRaises(KeyError):
            self.manager.get_prompt([])

