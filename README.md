# ScrAIbe
This is a exploratory project to see how well an agentic network can ideate and generate fiction and non-fiction stories. Essentially it leverages an Author, Editor and Critic agent to develo and refine a concept and then write it incrementally.


## Setup
- clone github repo
- create a venv / conda env using requirements.txt
- run CLI (e.g.)

`python scraibe.py longform-fiction /path/to/working/dir -e bedrock`

## Overview
The app consists of two primary constructs: Composers and Actors
- **Composers** create actors and orchestrate the interactions between them. For example, a book writing composer will Author, Editor, Critic and Human actors and then use them to create/refine a concept and then draft a narrative.
- **Actors**, which can be Author, Editor, Critic, Human, or something else. Actors have actions they can take (typically invoking an LLM or prompting a user) and typically return strings.

## To experiment
- Subclass Conductor and then orchestrate some operation between the actors in the _do_develop_concept() and _do_draft_narrative() methods.
- Add prompts for your specific Actor operations. For example, if you wanted a Newsletter composer:
  - Add a [NEWSLETTER] top level section to prompts.toml
  - Add specific prompts. 
    - If you're using an existing actor (e.g. Author), copy the prompts from another section (e.g. AUTHOR) and override the prompts.
  - Add your CreativeMode to Actor
  - Add the new CreativeMode to your Conductor subclass _post_init

Have fun. All artifacts will be written to a dated project directory under the working dir.