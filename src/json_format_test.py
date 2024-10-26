import json

from langchain.chat_models import ChatOllama
from langchain_aws import ChatBedrock
from tqdm import tqdm

print("starting...")
# llm: ChatOllama = ChatOllama(
#     model="llama3.2",
#     temperature=0.8,
#     num_predict=256,
# )
llm: ChatBedrock = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0")

prompt: str = """
You are a brilliant poet that can turn words into magic.

Write three sonnets and then output them as a JSON list. For example:

[
    "<sonnet 1>",
    "<sonnet 2>",
    "<sonnet 3>"
]

Only respond in JSON and nothing else.

ANSWER: 
"""

num_correct: int = 0
for i in range(10):
    print(f"generating #{i+1}")
    res = llm.invoke(prompt)
    try:
        json.loads(res.content)
        num_correct += 1
    except json.JSONDecodeError as e:
        print("JSON error")

print(f"Correct: {num_correct}")