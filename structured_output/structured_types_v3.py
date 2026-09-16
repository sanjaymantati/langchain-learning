from typing import Optional, Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import json
load_dotenv()

model = ChatOpenAI()

with open('json_schema.json') as f:
    json_schema=json.loads(f.read())


structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("""I just recently updated to the new Iphone 18 Pro. and the experience is amazing. The smooth and sesible UI with the apple ecosystem built for privacy and the performance.
Althoug i still miss the tweaking capabilities that i was getting from my old Galaxy s24. Apps that gives free youtube and can install app outside from the app store.
""")

print(result)
