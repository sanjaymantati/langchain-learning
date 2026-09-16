from typing import Optional, Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI()


class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all key themes in the review.")
    summary: str = Field(description="A brief summary")
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Return sentiment of the review either negative, positive, or neutral")
    good: Optional[list[str]] = Field(default=None, description="Return good review")
    bad: Optional[list[str]] = Field(default=None, description="Return bad review")


structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I just recently updated to the new Iphone 18 Pro. and the experience is amazing. The smooth and sesible UI with the apple ecosystem built for privacy and the performance.
Althoug i still miss the tweaking capabilities that i was getting from my old Galaxy s24. Apps that gives free youtube and can install app outside from the app store.
""")

print(result)
