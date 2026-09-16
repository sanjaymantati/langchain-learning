from google.genai._gaos.resources.interactions.googlemapsresultstep import result
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

model = ChatOpenAI()


class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all key themes in the review."]
    summary: Annotated[str, "A brief summary"]
    sentiment: Annotated[Literal[
        "positive", "negative", "neutral"], "Return sentiment of the review either negative, positive, or neutral"]
    good: Annotated[Optional[list[str]], "Return good review"]
    bad: Annotated[Optional[list[str]], "Return bad review"]


structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I just recently updated to the new Iphone 18 Pro. and the experience is amazing. The smooth and sesible UI with the apple ecosystem built for privacy and the performance.
Althoug i still miss the tweaking capabilities that i was getting from my old Galaxy s24. Apps that gives free youtube and can install app outside from the app store.
""")

print(result)
