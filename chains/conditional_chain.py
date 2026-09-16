from typing import Literal

from anthropic import BaseModel
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import Field

load_dotenv()


class FeedbackSentiment(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Return sentiment of the review either negative, positive, or neutral")


model = ChatOpenAI()
parser = StrOutputParser()
feedback_parser = PydanticOutputParser(pydantic_object=FeedbackSentiment)
prompt1 = PromptTemplate(
    template="Classify the sentiments in positive or negative for provided feedback \n{feedback} \n {format_instructions}",
    input_variables=['feedback'],
    partial_variables={
        'format_instructions': feedback_parser.get_format_instructions()
    }
)

classification_chain = prompt1 | model | feedback_parser


positive_feedback_prompt = PromptTemplate(
    template="Write an appropriate response from this positive feedback \n {feedback} ",
    input_variables=['feedback'],
    partial_variables={
        'format_instructions': feedback_parser.get_format_instructions()
    }
)


negative_feedback_prompt = PromptTemplate(
    template="Write an appropriate response from this negative feedback \n {feedback} ",
    input_variables=['feedback'],
    partial_variables={
        'format_instructions': feedback_parser.get_format_instructions()
    }
)

positive_chain = positive_feedback_prompt | model | parser
negative_chain = negative_feedback_prompt | model | parser
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', positive_chain),
    (lambda x: x.sentiment == 'negative', negative_chain),
    RunnableLambda(lambda x: 'No sentiment found')
)


chain = classification_chain | branch_chain

result = chain.invoke({
    'feedback': 'Terrible machine',
})
print(result)

chain.get_graph().print_ascii()