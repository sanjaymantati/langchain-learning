
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI


load_dotenv()
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.3)


template = PromptTemplate(
    template="""
    Suggest a good title about {topic}
    """,
    input_variables=['topic']

)

topic = input('Topic: ')

formatted_prompt = template.format(topic=topic)
result = llm.invoke(formatted_prompt)
print(result)