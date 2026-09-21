from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import  ChatOpenAI

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a poetry in 4-5 lines on {topic}",
    input_variables=['topic'] )

prompt2 = PromptTemplate(
    template="Explain this poem {poem}",
    input_variables=['poem'] )

str_out_parser = StrOutputParser()

model = ChatOpenAI()

chain = RunnableSequence(prompt1, model, str_out_parser, prompt2, model ,str_out_parser)

result = chain.invoke({'topic': 'mountains'})
print(result)
chain.get_graph().print_ascii()
