from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_openai import  ChatOpenAI

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a poetry in 4-5 lines on {topic}",
    input_variables=['topic'] )

prompt2 = PromptTemplate(
    template="Write a joke on {topic}",
    input_variables=['topic'] )
str_out_parser = StrOutputParser()

model = ChatOpenAI()

chain = RunnableParallel({
    'poem': RunnableSequence(prompt1, model, str_out_parser),
    'joke': RunnableSequence(prompt2, model, str_out_parser),
})

result = chain.invoke({'topic': 'mountains'})
print(result)
chain.get_graph().print_ascii()
