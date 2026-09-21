    from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a poetry in 4-5 lines on {topic}",
    input_variables=['topic'])

prompt2 = PromptTemplate(
    template="Write a meaning of this poetry {topic}",
    input_variables=['topic'])
str_out_parser = StrOutputParser()

model = ChatOpenAI()

poem_chain = RunnableSequence(prompt1, model, str_out_parser)
parallel_chain = RunnableParallel({
    'poem': RunnablePassthrough(),
    'meaning': RunnableSequence(prompt2, model, str_out_parser),
})

final_chain = RunnableSequence(poem_chain, parallel_chain)
result = final_chain.invoke({'topic': 'mountains'})
print(result)
final_chain.get_graph().print_ascii()
