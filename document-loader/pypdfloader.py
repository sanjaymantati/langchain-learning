from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

loader = PyPDFLoader('directory/Indian_Constitution_Drafting_Committee_Report.pdf')

docs = loader.load()

# Model
model = ChatOpenAI()

## Parsers
str_parser = StrOutputParser()

template = PromptTemplate(
    template="Tell me about the query: {query} from the source PDF {doc}",
    input_variables=['query', 'doc']
)

## Inference
chain = template | model | str_parser

result = chain.invoke({'query': 'Who was the chairman of this commitee?', 'doc': docs})

print(result)
