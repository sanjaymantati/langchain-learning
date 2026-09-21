from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
loader = CSVLoader(file_path="sample_sales.csv")
docs = loader.load()
parser = StrOutputParser()
model = ChatOpenAI()
template = PromptTemplate(
    template="Write answer of the user query: {query} from the csv doc: {doc}",
    input_variables=['query', 'doc']
)


chain = template | model |  parser


result = chain.invoke({
    'query': "What's total revenue by region?",
    'doc': docs
})

print(result)