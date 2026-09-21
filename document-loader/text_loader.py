from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


loader = TextLoader('solar-system.txt', encoding='utf-8')


load_dotenv()
docs = loader.load()


# Model
model = ChatOpenAI()

## Parsers
str_parser = StrOutputParser()

template=     PromptTemplate(
    template="Tell me about {topic} from the {doc}",
    input_variables=['topic', 'doc']
)


## Inference
chain = template | model | str_parser


result = chain.invoke({'topic': 'Sun', 'doc': docs[0].page_content})

print(result)