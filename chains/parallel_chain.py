from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain.schema.runnable import RUnnableParallel
load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a short and simple note from the following text \n{text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text \n {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="Merge provided notes and q&a into single documents. \n notes -> {notes}, q&a -> {quiz}",
    input_variables=['notes', 'quiz']
)

model = ChatOpenAI()

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

parallel_chain = RunnableParallel({
    'notes': prompt1 | model | parser,
    'quiz': prompt2 | model | parser,
})

merge_chain = prompt3 | model | parser


chain = parallel_chain | merge_chain

text = """
A field-programmable gate array (FPGA) is a type of configurable integrated circuit that can be repeatedly programmed after manufacturing. FPGAs are a subset of logic devices referred to as programmable logic devices (PLDs). They consist of a grid-connected array of programmable logic blocks that can be configured "in the field" to interconnect with other logic blocks to perform various digital functions. FPGAs are often used in limited (low) quantity production of custom-made products, and in research and development, where the higher cost of individual FPGAs is not as important and where creating and manufacturing a custom circuit would not be feasible. Other applications for FPGAs include the telecommunications, automotive, aerospace, and industrial sectors, which benefit from their flexibility, high signal processing speed, and parallel processing abilities.

An FPGA configuration is generally written using a hardware description language (HDL), e.g., VHDL, similar to the ones used for application-specific integrated circuits (ASICs). Circuit diagrams were formerly used to write the configuration.

The logic blocks of an FPGA can be configured to perform complex combinational functions, or act as simple logic gates like AND and XOR. In most FPGAs, logic blocks also include memory elements, which may be simple flip-flops or more sophisticated blocks of memory.[1] Many FPGAs can be reprogrammed to implement different logic functions, allowing flexible reconfigurable computing as performed in computer software.

FPGAs also have a role in embedded system development due to their capability to start system software development simultaneously with hardware, enable system performance simulations at a very early phase of the development, and allow various system trials and design iterations before finalizing the system architecture.[2]

FPGAs are also commonly used during the development of ASICs to speed up the simulation process.
"""

result = chain.invoke({'text': text})
print(result)
chain.get_graph().print_ascii()
