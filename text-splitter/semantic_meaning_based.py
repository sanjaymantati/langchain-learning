from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
load_dotenv()
sample_text = """
The Quiet Power of Sleep
Sleep is the most underrated performance tool we have. 
During deep sleep, the brain clears metabolic waste, consolidates memories, and resets emotional regulation. 
A single night of poor sleep can cut attention and decision-making quality as much as mild intoxication. 
Yet many people trade sleep for extra work hours, not realizing they're borrowing time at a steep interest rate. 
Seven to nine hours isn't laziness. 
It's maintenance for the machine that does all your thinking.

Why Cities Are Getting Hotter
Urban areas are often several degrees warmer than surrounding countryside, a phenomenon called the urban heat island effect. 
Concrete and asphalt absorb sunlight during the day and release it slowly at night, while the lack of trees removes natural cooling through shade and evaporation. 
Air conditioners make it worse by pumping heat outdoors. 
Cities like Hyderabad, Delhi, and Phoenix are fighting back with reflective roofs, green corridors, and urban forests, proof that smarter design can cool a city without waiting for the climate to change.
"""


text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

document = text_splitter.create_documents([sample_text])
print(len(document))
