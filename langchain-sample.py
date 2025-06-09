from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from VertexAIFactory import create_gemini_llm
from IPython.display import Markdown, display

llm = create_gemini_llm()

system_template = """
You are a trip planner expert. Help me plan a trip to {destination}.
Consider my preferences for {preferences}.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "What should I do in {destination}?")
])

parser = StrOutputParser()

trip_planner_chain = prompt_template | llm | parser

def plan_trip(destination, preferences):
    response = trip_planner_chain.invoke({"destination": destination, "preferences": preferences})
    return response

def render_markdown(text):
    display(Markdown(text))

result = plan_trip("Paris", "museums, cafes, historical sites")
render_markdown(result)


