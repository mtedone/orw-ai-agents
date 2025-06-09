from langchain_tavily import tavily_search, TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from datetime import datetime
from IPython.display import display, Markdown, Image
from VertexAIFactory import create_gemini_llm
from dotenv import load_dotenv
import os

_ = load_dotenv()

tavily_key = os.getenv("TAVILY_API_KEY")

llm = create_gemini_llm()

search_tool = TavilySearch(max_results=5, api_key=tavily_key)

@tool
def get_current_date():
    """Returns the current date and time. Use this tool first for any time-based questions."""
    return f"The current date is: {datetime.now().strftime("%d -%B-%Y")}"

tools = [get_current_date, search_tool]

graph = create_react_agent(llm, tools)

# display(Image(graph.get_graph().draw_mermaid_png()))

def render_markdown(text):
    display(Markdown(text))

def process_stream(stream):
    message = ""
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

    return message

def process_query(query):
    inputs = {"messages": [("user", query)]}
    message = process_stream(graph.stream(inputs, stream_mode="values"))
    render_markdown(f"##Answer: \n {message}")

process_query("Your query here:")








