import os
import vertexai

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

_ = load_dotenv()

def initialise_vertex_ai():



    google_project = os.getenv("GOOGLE_PROJECT")
    project_region = os.getenv("PROJECT_REGION")

    vertexai.init(project=google_project, location=project_region)
    return vertexai

def create_gemini_llm():
    model = os.getenv("GEMINI_MODEL")
    self_llm = ChatGoogleGenerativeAI(model=model)
    return self_llm

