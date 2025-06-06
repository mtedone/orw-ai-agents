import os

from langchain_google_genai import ChatGoogleGenerativeAI


def retrieve_gemini_llm():
    model = os.getenv("GEMINI_MODEL")
    self_llm = ChatGoogleGenerativeAI(model=model)
    return self_llm
