import vertexai
import dotenv
import os

from dotenv import load_dotenv


def initialise_vertex_ai():

    _ = load_dotenv()

    google_project = os.getenv("GOOGLE_PROJECT")
    project_region = os.getenv("PROJECT_REGION")

    vertexai.init(project=google_project, location=project_region)
    return vertexai

