from google.cloud import storage
from google.auth import default
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool
import vertexai

# Authenticate with default credentials
credentials, project = default()


# Create a RAG Corpus, Import Files, and Generate a response

# TODO(developer): Update and un-comment below lines
PROJECT_ID = "techwings-production"
display_name = "vito-vanilla-corpus"
# paths = ["https://drive.google.com/file/d/123", "gs://my_bucket/my_files_dir"]  # Supports Google Cloud Storage and Google Drive Links
paths = [
    "gs://tw-vito/training/vanilla/books/",
    "gs://tw-vito/training/vanilla/theories/"
]


# Initialize Vertex AI API once per session
vertexai.init(project=PROJECT_ID, location="europe-west4", credentials=credentials)

# Create RagCorpus
# Configure embedding model, for example "text-embedding-005".
embedding_model_config = rag.RagEmbeddingModelConfig(
    vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
        publisher_model="publishers/google/models/text-embedding-005"
    )
)

rag_corpus = rag.create_corpus(
    display_name=display_name,
    backend_config=rag.RagVectorDbConfig(
        rag_embedding_model_config=embedding_model_config
    ),
)

# Import Files to the RagCorpus
rag.import_files(
    rag_corpus.name,
    paths,
    # Optional
    transformation_config=rag.TransformationConfig(
        chunking_config=rag.ChunkingConfig(
            chunk_size=512,
            chunk_overlap=100,
        ),
    ),
    max_embedding_requests_per_min=1000,  # Optional
)

# Direct context retrieval
rag_retrieval_config = rag.RagRetrievalConfig(
    top_k=3,  # Optional
    filter=rag.Filter(vector_distance_threshold=0.5),  # Optional
)
# response = rag.retrieval_query(
#     rag_resources=[
#         rag.RagResource(
#             rag_corpus=rag_corpus.name,
#             # Optional: supply IDs from `rag.list_files()`.
#             # rag_file_ids=["rag-file-1", "rag-file-2", ...],
#         )
#     ],
#     text="What can I do to feel better after a breakup?",
#     rag_retrieval_config=rag_retrieval_config,
# )
# print(response)

# Enhance generation
# Create a RAG retrieval tool
rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[
                rag.RagResource(
                    rag_corpus=rag_corpus.name,  # Currently only 1 corpus is allowed.
                    # Optional: supply IDs from `rag.list_files()`.
                    # rag_file_ids=["rag-file-1", "rag-file-2", ...],
                )
            ],
            rag_retrieval_config=rag_retrieval_config,
        ),
    )
)

# Create a Gemini model instance
rag_model = GenerativeModel(
    model_name="gemini-2.0-flash-001", tools=[rag_retrieval_tool]
)
user_prompt = "What can I do to feel better after a breakup?"
# Generate response
prompt = f"""You are a world renowned therapist who is well versed in psychotherapy and epigenetics.
                    Your answers should be easy to understand and explained in layman terms. "
                    In your answers include in-text references and a list of references at the end"
                    in APA 7th Format. User asks {user_prompt} and you answer."""



response = rag_model.generate_content(prompt)
print(response.text)
# Example response:
#   RAG stands for Retrieval-Augmented Generation.
#   It's a technique used in AI to enhance the quality of responses
# ...