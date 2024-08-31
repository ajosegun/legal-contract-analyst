import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# Load environment variables
load_dotenv()


class Config:
    # Literal AI configurations
    LITERAL_API_KEY = os.getenv("LITERAL_API_KEY")

    # Azure OpenAI configurations
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT")
    AZURE_OPENAI_MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"
    )
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    AZURE_OPENAI_API_TYPE = os.getenv("AZURE_OPENAI_API_TYPE")


config = Config()

# Initialize Azure OpenAI
llm = AzureChatOpenAI(
    deployment_name=config.AZURE_DEPLOYMENT,
    temperature=0,
    openai_api_version=config.AZURE_OPENAI_API_VERSION,
    openai_api_type=config.AZURE_OPENAI_API_TYPE,
    azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
    api_key=config.AZURE_OPENAI_API_KEY,
)

embeddings_model = AzureOpenAIEmbeddings(
    azure_deployment=config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    openai_api_version=config.AZURE_OPENAI_API_VERSION,
)
