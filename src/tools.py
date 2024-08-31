from langchain.tools import Tool
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.pydantic_v1 import BaseModel, Field
from typing import List, Dict
import pandas as pd
from literalai import LiteralClient
from src.helper import load_and_process_documents, setup_rag_system
from src.config import embeddings_model, Config
import os
from fuzzywuzzy import fuzz

# Initialize Literal AI client
literal_client = LiteralClient(api_key=Config.LITERAL_API_KEY)

contracts_folder = "./data/contracts"
all_documents = []
for filename in os.listdir(contracts_folder):
    if filename.endswith(".pdf") or filename.endswith(".txt"):
        file_path = os.path.join(contracts_folder, filename)
        documents = load_and_process_documents(file_path)
        all_documents.extend(documents)

vectorstore = setup_rag_system(all_documents, embeddings_model)


class VendorPerformanceInput(BaseModel):
    vendor_name: str = Field(description="Name of the vendor")


@tool("vendor-performance-tool")
def get_vendor_performance_old(vendor_name: str) -> str:
    """
    Get performance data for a specific vendor from the old CSV file
    """

    try:
        df = pd.read_csv("data/Vendor_Performance2.csv", sep=";")
        df["similarity"] = df["Vendor Name"].apply(
            lambda x: fuzz.ratio(x.lower(), vendor_name.lower())
        )
        vendor_data = df[df["similarity"] == df["similarity"].max()]

        if vendor_data.empty:
            performance_summary = (
                f"No performance data available for vendor: {vendor_name}"
            )
        else:
            performance_summary = f"Vendor Performance Summary for {vendor_name}:\n"
            performance_summary += (
                f"Service Provided: {vendor_data['Service Provided'].values[0]}\n"
            )
            performance_summary += f"Performance Rating: {vendor_data['Performance Rating (1-10)'].values[0]}/10\n"
            performance_summary += f"Comments: {vendor_data['Comments'].values[0]}\n"

        return performance_summary
    except Exception as e:
        print(f"Error in get_vendor_performance: {e}")


@tool("new-vendor-performance-tool")
def get_vendor_performance(vendor_name: str) -> str:
    """
    Get performance data for a specific vendor from the CSV file
    """

    try:
        df = pd.read_csv("data/Vendor_Performance.csv")
        df["similarity"] = df["Vendor Name"].apply(
            lambda x: fuzz.ratio(x.lower(), vendor_name.lower())
        )
        vendor_data = df[df["similarity"] == df["similarity"].max()]

        if vendor_data.empty:
            performance_summary = (
                f"No performance data available for vendor: {vendor_name}"
            )
        else:
            performance_summary = f"Vendor Performance Summary for {vendor_name}:\n"
            performance_summary += (
                f"Service Provided: {vendor_data['Service Provided'].values[0]}\n"
            )
            performance_summary += f"Performance Rating: {vendor_data['Performance Rating (1-10)'].values[0]}/10\n"
            performance_summary += (
                f"On-time Delivery: {vendor_data['On-time Delivery (%)'].values[0]}%\n"
            )
            performance_summary += f"Quality of Deliverables: {vendor_data['Quality of Deliverables (1-10)'].values[0]}/10\n"
            performance_summary += (
                f"Cost Efficiency: {vendor_data['Cost Efficiency (%)'].values[0]}%\n"
            )
            performance_summary += (
                f"Renewal Status: {vendor_data['Renewal Status'].values[0]}\n"
            )
            performance_summary += f"Comments: {vendor_data['Comments'].values[0]}\n"

        return performance_summary
    except Exception as e:
        print(f"Error in get_new_vendor_performance: {e}")


@tool("custom_retriever")
def custom_retriever(query: str):
    """
    useful to retrieve contracts documents and get information about them.
    """

    try:
        results = vectorstore.similarity_search_with_score(query, k=5)
        return results
    except Exception as e:
        print(f"Error: {e}")
        raise ValueError(f"Error in custom_retriever: {e}")


@tool("single_contract_tool")
def single_contract_tool(chunks: List[str], embeddings: List[str]):
    """
    Use this tool analyse this new contract.
    Take into consideration the existing contracts in the company and the current vendor’s performance.
    Suggest improvements on new contracts to avoid getting performance issues with new vendors
    """

    try:
        return chunks, embeddings
    except Exception as e:
        print(f"Error: {e}")


wrapper = DuckDuckGoSearchAPIWrapper(region="us-en", time="m", max_results=3)
search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news")


tools_list = [get_vendor_performance, custom_retriever, search]
