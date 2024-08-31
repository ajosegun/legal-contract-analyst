from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from src.config import embeddings_model
from src.evaluator import evaluate
from typing import List, Dict
import sqlite3
import pandas as pd


def process_uploaded_pdf(file_path: str) -> List[str]:
    loader = PyPDFLoader(file_path)
    pdf_text = loader.load()[0].page_content
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_text(pdf_text)
    return pdf_text, embeddings_model.embed_documents(chunks)


def load_and_process_documents(file_path: str) -> List[str]:
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Please use PDF or TXT files.")

    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)
    return chunks


def setup_rag_system(documents: List[str], embeddings):
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


def setup_vendor_performance_db():
    conn = sqlite3.connect("vendor_performance.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS vendor_performance
                 (vendor_id TEXT, contract_id TEXT, metric TEXT, value REAL, date TEXT)""")
    conn.commit()
    conn.close()


def insert_vectors(vectorstore, new_documents: List[str], embeddings):
    new_vectors = embeddings.embed_documents(new_documents)
    vectorstore.add_embeddings(zip(new_documents, new_vectors))
    return vectorstore


def extract_vendor_data(file_path: str) -> Dict[str, pd.DataFrame]:
    # This is a placeholder function. In a real-world scenario, I will implement
    # a logic to extract data from your vendor management system.
    df = pd.read_csv(file_path)
    return {"vendors": df}


def clean_and_normalize_data(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    for key, df in data.items():
        df.drop_duplicates(inplace=True)
        df.fillna(0, inplace=True)
        numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
        df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].mean()) / df[
            numeric_columns
        ].std()

        data[key] = df

    return data


def etl_pipeline(contract_file: str, vendor_data_file: str, embeddings):
    # Extract and process contract data
    contract_documents = load_and_process_documents(contract_file)
    vendor_data = extract_vendor_data(vendor_data_file)
    cleaned_vendor_data = clean_and_normalize_data(vendor_data)
    vectorstore = setup_rag_system(contract_documents, embeddings)

    return vectorstore, cleaned_vendor_data


def log_performance(function_name: str, execution_time: float, accuracy: float = None):
    conn = sqlite3.connect("performance_logs.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS performance_logs
                 (function_name TEXT, execution_time REAL, accuracy REAL, timestamp TEXT)""")
    c.execute(
        "INSERT INTO performance_logs VALUES (?, ?, ?, datetime('now'))",
        (function_name, execution_time, accuracy),
    )
    conn.commit()
    conn.close()
