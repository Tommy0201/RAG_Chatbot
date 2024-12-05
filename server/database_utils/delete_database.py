import os
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()
os.getenv("OPENAI_API_KEY")

def delete_database():
    db = Chroma(
        persist_directory="chroma",
        embedding_function=OpenAIEmbeddings()
    )
    db.delete_collection()
    data_directory = "data"
    if os.path.exists(data_directory):
        shutil.rmtree(data_directory)
        os.makedirs(data_directory) 
    return
    

if __name__ == "__main__":
    print(delete_database())