import os

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

os.environ["OPENAI_API_KEY"] = "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"


def text_split(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
        length_function = len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def check_and_add_chunks(db, chunks_with_ids):
    #check if new chunks already exist in Chroma database using their ids
    existing_chunk_items = db.get(include=[])
    existing_chunk_ids = set(existing_chunk_items["ids"])
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_chunk_ids:
            new_chunks.append(chunk)
    if len(new_chunks):
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        print(f"""new documents have been added, ids:
              {new_chunk_ids}
              """)
    else:
        print("no new documents added")
    return

def create_ids_to_chunks(chunks):
    last_source_page = None
    chunk_idx = 0
    for chunk in chunks:
        source = chunk.metadata["source"]
        page = chunk.metadata["page"]
        curr_source_page = f"{source}_{page}"
        if curr_source_page == last_source_page:
            chunk_idx +=1
        else: 
            chunk_idx = 0
        chunk_id = f"{source}_{page}_{chunk_idx}"
        chunk.metadata["id"] = chunk_id  
        last_source_page = curr_source_page
    return chunks

def add_data_Chroma(chunks):
    db = Chroma(
        persist_directory = "chroma", 
        embedding_function=OpenAIEmbeddings()
    )   #Storing chunks as vectore in the Chrome Database
    
    chunks_with_ids = create_ids_to_chunks(chunks) #returning chunks accompanied w id
    check_and_add_chunks(db, chunks_with_ids)
    
        
def database():
    document_loader = PyPDFDirectoryLoader("data")    #Loading documents
    documents = document_loader.load()                #Storing as list of pages
    chunks = text_split(documents)                    #Splitting docs into chunks   
     
    # with open("data/monopoly.txt","w") as file:
        
    #     for split in splits:
    #         file.write(f"""
    #                    {split}
    #                    ________________________
    #                    """)
    add_data_Chroma(chunks)                         #adding chunks to Chroma database

if __name__ == "__main__":
    database()

