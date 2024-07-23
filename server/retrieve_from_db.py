import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def get_similarity_from_db(query_text):
    
    db = Chroma(
        persist_directory = "chroma", 
        embedding_function=OpenAIEmbeddings(api_key=openai_api_key)
    )   #Storing chunks as vectore in the Chrome Database

    results = db.similarity_search_with_score(query_text, k=3) # Search the database
    # print(results)

    context = """"""  #This will be passed into query template
    references = set() # Storing the sources and the page that answer was generated from
    """List of sources: """
    for result in results:
        context = context + (f"""
        {result[0].page_content}
                            """)
        page = result[0].metadata['page']
        source = result[0].metadata['source']
        references.add(f"file: {source}, page: {page} | ")
    # print("Context:")
    # print(context)
    # print("\nReferences:")
    # for ref in references:
    #     print(ref)
    return [context,references]


if __name__ == "__main__":
    context,references = get_similarity_from_db(query_text="Getting money from the bank")
    print("Context:")
    print(context)
    print("\nReferences:")
    for ref in references:
        print(ref)