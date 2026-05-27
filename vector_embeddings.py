import os
import glob
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def chunk_anime_json(json_file, anime_name):
    """Chunk a merged anime json file, adding title and anime as metadata."""
    with open(json_file, "r", encoding="utf-8") as f:
        articles = json.load(f)
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    documents = []
    for article in articles:
        title = article.get("title", "")
        text = article.get("text", "")
        if not text.strip():
            continue
        for chunk in splitter.split_text(text):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "title": title,
                        "anime": anime_name
                    }
                )
            )
    return documents

if __name__ == "__main__":
    """Process all merged anime json files: chunk, embed, and store in FAISS vectorstore."""
    json_files = glob.glob(os.path.join("merged_json_files", "*_all_articles_merged.json"))
    embeddings = OpenAIEmbeddings()
    for json_file in json_files:
        anime_name = os.path.basename(json_file).replace("_all_articles_merged.json", "")
        print(f"Processing {anime_name}...")
        documents = chunk_anime_json(json_file, anime_name)
        if not documents:
            print(f"No documents found for {anime_name}, skipping.")
            continue
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(os.path.join("vector_stores", anime_name))
        print(f"Saved vectorstore for {anime_name} with {len(documents)} chunks.")

