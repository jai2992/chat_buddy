import langchain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import os

os.environ["PINECONE_API_KEY"] = 'a192f216-4cd3-4b0c-b5b7-1db783efd833'

def doc_load(documents):
    file_loader=PyPDFDirectoryLoader(documents)
    docs=file_loader.load()
    return docs
doc=doc_load('sece-doc/')

def chunk_split(docs):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    doc=text_splitter.split_documents(docs)
    return doc
documents=chunk_split(doc)

embeddings = HuggingFaceInferenceAPIEmbeddings(api_key='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', model_name="sentence-transformers/all-MiniLM-l6-v2")
PineconeVectorStore.from_documents(documents, embeddings, index_name='sece-chat')