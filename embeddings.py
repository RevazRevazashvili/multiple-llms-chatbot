from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
import pinecone
import os
from langchain_community.vectorstores import Pinecone
import document_processing

os.environ['PINECONE_API_KEY'] = 'your pinecone api here'


# function to load pdf document
def load_docs(directory):
    loader = PyPDFLoader(directory)
    documents = loader.load()
    return documents


# load document
# not use this if you are using document processing else uncomment
# documents = load_docs('your pdf file here')


# splitting document with chunk_size
def split_docs(documents, chunk_size=500, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs


# docs = split_docs(documents)  # text spliter
docs = document_processing.docs_list


# model to create embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


# Vector DataBase
# initialize pinecone
environment = "your environment"
pc = pinecone.Pinecone(
    api_key=os.environ['PINECONE_API_KEY'],
    environment=environment
    )

# database index name
index_name = "your index name here"

# storing embeddings on database
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
