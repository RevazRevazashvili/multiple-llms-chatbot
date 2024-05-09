from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
import pinecone
import os
from langchain_community.vectorstores import Pinecone
import document_processing

os.environ['PINECONE_API_KEY'] = 'your pinecone api here'

#load data
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
