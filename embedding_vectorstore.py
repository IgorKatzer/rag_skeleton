from langchain_community.vectorstores.chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st

@st.cache_resource
def create_vectorstore(chunks):
    
   embed_model = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')

   vectorstore = Chroma.from_documents(
      documents=chunks,
      embedding=embed_model,
      persist_directory='chromadb_rh'
   )

   return vectorstore
