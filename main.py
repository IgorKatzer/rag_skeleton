
#  ==============================================================
#  Injetando a chave da API da LLM como uma variável de ambiente
#  ==============================================================

import os
os.environ['GROQ_API_KEY'] = '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'

#  =========================================
#  Módulo de leitura e carregamento de PDFs
#  =========================================

from pdf_loader_reader import pdf_loader

pdf_pathway = [
    'C:\\Users\\Igor\\Documents\\Programacao\\ws-vscode\\skeleton_rag\\manuais\\codigo_conduta.pdf',
    'C:\\Users\\Igor\\Documents\\Programacao\\ws-vscode\\skeleton_rag\\manuais\\politica_ferias.pdf',
    'C:\\Users\\Igor\\Documents\\Programacao\\ws-vscode\\skeleton_rag\\manuais\\politica_home_office.pdf'
]

documents = pdf_loader(pdf_pathway)

#  ============================
#  Módulo de criação de chunks
#  ============================

from chunking import chunker

chunks = chunker(documents)

#  =======================================================
#  Módulo de enriquecimento de préprocessamento dos dados
#  =======================================================

from enrichment_preprocessment import enriching_preprocessing_chunks

chunks = enriching_preprocessing_chunks(chunks)

#  ==============================================
#  Módulo de embedding e criação do vector store
#  ==============================================

from embedding_vectorstore import create_vectorstore

vectorstore = create_vectorstore(chunks)

#  ==========================================
#  Modulo de ReRanking (maior assertividade)
#  ==========================================

'''
O módulo reranking.py será chamado na nossa pipeline de rag, após a declaração da LLM
'''

#  ====================================================
#  Modulo de pipeline do RAG + Interface com Stremalit
#  ====================================================

from pipeline_rag import answer_question
import streamlit as st

st.set_page_config(page_title="Agente de RH com RAG", layout="wide")
st.title("🤖 Agente de RH — Políticas Internas")

request = st.text_input("Digite sua pergunta sobre políticas internas de RH:")

if request:
    with st.spinner("Consultando políticas internas..."):
        documents = pdf_loader(pdf_pathway)
        chunks = chunker(documents)
        chunks = enriching_preprocessing_chunks(chunks)
        vectorstore = create_vectorstore(chunks)

        answer, source = answer_question(request, vectorstore)

    st.subheader("Resposta")
    st.write(answer)

    st.subheader("Fontes utilizadas")
    for i, doc in enumerate(source, start=1):
        st.markdown(f"**Trecho {i}**")
        st.write(f"Documento: {doc.metadata.get('documento')}")
        st.write(f"Categoria: {doc.metadata.get('categoria')}")
        st.write(doc.page_content)
        st.divider()
