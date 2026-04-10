from langchain_groq import ChatGroq
from reranking import rerank_documents
def answer_question(question, vectorstore):

    llm = ChatGroq(model='llama-3.3-70b-versatile')

    #  Os 10 chunks mais semanticamente compatíveis com a solicitação do usuário
    similar_documents = vectorstore.similarity_search(
        question,
        k=10
    )

    #  Reranking do top 8 selecionado acima
    reranked_documents = rerank_documents(
        question=question,
        documents=similar_documents,
        llm=llm
    )

    #  Seleção dos top X dentre os 10 melhores contextos rerankeados
    final_documents = reranked_documents[:4]

    final_context = '\n\n'.join(
        doc.page_content for doc in final_documents
    )

    final_prompt = f'''
Você é um agente de RH corporativo.
Responda APENAS com base nas políticas internas abaixo.

Contexto:
{final_context}

Pergunta:
{question}
'''
    
    response = llm.invoke(final_prompt)

    return response.content, final_context