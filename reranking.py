from langchain_core.prompts import PromptTemplate

def rerank_documents(question, documents, llm):

    prompt_rerank = PromptTemplate(
        input_variables=["pergunta", "texto"],
        template="""
Você é um especialista em políticas internas de RH.

Pergunta do usuário:
{pergunta}

Trecho do documento:
{texto}

Avalie a relevância desse trecho para responder a pergunta.
Responda apenas com um número de 0 a 10.
"""
    )

    scored_documents = []

    for doc in documents:
        score = llm.invoke(
            prompt_rerank.format(
                pergunta=question,
                texto=doc.page_content
            )
        ).content

        try:
            score = float(score)
        except:
            score = 0

        scored_documents.append((score, doc))

    # Ordena do mais relevante para o menos relevante
    ordenated_documents = sorted(
        scored_documents,
        key=lambda x: x[0],
        reverse=True
    )

    # Retorna apenas os documentos
    return [doc for _, doc in ordenated_documents]