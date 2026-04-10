from langchain_community.document_loaders import PyPDFLoader

def pdf_loader(pdf_pathway_list):

    documents = []

    for pathway in pdf_pathway_list:
        loader = PyPDFLoader(pathway)
        docs = loader.load()

        documents.extend(docs)

    return documents