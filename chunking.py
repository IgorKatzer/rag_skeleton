from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunker(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=250
    )

    return text_splitter.split_documents(documents)