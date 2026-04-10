import unicodedata

def remove_accent(text):
    normalized_text = unicodedata.normalize('NFD', text)
    accentless_text = ''.join(
        char for char in normalized_text
        if unicodedata.category(char) != 'Mn'
    )

    return accentless_text

def enriching_preprocessing_chunks(chunks):
    
    for chunk in chunks:
        text = remove_accent(chunk.page_content).lower()

        if "ferias" in text:
            chunk.metadata["categoria"] = "ferias"
        elif "home office" in text or "remoto" in text:
            chunk.metadata["categoria"] = "home_office"
        elif "conduta" in text or "ética" in text:
            chunk.metadata["categoria"] = "conduta"
        else:
            chunk.metadata["categoria"] = "geral"

    return chunks