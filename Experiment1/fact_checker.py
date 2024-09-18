from pdf_converter import pdf_to_json
from langchain_integration import check_fact_with_model

def fact_checker(fact, additional_documents):
    if additional_documents:
        additional_documents_json = pdf_to_json(additional_documents)
    else:
        additional_documents_json = None
    
    verdict, similarity, references = check_fact_with_model(fact, additional_documents_json)
    return verdict, similarity, references
