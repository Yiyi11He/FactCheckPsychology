from pdf_converter import pdf_to_json
from similarity import load_llama_model, check_fact_with_model  # Import from similarity

# Load the llama model once when the fact_checker is called
llama_model = load_llama_model()

def fact_checker(fact, additional_documents):
    if additional_documents:
        additional_documents_json = pdf_to_json(additional_documents)
    else:
        additional_documents_json = None
    
    # Call the fact-checking function and pass the llama model
    verdict, similarity, references = check_fact_with_model(fact, additional_documents_json, llama_model)
    return verdict, similarity, references
