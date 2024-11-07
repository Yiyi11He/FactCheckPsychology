import gradio as gr
import torch
from pdf_converter import pdf_to_json
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load the Meta-Llama-3-8B-Instruct model via Gradio
def load_llama_model():
    model = gr.load("models/meta-llama/Meta-Llama-3-8B-Instruct")
    return model

# Use the Gradio model to generate embeddings for a given input text
def get_embeddings(text, llama_model):
    response = llama_model(text)
    # Assuming the model returns embeddings or you can extract them from the response
    embeddings = extract_embeddings_from_response(response)
    return embeddings

# Extract embeddings from the model response (you may need to adjust this based on the response structure)
def extract_embeddings_from_response(response):
    # Process the model's response to extract embeddings (depends on output format)
    embeddings = response['embeddings']  # Adjust this based on actual output structure
    return embeddings

# Compute similarity using embeddings from the Llama model
def compute_similarity(fact, response, llama_model):
    # Get embeddings for the fact and response using the Gradio Llama model
    fact_embeddings = get_embeddings(fact, llama_model)
    response_embeddings = get_embeddings(response, llama_model)
    
    # Compute cosine similarity between the fact and the response embeddings
    similarity_matrix = cosine_similarity(fact_embeddings.cpu().numpy(), response_embeddings.cpu().numpy())
    
    return similarity_matrix[0][0]  # Return similarity between the fact and response

# Fact-checking logic that combines the model and additional documents
def check_fact_with_model(fact, additional_docs_json, llama_model):
    # Load internal resource if needed (for context)
    # resource = load_internal_resource()

    # Retrieve context similar to the fact (you can implement a more sophisticated retrieval mechanism here)
    context = "Extracted context from internal resource"
    
    # Use the llama model to check the fact against the retrieved context
    response = f"Fact: {fact}. Context: {context}."
    
    # Perform similarity checks
    similarity_level = compute_similarity(fact, response, llama_model)
    references = extract_references(response)
    
    verdict = "True" if similarity_level > 0.8 else "False"
    return verdict, similarity_level, references

# Extract references from the response
def extract_references(response):
    return "Reference 1, Reference 2"

# # Update this with psychology internal resource.
# def load_internal_resource():
#     with open('train.jsonl', 'r') as file:
#         data = json.load(file)
#     return data