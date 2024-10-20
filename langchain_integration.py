#this is a version that works in app, but cant get embeddings.

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
    try:
        # Call the model with the input text
        response = llama_model(text)
        
        # Debug: Print the response to check its structure
        print("Model Response:", response)
        
        # Extract embeddings from the response if they exist (adjust this as per the model's output)
        embeddings = extract_embeddings_from_response(response)
        
        # Ensure that embeddings were found
        if embeddings is None:
            raise ValueError("No embeddings found in the model's response.")
        
        return embeddings
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return None

# Extract embeddings from the model response (adjust based on the actual response structure)
def extract_embeddings_from_response(response):
    try:
        # Assuming the response is a dictionary or structured output, adjust accordingly
        if 'embeddings' in response:
            embeddings = response['embeddings']
            return embeddings
        else:
            # If response structure is different, adjust this block
            print("Embeddings not found in response. Please check response format.")
            return None
    except Exception as e:
        print(f"Error extracting embeddings from response: {e}")
        return None

# Compute similarity using embeddings from the Llama model
def compute_similarity(fact, response, llama_model):
    # Get embeddings for the fact and response using the Gradio Llama model
    fact_embeddings = get_embeddings(fact, llama_model)
    response_embeddings = get_embeddings(response, llama_model)
    
    # Check if embeddings were successfully retrieved
    if fact_embeddings is None or response_embeddings is None:
        print("Error: Could not retrieve embeddings for similarity computation.")
        return 0.0
    
    # Compute cosine similarity between the fact and the response embeddings
    similarity_matrix = cosine_similarity(fact_embeddings.cpu().numpy(), response_embeddings.cpu().numpy())
    
    return similarity_matrix[0][0]  # Return similarity between the fact and response

# Fact-checking logic that combines the model and additional documents
def check_fact_with_model(fact, additional_docs_json, llama_model):
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
    # Dummy implementation, adjust based on actual response structure
    return "Reference 1, Reference 2"


Sidney Berman made significant contributions to cancer research