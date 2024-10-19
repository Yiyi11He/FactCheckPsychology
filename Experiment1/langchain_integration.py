from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

#authentication to huggingface - stored inside laptop desktop passwordmanager.doc

# # LLaMA 7B model
# model_name = "meta-llama/Llama-2-7b-hf"  
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", low_cpu_mem_usage=True)


# # Load the Llama 3.1 model and tokenizer
# model_name = "mattshumer/Reflection-Llama-3.1-70B"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Use low_cpu_mem_usage and device_map options for efficient model loading
# model = AutoModelForCausalLM.from_pretrained(
#     model_name, 
#     device_map="auto",  # Automatically map to available devices (GPU or CPU)
#     low_cpu_mem_usage=True  # Reduce memory usage for large models
# )

# Load the Meta-Llama-3-8B-Instruct model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", low_cpu_mem_usage=True)


# Use the model to generate a response based on fact and retrieved context
def llama_fact_checker(fact, context):
    prompt = f"Fact: {fact}\nContext: {context}\nIs the fact true? Provide an explanation."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate the response
    outputs = model.generate(inputs['input_ids'], max_length=512)  # Adjust max_length for longer responses
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



def load_internal_resource():
    
    with open('train.jsonl', 'r') as file:
        data = json.load(file)
    return data

# Use Langchain to check the fact using the Llama model
def check_fact_with_model(fact, additional_docs_json):
    # Load the internal resource into a vector store
    resource = load_internal_resource()
    vector_store = FAISS.from_texts([str(item) for item in resource['content']], OpenAIEmbeddings())
    
    # Use the vector store to retrieve similar documents to the fact
    context = vector_store.similarity_search(fact, k=5)
    
    # Combine the fact and retrieved context for Llama to process
    context_text = "\n".join([str(doc) for doc in context])
    response = llama_fact_checker(fact, context_text)
    
    # Perform similarity checks, compute references, etc.
    similarity_level = compute_similarity(fact, response)
    references = extract_references(response)
    
    verdict = "True" if similarity_level > 0.8 else "False"
    return verdict, similarity_level, references

def compute_similarity(fact, response):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([fact, response])
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix[0][1]

def extract_references(response):
    # Extract references from the Llama-generated response 
    return "Reference 1, Reference 2"