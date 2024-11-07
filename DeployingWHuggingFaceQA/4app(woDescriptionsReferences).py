import gradio as gr
import pymupdf
from transformers import pipeline, AutoTokenizer, AutoModel
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import torch
from fuzzywuzzy import fuzz

# Load the distilbert QA model and tokenizer
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
embedding_model = AutoModel.from_pretrained("distilbert-base-uncased")

# Modify the text extraction to split into sentences
def extract_text_with_references(pdf_path):
    doc = pymupdf.open(pdf_path)
    text_content = []
    
    # Iterate through pages to extract text
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text")
        # Split by sentences using a period
        sentences = page_text.split(".")
        
        for sentence in sentences:
            clean_sentence = sentence.strip()
            if len(clean_sentence) > 5:  # Only keep meaningful sentences
                text_content.append({
                    "page": page_num,
                    "text": clean_sentence
                })
    
    doc.close()
    return text_content

# Reference finder with fuzzy matching on sentences
def find_references(fact, pdf_text_content, threshold=85):  # Increased threshold
    references = []
    
    for content in pdf_text_content:
        # Fuzzy match the fact with each sentence from the PDF
        similarity_score = fuzz.token_set_ratio(fact.lower(), content["text"].lower())
        
        # If the similarity score is above the threshold, consider it a match
        if similarity_score >= threshold:
            # Limit the reference to a specific sentence or small portion of text
            references.append(f"Page {content['page']}: \"{content['text']}\"")
    
    return references if references else ["No close matches found for the fact."]


# exact match
def exact_match(prediction, fact):
    return 1 if prediction.strip().lower() == fact.strip().lower() else 0

# f1 score
def f1_score(prediction, fact):
    pred_tokens = prediction.strip().lower().split()
    fact_tokens = fact.strip().lower().split()
    
    common_tokens = Counter(pred_tokens) & Counter(fact_tokens)
    num_common = sum(common_tokens.values())
    
    if num_common == 0:
        return 0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(fact_tokens)
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# cosine similarity
def cosine_similarity_score(fact_embeddings, prediction_embeddings):
    return cosine_similarity(fact_embeddings, prediction_embeddings)[0][0]

# Function to extract embeddings using DistilBERT
def get_embeddings(text, embedding_model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()  # Return mean of hidden states as embeddings

# Compute all four scores
def compute_all_scores(prediction, fact, fact_embeddings, prediction_embeddings, result):
    em_score = exact_match(prediction, fact)
    f1 = f1_score(prediction, fact)
    qa_confidence = result['score']  # QA Confidence Score from the result
    cosine_sim = cosine_similarity_score(fact_embeddings, prediction_embeddings)
    
    return {
        "Exact Match": em_score,
        "F1 Score": f1,
        "QA Confidence": qa_confidence,
        "Cosine Similarity": cosine_sim
    }

# Function to handle fact-checking with an optional PDF document
def fact_checker(fact, pdf_file=None):
    if pdf_file:
        pdf_text_content = extract_text_with_references(pdf_file)
        context = " ".join([entry["text"] for entry in pdf_text_content])  # Combine all lines as context
    else:
        context = "Psychology knowledge base"
    
    # Use the QA model to check the fact against the context
    result = qa_model(question=fact, context=context)
    
    # Extract the answer and similarity score
    answer = result['answer']

    # Generate embeddings for the fact and predicted answer
    fact_embeddings = get_embeddings(fact, embedding_model, tokenizer)
    predicted_embeddings = get_embeddings(answer, embedding_model, tokenizer)

    similarity_score = cosine_similarity_score(fact_embeddings, predicted_embeddings)
    
    # Find specific references from the PDF (only page number and quote)
    if pdf_file:
        references = find_references(answer, pdf_text_content)
        # Format references with each one on a new line
        formatted_references = "\n".join(references)
    else:
        formatted_references = "Default knowledge base"
    
    # Set a verdict based on a similarity threshold (e.g., 0.8)
    verdict = "True" if similarity_score > 0.73 else "False"
    
    # Compute all four scores
    scores = compute_all_scores(answer, fact, fact_embeddings, predicted_embeddings, result)
    
    # Return each value individually
    return (
        verdict,
        scores['Exact Match'],  # Exact Match Score
        scores['F1 Score'],     # F1 Score
        scores['QA Confidence'],# QA Confidence Score
        scores['Cosine Similarity'],  # Cosine Similarity Score
        formatted_references    # References
    )

# Gradio interface 
fact_checker_interface = gr.Interface(
    fn=fact_checker,
    inputs=[
        gr.Textbox(label="Enter the fact to be checked"),
        gr.File(label="Upload additional PDF documents (optional)", type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Verdict"),
        gr.Textbox(label="Exact Match Score"),
        gr.Textbox(label="F1 Score"),
        gr.Textbox(label="QA Confidence Score"),
        gr.Textbox(label="Cosine Similarity Score"),
        gr.Textbox(label="References")
    ],
    title="Fact Checker with Multiple Scores"
)

# Launch the Gradio app
if __name__ == "__main__":
    fact_checker_interface.launch()