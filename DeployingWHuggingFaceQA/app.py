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

# Function to handle fact-checking with multiple PDF documents
def fact_checker(fact, pdf_files=None):
    context = ""
    if pdf_files:
        for pdf_file in pdf_files:
            pdf_text_content = extract_text_with_references(pdf_file.name)
            context += " ".join([entry["text"] for entry in pdf_text_content])  # Combine all sentences as context
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
    
    # Find specific references from the PDF (page number and quote)
    references = []
    if pdf_files:
        for pdf_file in pdf_files:
            pdf_text_content = extract_text_with_references(pdf_file.name)
            references.extend(find_references(answer, pdf_text_content))
        formatted_references = "\n".join(references)
    else:
        formatted_references = "Default knowledge base"
    
    # Set a verdict based on a similarity threshold
    verdict = "True" if similarity_score > 0.73 else "False"
    
    # Compute all four scores
    scores = compute_all_scores(answer, fact, fact_embeddings, predicted_embeddings, result)

    
    return (
        verdict,
        scores['Exact Match'],  # Exact Match Score
        scores['F1 Score'],     # F1 Score
        scores['QA Confidence'],# QA Confidence Score
        scores['Cosine Similarity'],  # Cosine Similarity Score
        formatted_references    # References
    )

example_fact_1 = "Ekman’s basic emotions include Anger, Disgust, Fear, Happiness, Sadness and Surprise."
example_pdf_path_1 = "IdentifyingDepressiononTwitter.pdf"

example_fact_2 = "Young children experience and perceive trauma exposure same as adolescents or adults."
example_pdf_path_2 = "The impact and long-term effects of childhood trauma.pdf" 



# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Psychology Fact Checker")
    # Add description
    gr.Markdown(
        """
     
        Hello, I am an AI Fact Checker designed to verify your claims and assist with your studies. I will evaluate the claims against the PDF files you provide and help you determine their accuracy.

        """
    )

    with gr.Row(): 
        # Section 1: Claim and Reference
        with gr.Column():
            gr.Markdown("## 1. Claim and Reference")
            fact_input = gr.Textbox(label="Enter the fact to be checked", lines=3)
            pdf_input = gr.Files(label="Upload additional PDF documents")
            submit_button = gr.Button("Submit")
            
            # Examples directly under submit button
            gr.Examples(
                examples=[
                    [example_fact_1, [example_pdf_path_1]],  
                    [example_fact_2, [example_pdf_path_2]]   
                ],
                inputs=[fact_input, pdf_input]
            )
                
            # Section 2: Results and Outcome
        with gr.Column():
            gr.Markdown("## 2. Results and Outcome")
            verdict_output = gr.Textbox(label="Verdict", placeholder="Waiting for results...")
            exact_match_output = gr.Textbox(label="Exact Match Score", placeholder="Waiting for results...")
            f1_output = gr.Textbox(label="F1 Score", placeholder="Waiting for results...")
            qa_confidence_output = gr.Textbox(label="QA Confidence Score", placeholder="Waiting for results...")
            cosine_similarity_output = gr.Textbox(label="Cosine Similarity Score", placeholder="Waiting for results...")
            references_output = gr.Textbox(label="References", lines=6, placeholder="Waiting for results...")

    # Fact-checking function integration with submit button
    submit_button.click(
        fn=fact_checker, 
        inputs=[fact_input, pdf_input],
        outputs=[verdict_output, exact_match_output, f1_output, qa_confidence_output, cosine_similarity_output, references_output]
    )



    gr.Markdown(
        """
        ## Detailed Descriptions

        This application allows you to verify the accuracy of a fact by comparing it with 
        uploaded PDF documents. Enter a fact and optionally upload one or more PDF files to 
        check for matches. The application will output the verdict (True or False), and 
        various similarity metrics like Exact Match Score, F1 Score, QA Confidence Score, 
        and Cosine Similarity Score.
        
        **Exact Match Score**: a metric based on the strict character match of the claim 
        and the answer in the documents. 1 meaning the words matches exactly to the document’s texts, otherwise 0.
        
        **F1 Score**: calculates each word in the predicted sequence against the correct answer
        
        **QA Confidence Score**: represents how certain the model is about its answer based on the provided context
        
        **Cosine Similarity**: measures the semantic similarity between two sets of text. 
        This is also used to determine the verdict of the claim.

        This model currently uses [distilbert/distilbert-base-uncased-distilled-squad](https://huggingface.co/distilbert/distilbert-base-uncased-distilled-squad) 
        which is a Transformer model trained by distilling BERT base that is especially good at question answering.

        Created by Yiyi He, supervised by A.Prof. Sonny Pham. This project is solely 
        dedicated for Dr. Welber Marinovic and Master of Computing, Artificial Intelligence, Computer Science Project (COMP6016).

        """
    )


demo.launch()

# # Gradio interface with examples
# fact_checker_interface = gr.Interface(
#     fn=fact_checker,
#     inputs=[
#         gr.Textbox(label="Enter the fact to be checked"),
#         gr.Files(label="Upload additional PDF documents (optional)")  
#     ],
#     outputs=[
#         gr.Textbox(label="Verdict"),
#         gr.Textbox(label="Exact Match Score"),
#         gr.Textbox(label="F1 Score"),
#         gr.Textbox(label="QA Confidence Score"),
#         gr.Textbox(label="Cosine Similarity Score"),
#         gr.Textbox(label="References")
#     ],
#     examples=[
#         [example_fact_1, [example_pdf_path_1]],  
#         [example_fact_2, [example_pdf_path_2]]   
#     ],
#     title="Fact Checker"
# )
# # Launch the Gradio app
# if __name__ == "__main__":
#     fact_checker_interface.launch()