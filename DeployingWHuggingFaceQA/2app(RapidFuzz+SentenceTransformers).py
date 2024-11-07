#multiple sentence input
#sentence tranformers
#Multiple pdf support
import gradio as gr
import pymupdf
from transformers import pipeline
from collections import Counter
import torch
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load sentence transformer model
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Extract embeddings with Sentence Transformers
def get_embeddings(text, embedding_model):
    return embedding_model.encode(text, convert_to_tensor=True)

# Text extraction with references
def extract_text_with_references(pdf_path):
    doc = pymupdf.open(pdf_path)
    text_content = []
    
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text")
        sentences = page_text.replace('\n', ' ').split('. ')
        
        for sentence in sentences:
            clean_sentence = sentence.strip()
            if len(clean_sentence) > 5:
                text_content.append({"page": page_num, "text": clean_sentence})
    
    doc.close()
    return text_content

# Find references with fuzzy and token matching
def find_references(fact, pdf_text_content, fuzzy_threshold=85, semantic_threshold=0.75):
    references = []
    fact_embedding = embedding_model.encode(fact, convert_to_tensor=True)

    for content in pdf_text_content:
        # Fuzzy match using `fuzz.partial_ratio` and `fuzz.token_set_ratio`
        partial_fuzzy_score = fuzz.partial_ratio(fact.lower(), content["text"].lower())
        token_set_score = fuzz.token_set_ratio(fact.lower(), content["text"].lower())
        
        # Semantic similarity with sentence transformers
        content_embedding = embedding_model.encode(content["text"], convert_to_tensor=True)
        semantic_score = util.cos_sim(fact_embedding, content_embedding).item()

        # If any of the scores meet the threshold, consider it a match
        if partial_fuzzy_score >= fuzzy_threshold or token_set_score >= fuzzy_threshold or semantic_score >= semantic_threshold:
            references.append(
                f"Page {content['page']}: \"{content['text']}\" "
                f"(Partial Fuzzy Score: {partial_fuzzy_score}, Token Set Score: {token_set_score}, Semantic Score: {semantic_score:.2f})"
            )
    
    return references if references else ["No close matches found for the fact."]

# Define the cosine similarity calculation
def cosine_similarity_score(fact_embeddings, prediction_embeddings):
    return util.cos_sim(fact_embeddings, prediction_embeddings).item()

# Load QA model
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Compute exact match, F1 score, and other metrics
def exact_match(prediction, fact):
    return 1 if prediction.strip().lower() == fact.strip().lower() else 0

def f1_score(prediction, fact):
    pred_tokens = prediction.strip().lower().split()
    fact_tokens = fact.strip().lower().split()
    
    common_tokens = Counter(pred_tokens) & Counter(fact_tokens)
    num_common = sum(common_tokens.values())
    
    if num_common == 0:
        return 0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(fact_tokens)
    
    return 2 * (precision * recall) / (precision + recall)

# Compute all four scores
def compute_all_scores(prediction, fact, fact_embeddings, prediction_embeddings, result):
    em_score = exact_match(prediction, fact)
    f1 = f1_score(prediction, fact)
    qa_confidence = result['score']
    cosine_sim = cosine_similarity_score(fact_embeddings, prediction_embeddings)
    
    return {
        "Exact Match": em_score,
        "F1 Score": f1,
        "QA Confidence": qa_confidence,
        "Cosine Similarity": cosine_sim
    }

# Combined similarity function
def token_cosine_similarity(text1, text2, model):
    tokens1 = text1.split()
    tokens2 = text2.split()
    token_embeddings1 = np.mean([model.encode(token) for token in tokens1], axis=0)
    token_embeddings2 = np.mean([model.encode(token) for token in tokens2], axis=0)
    return util.cos_sim(token_embeddings1, token_embeddings2).item()

def combined_similarity(fact, prediction, model, sentence_weight=0.7, token_weight=0.3):
    fact_embedding = model.encode(fact, convert_to_tensor=True)
    prediction_embedding = model.encode(prediction, convert_to_tensor=True)
    sentence_sim = util.cos_sim(fact_embedding, prediction_embedding).item()
    
    token_sim = token_cosine_similarity(fact, prediction, model)
    
    return sentence_weight * sentence_sim + token_weight * token_sim

# Fact checker function for Gradio
def fact_checker(fact, pdf_files=None):
    results = []
    fact_sentences = fact.split('. ')
    
    if pdf_files:
        for pdf_file in pdf_files:
            pdf_text_content = extract_text_with_references(pdf_file.name)
            pdf_results = []
            
            for fact_sentence in fact_sentences:
                context = " ".join([entry["text"] for entry in pdf_text_content])
                result = qa_model(question=fact_sentence, context=context)
                answer = result['answer']
                
                fact_embeddings = get_embeddings(fact_sentence, embedding_model)
                predicted_embeddings = get_embeddings(answer, embedding_model)
                similarity_score = combined_similarity(fact_sentence, answer, embedding_model)
                
                verdict = "True" if similarity_score > 0.73 else "False"
                
                scores = compute_all_scores(answer, fact_sentence, fact_embeddings, predicted_embeddings, result)
                references = find_references(fact_sentence, pdf_text_content)
                
                pdf_results.append({
                    "sentence": fact_sentence,
                    "verdict": verdict,
                    "exact_match": scores['Exact Match'],
                    "f1_score": scores['F1 Score'],
                    "qa_confidence": scores['QA Confidence'],
                    "cosine_similarity": scores['Cosine Similarity'],
                    "references": "\n".join(references)
                })
            
            results.append({f"PDF: {pdf_file.name}": pdf_results})

    else:
        context = "Psychology knowledge base"
        result = qa_model(question=fact, context=context)
        answer = result['answer']

        fact_embeddings = get_embeddings(fact, embedding_model)
        predicted_embeddings = get_embeddings(answer, embedding_model)
        similarity_score = combined_similarity(fact, answer, embedding_model)
        
        verdict = "True" if similarity_score > 0.73 else "False"
        scores = compute_all_scores(answer, fact, fact_embeddings, predicted_embeddings, result)
        
        results.append({
            "verdict": verdict,
            "exact_match": scores['Exact Match'],
            "f1_score": scores['F1 Score'],
            "qa_confidence": scores['QA Confidence'],
            "cosine_similarity": scores['Cosine Similarity'],
            "references": "Default knowledge base"
        })

    if pdf_files:
        first_pdf_result = results[0][f"PDF: {pdf_files[0].name}"][0]
        return (
            first_pdf_result["verdict"], 
            first_pdf_result["exact_match"], 
            first_pdf_result["f1_score"], 
            first_pdf_result["qa_confidence"], 
            first_pdf_result["cosine_similarity"], 
            first_pdf_result["references"]
        )
    else:
        first_result = results[0]
        return (
            first_result["verdict"], 
            first_result["exact_match"], 
            first_result["f1_score"], 
            first_result["qa_confidence"], 
            first_result["cosine_similarity"], 
            first_result["references"]
        )


example_fact_1 = "Ekman’s basic emotions include Anger, Disgust, Fear, Happiness, Sadness and Surprise."
example_pdf_path_1 = "IdentifyingDepressiononTwitter.pdf"

example_fact_2 = "Young children experience and perceive trauma exposure same as adolescents or adults."
example_pdf_path_2 = "The impact and long-term effects of childhood trauma.pdf" 


# Gradio interface
with gr.Blocks(css=".submit-button {background-color: orange; color: white; font-weight: bold;}") as demo:
    gr.Markdown("# Psychology Fact-Checking Helper Bot 🤗")
    gr.Markdown(
        """
        Hello, I am a Fact Checker Bot 😊 designed to verify your claims and assist with your studies. 
        I will evaluate the claims against the PDF files you provide and help you determine their accuracy.
        """
    )

    with gr.Row(): 
        # Section 1: Claim and Reference
        with gr.Column():
            gr.Markdown("## 1. Claims and References")
            fact_input = gr.Textbox(label="Enter the fact to be checked", lines=3)
            pdf_input = gr.Files(label="Upload additional PDF documents")

            # Submit and Clear buttons
            with gr.Row():
                clear_button = gr.Button("Clear")
                submit_button = gr.Button("Submit", elem_classes="submit-button")
            
            # Examples under submit button
            gr.Examples(
                examples=[
                    [example_fact_1, [example_pdf_path_1]],  
                    [example_fact_2, [example_pdf_path_2]]   
                ],
                inputs=[fact_input, pdf_input]
            )
                
            # Section 2: Results and Outcome
        with gr.Column():
            gr.Markdown("## 2. Results and Outcomes")
            verdict_output = gr.Textbox(label="Verdict", placeholder="Waiting for results...")
            exact_match_output = gr.Textbox(label="Exact Match Score", placeholder="Waiting for results...")
            f1_output = gr.Textbox(label="F1 Score", placeholder="Waiting for results...")
            qa_confidence_output = gr.Textbox(label="QA Confidence Score", placeholder="Waiting for results...")
            cosine_similarity_output = gr.Textbox(label="Cosine Similarity Score", placeholder="Waiting for results...")
            references_output = gr.Textbox(label="References", lines=6, placeholder="Waiting for results...")

    # Fact-checking function submit button
    submit_button.click(
        fn=fact_checker, 
        inputs=[fact_input, pdf_input],
        outputs=[verdict_output, exact_match_output, f1_output, qa_confidence_output, cosine_similarity_output, references_output]
    )

    # Clear button
    clear_button.click(
        fn=lambda: ("", None, "", "", "", "", ""),
        inputs=[],
        outputs=[fact_input, pdf_input, verdict_output, exact_match_output, f1_output, qa_confidence_output, cosine_similarity_output, references_output]
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

# # Initial Gradio interface
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