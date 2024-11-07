import gradio as gr
import pymupdf
from transformers import pipeline
import torch
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
from collections import Counter
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

# Compute exact match
def exact_match(prediction, fact):
    return 1 if prediction.strip().lower() == fact.strip().lower() else 0

# Detect contradictory phrases in text
def has_contradictory_phrases(reference_text):
    contradiction_phrases = ["differently than", "different from", "not the same as", "in contrast to"]
    return any(phrase in reference_text.lower() for phrase in contradiction_phrases)

# Updated Fact Checker Function for Gradio
def fact_checker(fact, pdf_files=None):
    results = []
    fact_sentences = fact.split('. ')
    
    if pdf_files:
        for pdf_file in pdf_files:
            pdf_text_content = extract_text_with_references(pdf_file.name)
            pdf_results = []
            
            # Initialize sums and count for overall scoring
            total_fuzzy_score = 0
            total_token_set_score = 0
            total_semantic_score = 0
            reference_count = 0
            
            for fact_sentence in fact_sentences:
                # Extract relevant references and scores for each fact sentence
                references = find_references(fact_sentence, pdf_text_content)
                
                exact_match = int(any(
                    ("Partial Fuzzy Score" in ref and float(ref.split("Partial Fuzzy Score: ")[1].split(",")[0]) > 95) or
                    ("Token Set Score" in ref and float(ref.split("Token Set Score: ")[1].split(",")[0]) > 95)
                    for ref in references
                ))
                
                # Sum up fuzzy, token, and semantic scores across references
                for reference in references:
                    if "Partial Fuzzy Score" in reference and "Token Set Score" in reference and "Semantic Score" in reference:
                        # Safely parse scores from the reference text
                        ref_text = reference
                        partial_fuzzy_score = float(ref_text.split("Partial Fuzzy Score: ")[1].split(",")[0])
                        token_set_score = float(ref_text.split("Token Set Score: ")[1].split(",")[0])
                        semantic_score = float(ref_text.split("Semantic Score: ")[1].split(")")[0].strip())

                        # Add to totals
                        total_fuzzy_score += partial_fuzzy_score
                        total_token_set_score += token_set_score
                        total_semantic_score += semantic_score
                        reference_count += 1

                # Append references for display
                pdf_results.append({
                    "sentence": fact_sentence,
                    "exact_match": exact_match,
                    "references": "\n".join(references)
                })

            # Calculate the average scores if references were found
            avg_fuzzy_score = total_fuzzy_score / reference_count if reference_count > 0 else 0
            avg_token_set_score = total_token_set_score / reference_count if reference_count > 0 else 0
            avg_semantic_score = total_semantic_score / reference_count if reference_count > 0 else 0
            
            # Enhanced Verdict Logic
            verdict = "True" if (avg_fuzzy_score > 75 or avg_token_set_score > 75 or avg_semantic_score > 0.9) else "False"
            
            # Append final results for display
            results.append({
                "verdict": verdict,
                "exact_match": exact_match,
                "avg_fuzzy_score": avg_fuzzy_score,
                "avg_token_set_score": avg_token_set_score,
                "avg_semantic_score": avg_semantic_score,
                "references": "\n".join([ref["references"] for ref in pdf_results])
            })

    # Display the first result for Gradio output
    first_result = results[0]
    return (
        first_result["verdict"],                # Output 1: verdict
        first_result["exact_match"],            # Output 2: exact match score
        first_result["avg_fuzzy_score"],        # Output 3: averaged fuzzy score
        first_result["avg_token_set_score"],    # Output 4: averaged token set score
        first_result["avg_semantic_score"],     # Output 5: averaged semantic score
        first_result["references"]              # Output 6: references
    )
    

example_fact_1 = "Ekman’s basic emotions include Anger, Disgust, Fear, Happiness, Sadness and Surprise."
example_pdf_path_1 = "IdentifyingDepressiononTwitter.pdf"

example_fact_2 = "Young children experience and perceive trauma exposure same as adolescents or adults."
example_pdf_path_2 = "ImpactChildhoodTrauma.pdf" 

example_fact_3 = "One of the most common disabilities in the college population is ADHD with 6% of first-year students reporting having received an ADHD diagnosis."
example_pdf_path_3 = "CollegeADHDPredictor.pdf"

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
            
            # Examples section
            gr.Examples(
                examples=[
                    [example_fact_1, [example_pdf_path_1]],  
                    [example_fact_2, [example_pdf_path_2]],
                    [example_fact_3, [example_pdf_path_3]]
                ],
                inputs=[fact_input, pdf_input]
            )

        # Section 2: Results and Outcomes
        with gr.Column():
            gr.Markdown("## 2. Results and Outcomes")
            verdict_output = gr.Textbox(label="Verdict", placeholder="Waiting for results...")
            exact_match_output = gr.Textbox(label="Exact Match Score", placeholder="Waiting for results...")
            avg_fuzzy_score_output = gr.Textbox(label="Average Fuzzy Score", placeholder="Waiting for results...")
            avg_token_set_score_output = gr.Textbox(label="Average Token Set Score", placeholder="Waiting for results...")
            avg_semantic_score_output = gr.Textbox(label="Average Semantic Score", placeholder="Waiting for results...")
            references_output = gr.Textbox(label="References", lines=6, placeholder="Waiting for results...")

    # Fact-checking function submit button
        submit_button.click(
            fn=fact_checker, 
            inputs=[fact_input, pdf_input],
            outputs=[
                verdict_output,             # Output 1: verdict
                exact_match_output,         # Output 2: exact match score
                avg_fuzzy_score_output,     # Output 3: average fuzzy score
                avg_token_set_score_output, # Output 4: average token set score
                avg_semantic_score_output,  # Output 5: average semantic score
                references_output           # Output 6: references
            ]
        )

    # Clear button
    clear_button.click(
        fn=lambda: ("", "", "", "", "", "", "", ""),
        inputs=[],
        outputs=[fact_input, pdf_input, verdict_output, exact_match_output, avg_fuzzy_score_output, avg_token_set_score_output, avg_semantic_score_output, references_output]
    )

    gr.Markdown(
        """
        ## Detailed Descriptions
        This application verifies the accuracy of a fact by comparing it with uploaded PDF documents. 
        Using a combination of metrics below, it determines whether the fact is true or false given 
        the combined output scores from **Fuzzy Score**, **Token Set Score** and **Semantic Score**.
        Enter a fact and optionally upload one or more PDF files to check for matches. 
        The application will output the verdict (True or False), alongside similarity metrics such as 
        Exact Match Score, Fuzzy Score, Token Set Score, and Semantic Score.
        
        **Exact Match Score**: Measures a strict character-by-character match between the claim and the text in the documents. 
        A score of 1 indicates an exact match, while 0 indicates no exact match.
        
        **Fuzzy Score**: Uses approximate matching to determine similarity between the claim and document text, 
        accounting for minor variations. It’s useful for detecting similar phrasing without requiring an exact match.
        
        **Token Set Score**: Compares the sets of unique words in the claim and the document text, 
        allowing flexible matching regardless of word order or minor differences.
        
        **Semantic Score**: Measures the similarity in meaning between the claim and document text using 
        sentence embeddings, capturing overall intent even with different wording.

        This model currently uses [paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2) 
        This is a sentence-transformers model, it maps sentences and paragraphs to a 384 dimensional dense vector 
        space and can be used for tasks like clustering or semantic search.
        Created by Yiyi He, supervised by A.Prof. Sonny Pham. This project is solely 
        dedicated for Dr. Welber Marinovic and Master of Computing, Artificial Intelligence, Computer Science Project (COMP6016).
        """
    )

demo.launch()