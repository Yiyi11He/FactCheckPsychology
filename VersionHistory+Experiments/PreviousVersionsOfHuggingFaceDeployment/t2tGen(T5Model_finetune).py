#deployed currently on factcheckpsychology for checking.

import gradio as gr
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load your model and tokenizer from Hugging Face Hub
model_name = "LilTomat0/FactCheckPsychology_T5small/t5-finetuned"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Define the function to process inputs and generate outputs
def generate_correction(input_text):
    # Tokenize the input and generate output
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=128, num_beams=5, early_stopping=True)
    
    # Decode the output
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

# Create Gradio interface
iface = gr.Interface(
    fn=generate_correction,
    inputs="text",
    outputs="text",
    title="Academic Writing Correction",
    description="Enter a passage for correction. The model will focus on common writing issues such as grammar, jargon, tone, and structure."
)

# Launch the interface
iface.launch()