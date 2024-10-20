import gradio as gr
from fact_checker import fact_checker
from huggingface_hub import InferenceClient
from similarity import *
from pdf_converter import pdf_to_json

# # load model using Gradio
# gr.load("models/meta-llama/Meta-Llama-3-8B-Instruct").launch()

# system message where the chatbot introduces itself and explains its role
system_intro_message = (
    "Hello, I am an AI chatbot specifically designed to fact-check your facts and help you in your studies. "
    "I will critically evaluate the information you provide and assist you in verifying its accuracy."
)

# Chatbot response function
def respond(
    message,
    history: list[dict], 
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [{"role": "system", "content": system_message}]  # System message

    # Add the conversation history to the prompt
    for val in history:
        if val["role"] == "user":
            messages.append({"role": "user", "content": val["content"]})
        elif val["role"] == "assistant":
            messages.append({"role": "assistant", "content": val["content"]})

    # Add the latest user message
    messages.append({"role": "user", "content": message})

    response = ""

    # Generate response from the model
    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

# Fact checker interface
fact_checker_interface = gr.Interface(
    fn=fact_checker,
    inputs=[
        gr.Textbox(label="Enter the fact to be checked"),
        gr.File(label="Upload additional documents")
    ],
    outputs=[
        gr.Textbox(label="Verdict"),
        gr.Textbox(label="Similarity Level"),
        gr.Textbox(label="References")
    ],
    title="Closed Domain Fact Checker"
)

# Chatbot interface
chatbot_interface = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(
            value=system_intro_message, 
            label="System message (Introduction)", 
            lines=3, 
            interactive=True
        ),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)"
        ),
    ],
    title="AI Fact-Checking Chatbot"
)



def fact_checking_process(fact, additional_documents=None):
    if additional_documents:
        additional_documents_json = pdf_to_json(additional_documents)
    else:
        additional_documents_json = None
    
    # Call the fact checker with the fact, documents, and llama_model
    verdict, similarity, references = check_fact_with_model(fact, additional_documents_json, llama_model)
    return f"Verdict: {verdict}, References: {references}", similarity

# Gradio app setup
with gr.Blocks() as demo:
    llama_model = load_llama_model()  # Load model once when the app starts
    
    fact_input = gr.Textbox(label="Enter the fact to be checked")
    result = gr.Textbox(label="Fact-checking result")
    similarity_score = gr.Textbox(label="Similarity Score")
    
    # Button to trigger fact-checking
    check_button = gr.Button("Check Fact")
    
    # Connect the button with the function
    check_button.click(fn=fact_checking_process, inputs=fact_input, outputs=[result, similarity_score])



# Combine both interfaces into a tabbed Gradio app
app = gr.TabbedInterface(
    [fact_checker_interface, chatbot_interface],
    ["Fact Checker", "Chatbot"]
)

if __name__ == "__main__":
    app.launch()
