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
    history,
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    # Load the Llama model via Gradio (using gr.load)
    llama_model = load_llama_model()

    # Create a conversation history
    messages = [{"role": "system", "content": system_message}]  # System message

    # Add the conversation history to the prompt
    for val in history:
        if val["role"] == "user":
            messages.append({"role": "user", "content": val["content"]})
        elif val["role"] == "assistant":
            messages.append({"role": "assistant", "content": val["content"]})

    # Add the latest user message
    messages.append({"role": "user", "content": message})

    # Generate the prompt to send to the model
    conversation_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

    outputs = llama_model(conversation_text, max_tokens=max_tokens)

    response = outputs['text'] if 'text' in outputs else "No response"

    return response

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