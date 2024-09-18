import gradio as gr
from fact_checker import fact_checker
from huggingface_hub import InferenceClient

# Initialize the Hugging Face inference client for the chatbot
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

# Custom system message where the chatbot introduces itself and explains its role
system_intro_message = (
    "Hello, I am an AI chatbot specifically designed to fact-check your facts and help you in your studies. "
    "I will critically evaluate the information you provide and assist you in verifying its accuracy."
)

# Chatbot response function
def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [{"role": "system", "content": system_message}]

    # Add the conversation history to the prompt
    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

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
        gr.inputs.Textbox(label="Enter the fact to be checked"),
        gr.inputs.File(label="Upload additional documents (Optional)")
    ],
    outputs=[
        gr.outputs.Textbox(label="Verdict"),
        gr.outputs.Textbox(label="Similarity Level"),
        gr.outputs.Textbox(label="References")
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

# Combine both interfaces into a tabbed Gradio app
app = gr.TabbedInterface(
    [fact_checker_interface, chatbot_interface],
    ["Fact Checker", "Chatbot"]
)

if __name__ == "__main__":
    app.launch()
