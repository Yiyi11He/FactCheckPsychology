import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

instruction_prompt = (
    "Please correct the following text based on the common writing mistakes listed below for assignments and theses:"
    "1. Jargon and technical terms: Using specialized language that the audience may not understand, making the writing difficult to follow and possibly alienating readers. "
    "2. Inappropriate tone: Writing in a tone that is too formal or informal, which can be off-putting for the audience."
    "3. Overgeneralization: Making sweeping statements without providing sufficient evidence or support. "
    "4. Lack of transitions: Failing to connect ideas and paragraphs smoothly, resulting in disjointed and difficult-to-follow writing. "
    "5. Passive voice: Using passive constructions excessively, making the writing less engaging and harder to understand."
    "6. Lack of concrete examples: Failing to provide real-world examples or anecdotes to help illustrate abstract concepts. "
    "7. Inadequate or unclear introductions and conclusions: Failing to provide a clear roadmap for the reader, making it difficult to understand the main points and purpose of the writing. "
    "8. Inaccurate or incomplete citations: Failing to properly credit sources or provide complete citation information, leading to plagiarism or an appearance of intellectual dishonesty. "
    "9. Wordiness: Using more words than necessary to convey a point, making the writing less concise and harder to read. "
    "10. Assuming reader knowledge: Writing as if the reader already knows the topic, leading to insufficient description of key concepts and potential confusion. "
    "11. Awkward and difficult-to-follow sentences: Crafting sentences that are hard to read or understand, often due to poor syntax or grammar. "
    "12. Poor use of punctuation: Misusing commas, semicolons, and other punctuation marks, resulting in unclear or confusing sentences. "
    "13. Long, complex sentences: Writing sentences that are overly long and convoluted, making them difficult for readers to follow. "
    "14. Circular logic: Presenting arguments that assume the conclusion is true, rather than providing adequate evidence to support the claim. "
    "15. Irrelevant details from research: Including unnecessary or unrelated information from research, which can distract from the main points and weaken the overall argument. "
    "16. Disorganized paragraphs: Failing to structure paragraphs logically, making it difficult for readers to follow the flow of ideas. "
    "17. Inconsistent tense: Shifting between past, present, and future tense within a text, which can be confusing for the reader. "
    "18. Incorrect word choice: Using words that are inappropriate or do not accurately convey the intended meaning. "
    "19. Redundancy: Repeating information or ideas unnecessarily, which can make the writing less concise and engaging. "
    "20. Misuse of homophones: Confusing words that sound alike but have different meanings, such as 'their,' 'there,' and 'they're.' "
    "21. Ambiguity: Writing in a way that is open to multiple interpretations or lacks clarity, making it difficult for the reader to understand the intended meaning. "
    "22. Lack of balance: Providing too much detail or focus on one aspect of the topic while neglecting others, which can result in an unbalanced or biased presentation of information. "
    "23. Plagiarism: Using someone else's words, ideas, or work without proper attribution, which is a serious academic and ethical violation. "
    "Here is the text:"

)

# Define the response function with fixed parameters
def respond(
    message,
    history: list[tuple[str, str]],
):
    # Fixed parameters
    system_message = "You are a critical and helpful professor, providing writing feedback to your students. Please keep your advise original relating to the text given, but watching closely for the instruction_prompt given. "
    max_tokens = 512
    temperature = 0.5
    top_p = 0.85

    # Combine the instruction prompt with the user's message
    full_message = instruction_prompt + " " + message

    # Create message history
    messages = [{"role": "system", "content": system_message}]
    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    # Add the full_message to messages as the user's input
    messages.append({"role": "user", "content": full_message})

    # Get response
    response = ""
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

# Define Gradio ChatInterface without additional inputs
demo = gr.ChatInterface(
    fn=respond, 
    title="Academic Writing Correction ✍🏻",
    description="Enter a passage for correction. The chatbot will focus on providing you feedback with the common writing issues such as grammar, jargon, tone, and structure."
)

if __name__ == "__main__":
    demo.launch()