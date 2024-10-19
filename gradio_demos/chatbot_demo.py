from huggingface_hub import login
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Log in to Hugging Face Hub
login()

# Determine the device to use (GPU if available, otherwise CPU)
device = 0 if torch.cuda.is_available() else -1

# Dictionary mapping model names to their Hugging Face Hub identifiers
llama_models = {
    "Llama 3 70B Instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
    "Llama 3 8B Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Llama 3.1 70B Instruct": "meta-llama/Llama-3.1-70B-Instruct",
    "Llama 3.1 8B Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama 3.2 3B Instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "Llama 3.2 1B Instruct": "meta-llama/Llama-3.2-1B-Instruct",
}

# Function to load the model and tokenizer
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
    return generator

# Cache to store loaded models
model_cache = {}

# Function to generate chat responses
def generate_chat(user_input, history, model_choice):
    # Load the model if not already cached
    if model_choice not in model_cache:
        model_cache[model_choice] = load_model(llama_models[model_choice])
    generator = model_cache[model_choice]

    # Initial system prompt
    system_prompt = {"role": "system", "content": "You are a helpful assistant"}

    # Initialize history if it's None
    if history is None:
        history = [system_prompt]
    
    # Append user input to history
    history.append({"role": "user", "content": user_input})

    # Generate response using the model
    response = generator(
        history,
        max_length=512,
        pad_token_id=generator.tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )[-1]["generated_text"][-1]["content"]

    # Append model response to history
    history.append({"role": "assistant", "content": response})
    
    return history

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("<h1><center>Chat with Llama Models</center></h1>")

    # Dropdown to select model
    model_choice = gr.Dropdown(list(llama_models.keys()), label="Select Llama Model")
    # Chatbot interface
    chatbot = gr.Chatbot(label="Chatbot Interface", type="messages")
    # Textbox for user input
    txt_input = gr.Textbox(show_label=False, placeholder="Type your message here...")

    # Function to handle user input and generate response
    def respond(user_input, chat_history, model_choice):
        if model_choice is None:
            model_choice = list(llama_models.keys())[0]
        updated_history = generate_chat(user_input, chat_history, model_choice)
        return "", updated_history

    # Submit user input on pressing Enter
    txt_input.submit(respond, [txt_input, chatbot, model_choice], [txt_input, chatbot])
    # Button to submit user input
    submit_btn = gr.Button("Submit")
    submit_btn.click(respond, [txt_input, chatbot, model_choice], [txt_input, chatbot])

# Launch the Gradio demo
demo.launch()
