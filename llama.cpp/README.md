# Quantizing a Llama 3.1 8B parameter model and serving it with llama.cpp on google colab

## Part 1 - Quantizing Your Fine-Tuned Model

1. Push your fine-tuned Llama model to the Hugging Face Hub.
2. Navigate to the [ggml-org/gguf-my-repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo) page.
3. Sign in with your Hugging Face account.
4. Search for your model ID.
5. Select the desired quantization type.
6. Click on "Submit" and wait for the space to generate your quantized model!

## Part 2 - Serving the model on google colab using llama.cpp

1. Check out [Quantize_and_Serve.ipynb](Quantize_and_Serve.ipynb)