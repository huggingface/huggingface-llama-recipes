# Hugging Face Llama Recipes

🤗🦙Welcome! This repository contains minimal recipes to quickly get started with Llama 3.1. 

* To get an overview of Llama 3.1, please visit [Hugging Face announcement blog post](https://huggingface.co/blog/llama31).
* For more advanced end-to-end use cases with open ML, please visit the [Open Source AI Cookbook](https://huggingface.co/learn/cookbook/index).

This repository is WIP, so you might see considerable changes in the coming days.

## Local Inference

Do you want to run inference of the Llama 3.1 models locally? So do we! The memory requirements depend on the model size and the precision of the weights. Here's a table showing the approximate memory needed for different configurations:

<table>
  <tr>
   <td><strong>Model Size</strong>
   </td>
   <td><strong>FP16</strong>
   </td>
   <td><strong>FP8</strong>
   </td>
   <td><strong>INT4 (AWQ/GPTQ)</strong>
   </td>
  </tr>
  <tr>
   <td>8B
   </td>
   <td>16 GB
   </td>
   <td>8 GB
   </td>
   <td>4 GB
   </td>
  </tr>
  <tr>
   <td>70B
   </td>
   <td>140 GB
   </td>
   <td>70 GB
   </td>
   <td>35 GB
   </td>
  </tr>
  <tr>
   <td>405B
   </td>
   <td>810 GB
   </td>
   <td>405 GB
   </td>
   <td>203 GB
   </td>
  </tr>
</table>

_Note: These are estimated values and may vary based on specific implementation details and optimizations._

Here are some notebooks to help you started:

* Run Llama 8B in free Google Colab in half precision
* Run Llama 8B in 8-bit and 4-bits!
* Run Llama with AWQ
* Execute some Llama-generated Python code
* Use tools with Llama!

## API inference

Are these models too large for you to run at home? Do you want to experiment with Llama 405B? Try out the following examples!

* [Use the Inference API for PRO users](inference-api.ipynb)
* Use a dedicated Inference Endpoint

## Llama Guard and Prompt Guard

In addition to the generative models, Meta released two new models: Llama Guard 3 and Prompt Guard. Prompt Guard is a small classifier that detects prompt injections and jailbreaks. Llama Guard 3 is a safeguard model that can classify LLM inputs and generations. Learn how to use them as done in the following notebooks:

* Detecting jailbreaking with Prompt Guard
* Using Llama Guard for Guardrailing

## Advanced use cases

* How to fine-tune Llama 3.1 8B on consumer GPU with `trl` and QLoRA
* Generate synthetic data with `distilabel`
* Do assisted decoding with a large and a small model
* Build a ML demo using Gradio
