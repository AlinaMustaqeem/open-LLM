# Zero Shot Inference with Transformers
This repository contains code that demonstrates the use of the Hugging Face Transformers library for zero-shot text classification and text generation. It includes examples with BART for classification and GPT-2 for text generation.
## Contents

`ZeroShotInference` Class: A class to handle zero-shot text classification and text generation.
`main.py`: Example usage of the `ZeroShotInference` class for text classification and text generation.
## Installation
To run this code, you need to install the Hugging Face Transformers library and its dependencies. You can install them with:
```bash
pip install transformers torch
```
## Usage
### Zero-shot Text Classification
This example uses the Facebook BART large model (BART-Large-MNLI) to classify a text based on given candidate labels.
```python
from transformers import pipeline
class ZeroShotInference:
    def __init__(self, task, pre_trained_model):
        self.model = pre_trained_model
        self.task = task
        self.pipe = pipeline(task=self.task, model=self.model)
    def classify_text(self, text, candidate_labels):
        result = self.pipe(text, candidate_labels)
        return result
```
Example usage:
```python
text = "I love to play video games"
candidate_labels = ['hobby', 'habit', 'adventure', 'boredom', 'business']
task = "zero-shot-classification"
model = "facebook/bart-large-mnli"
zero_short_inference = ZeroShotInference(task, model)
result = zero_short_inference.classify_text(text, candidate_labels)
print(result)  # Output: {'sequence': 'I love to play video games', 'labels': ['hobby', 'habit', 'adventure', 'business', 'boredom'], 'scores': [0.8799885511398315, 0.09845343977212906, 0.016700521111488342, 0.0031407771166414022, 0.0017165272729471326]}
```
### Text Generation
This example uses GPT-2 to generate text based on a given prompt.
```python
def generate_text(self, prompt, max_length=100, temperature=0.7):
    output = self.pipe(prompt, max_length=max_length, do_sample=True, temperature=temperature, truncation=True)
    return output[0]['generated_text']
```
Example usage:
```python
prompt = "I was doing coding last night"
model_2 = "gpt2"
task_2 = "text-generation"
zero_shot_infernece_2 = ZeroShotInference(task_2, model_2)
result_2 = zero_shot_infernece_2.generate_text(prompt)
print(result_2)
```

### IN 2nd Task
## Quatization
The model uses quantization techniques for efficient inference, reducing memory usage and speeding up response times using `BitsAndBytesConfig`

### HF_TOKEN
for HF_token as mention i have generated the token from huggingFace and have added to secret of google colab to be used


### IN 3rd Task
This task  aims to fine-tune the Mistral model for generating instructional text using the 7B variant. It involves training the model on a specific dataset and then using it to generate instructional text based on given prompts.

## FineTunning
The Fine-Tuning Mistral Instruct 7B project leverages the Mistral model, a large-scale language model, for generating instructional text. By fine-tuning the model on a dataset containing instructional text, the goal is to create a specialized model capable of generating coherent and informative instructions.


## Contributions
Contributions to the project are welcome. Feel free to fork the repository, make your changes, and submit a pull request. You can also open issues to suggest improvements or report bugs.
