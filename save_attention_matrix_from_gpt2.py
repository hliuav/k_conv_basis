import os

from transformers import GPT2Model, AutoTokenizer

import torch

import seaborn as sns
import matplotlib.pyplot as plt

#from k_conv_basis_utils import exact_attention_matrix_by_conv, exact_attention_matrix

#n = 80000
#max_load = torch.ones(n, n, dtype=torch.float32).to('cuda')

def save_qkv_values(inputs):

    # Save qkv value
    token_size = len(inputs[0])
    print(token_size)
    model = GPT2Model.from_pretrained("openai-community/gpt2")
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    model.to('cuda')
    outputs = model(**inputs)
    
def main():
    # Read txt as input prompt text
    text_root_dir = '/home/heshanliu/LLMTest_NeedleInAHaystack/needlehaystack/PaulGrahamEssays'
    with open(f'{text_root_dir}/addiction.txt', 'r') as file:
        input_prompt = file.read().replace('\n', '')
    with open(f'{text_root_dir}/gh.txt', 'r') as file:
        input_prompt += file.read().replace('\n', '')
    with open(f'{text_root_dir}/vb.txt', 'r') as file:
        input_prompt += file.read().replace('\n', '')
    with open(f'{text_root_dir}/vw.txt', 'r') as file:
        input_prompt += file.read().replace('\n', '')
    with open(f'{text_root_dir}/gba.txt', 'r') as file:
        input_prompt += file.read().replace('\n', '')
    with open(f'{text_root_dir}/gap.txt', 'r') as file:
        input_prompt += file.read().replace('\n', '')
    with open(f'{text_root_dir}/gap.txt', 'r') as file:
        input_prompt += file.read().replace('\n', '')
    with open(f'{text_root_dir}/gap.txt', 'r') as file:
        input_prompt += file.read().replace('\n', '')
    
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    inputs = tokenizer(input_prompt, return_tensors="pt")
    inputs[0] = inputs[0]
    #prompt_lengths = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
    prompt_lengths = [32000]
    for i in prompt_lengths:
        # if there is a error in the prompt, skip it and continue to run the next one
        try:
            inputs = tokenizer(input_prompt, return_tensors="pt")
            inputs['input_ids'] = inputs['input_ids'][:, :i]
            inputs['attention_mask'] = inputs['attention_mask'][:, :i]
            save_qkv_values(inputs)
        except:
            print(f"Error in prompt length {i}")
            continue
if __name__ == '__main__':
    main()

