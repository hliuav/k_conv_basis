# k conv basis Implementation and Visualization

## Prerequisites
1. Please refer to the README.md in the llama3 directory for the prerequisites.
2. Clone text prompt we use https://github.com/gkamradt/LLMTest_NeedleInAHaystack 
```bash
git clone https://github.com/gkamradt/LLMTest_NeedleInAHaystack
```
replace the variable `text_root_dir` in `save_attention_matrix_from_gpt2.py` with the path to the cloned directory.

## K conv basis FFT unit test and visualization
Please run the following command to run the unit test for k conv basis computation and visualization:
```bash
python unit_test.py
```

## Visualization of attention matrix from Meta-Llama-3-8B
Please run the following command to visualize the attention from Meta-Llama-3-8B:
```bash
torchrun --nproc_per_node 1 visualize_attention_llama3.py
```

## Save attention matrix from GPT2 and run comparison test between naive attention and k conv basis
To save (visualize) the attention from GPT2, first install huggingface transformers and replace the file modeling_gpt2.py to the path `transformers/src/transformers/models/gpt2/modeling_gpt2.py`. Then run the following command:
```bash
cp modeling_gpt2.py {path_to_transformers_lib}/models/gpt2/modeling_gpt2.py
```
Then run the following command to save the attention matrix from GPT2:
```bash
python visualize_attention_gpt2.py
```
Then compute and save result for different k values:
```bash
zsh run_k_conv_basis.sh
```

Finally, to visualize the result, run the following command:
```bash
python plot_k_conv_basis.py
```

