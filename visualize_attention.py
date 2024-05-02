import os
from typing import List, Optional

import fire
import seaborn as sns
import torch
from llama3.llama import Llama
from matplotlib import pyplot as plt

CKPT_DIR = "Meta-Llama-3-8B"
TOKENIZER_PATH = "Meta-Llama-3-8B/tokenizer.json"
MAX_SEQ_LEN = 128
MAX_BATCH_SIZE = 1


def main():
    output_folder = "attention_diagonal"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    generator = Llama.build(
        ckpt_dir=CKPT_DIR,
        tokenizer_path=TOKENIZER_PATH,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
    )

    prompt = "sentence: based on a true and historically significant story\nthe answer is positive\nsentence:contains very few laughs and even less surprises\nthe answer is negative\nsentence: generous and subversive artworks\nthe answer is "
    tokens = generator.tokenizer.encode(
        prompt,
        bos=True,
        eos=True,
    )
    tokens = torch.tensor([tokens], dtype=torch.long, device="cuda")

    _ = generator.model.forward(tokens, start_pos=0)
    for i, transformer_block in enumerate(generator.model.layers):
        print(transformer_block.attention.cache_k.shape)
        k_value = transformer_block.attention.k_value
        q_value = transformer_block.attention.q_value
        _, num_head, n, _ = q_value.shape
        print(i)
        for h in range(num_head):
            q = q_value[0, h, :, :]
            k = k_value[0, h, -n:, :]
            mask = torch.tril(torch.ones(n, n), diagonal=0).bool()
            A = torch.matmul(q, k.T)
            # Normalize the attention matrix
            A = (A - A.mean()) / torch.max(torch.abs(A))
            A = mask * A
            # Plotting the heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                A.cpu().detach().numpy(),
                annot=False,
                cmap="viridis",
                cbar=True,
                square=True,
            )
            plt.title("Heatmap of M * Q * K^T")
            plt.xlabel("Target Position")
            plt.ylabel("Source Position")
            plt.savefig(f"{output_folder}/attention_{i}_{h}.png")
            break


if __name__ == "__main__":
    fire.Fire(main)
