import os
from typing import List, Optional

import fire
import seaborn as sns
import torch
from llama3.llama.generation import Llama
from matplotlib import pyplot as plt

CKPT_DIR = "Meta-Llama-3-8B"
TOKENIZER_PATH = "Meta-Llama-3-8B/tokenizer.model"
MAX_SEQ_LEN = 128
MAX_BATCH_SIZE = 1
NUM_HEADS = 32
SAVE_BLOCK_INDEX = 25


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
    #with open('/home/heshanliu/LLMTest_NeedleInAHaystack/needlehaystack/PaulGrahamEssays/addiction.txt', 'r') as file:
    #    prompt = file.read().replace('\n', '')
    # repeat prompt 10 times
    tokens = generator.tokenizer.encode(
        prompt,
        bos=True,
        eos=True,
    )
    output = generator.tokenizer.decode(tokens)
    print(output)
    tokens = torch.tensor([tokens], dtype=torch.long, device="cuda")
    seq_len = tokens.shape[1]
    
    _ = generator.model.forward(tokens, start_pos=0)

    # Save the attention matrix for each layer
    cols = len(generator.model.layers)
    rows = NUM_HEADS
    mask_all = torch.tril(torch.ones(seq_len, seq_len), diagonal=0).bool()
    mask_all = mask_all.repeat(MAX_BATCH_SIZE, NUM_HEADS, 1, 1)
    plot_data = []
    for i, transformer_block in enumerate(generator.model.layers):
        k_value = transformer_block.attention.k_value
        q_value = transformer_block.attention.q_value
        v_value = transformer_block.attention.v_value
        batch_size, num_head, n, d = q_value.shape
        # Create a 4d mask to only show the lower triangular part of the matrix q_value
        A_all = torch.matmul(q_value, k_value.permute(0, 1, 3, 2))
        # Normalize the attention matrix along the last two dimensions
        mean_A1 = A_all.mean(dim=(-2, -1), keepdim=True)  # Correct usage
        max_abs_A1 = torch.max(torch.abs(A_all), dim=-1, keepdim=True)[0]
        max_abs_A1 = torch.max(max_abs_A1, dim=-2, keepdim=True)[0]
        A_all = (A_all - mean_A1) / max_abs_A1
        A_all = mask_all * A_all
        if i == SAVE_BLOCK_INDEX:
            print("save q k v value")
            torch.save(q_value, 'q_value.pth')
            torch.save(k_value, 'k_value.pth')
            torch.save(v_value, 'v_value.pth')
            A_all_cpu = A_all.cpu().detach().numpy()
            plot_data.append(A_all_cpu)

    # Plot the attention matrix for each layer and each head
    #fig, axes = plt.subplots(cols, rows, figsize=(cols * 5, rows * 5))
    plt.rcParams.update({'font.size': 55, 'legend.fontsize': 50})
    fig, ax = plt.subplots(figsize=(14, 14))
    for i, data in enumerate(plot_data):
        for h in [20]: #range(rows):
            #ax = axes[i, h]
            cax = ax.imshow(data[0, h, :, :], cmap="viridis")
            ax.set_title(f"Heatmap of Layer 25 Head {h}", fontweight='bold')
            ax.set_xlabel("Target Position", fontweight='bold')
            ax.set_ylabel("Source Position", fontweight='bold')
            # Add a color bar to each subplot
            plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"{output_folder}/best.pdf")
    #plt.savefig(f"{output_folder}/all_attention_block.png")


if __name__ == "__main__":
    fire.Fire(main)
