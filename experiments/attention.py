import os, argparse, torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import GPT2LMHeadModel, GPT2Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model-name', type=str, default='openai-community/gpt2')
parser.add_argument('--prompt', type=str, default='You are a helpful assistant, I can')
args = parser.parse_args()

model_name = args.model_name
config = GPT2Config.from_json_file(os.path.join(model_name, 'config.json'))
config._attn_implementation = 'eager'
model = GPT2LMHeadModel.from_pretrained(model_name, config=config, weights_only=True)
tokenizer = GPT2Tokenizer.from_pretrained(model_name, config=config)
tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer(args.prompt, return_tensors="pt", padding=True, truncation=True)
outputs = model.generate(
    **inputs,
    max_length=128,
    num_return_sequences=1,
    temperature=0.8,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.2,
    use_cache=True,
)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

with torch.no_grad():
    inputs = tokenizer(generated_text, return_tensors="pt")
    outputs = model(**inputs, output_attentions=True, use_cache=False)
    attentions = outputs.attentions
cm = plt.get_cmap('viridis')
norm = LogNorm(vmin=7e-4, vmax=1.0)
for i, attn in enumerate(attentions):
    with torch.no_grad():
        # batch_size, num_heads, seq_len, seq_len
        attn = attn.mean(dim=1).squeeze().detach().numpy()
    seq_len = attn.shape[-1]
    step = seq_len // 5
    plt.imshow(attn, cmap=cm, norm=norm)
    cbar = plt.colorbar()
    cbar.set_label('attention score')
    plt.xticks(np.arange(0, seq_len, step))
    plt.yticks(np.arange(0, seq_len, step))
    plt.title(f'layer-{i:02d}')
    plt.xlabel('tokens')
    plt.ylabel("tokens")
    plt.savefig(f'attn-score-{i:02d}.png', bbox_inches='tight', dpi=100)
    plt.close()