import torch, random
from trl import GRPOTrainer, GRPOConfig
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from datasets import load_dataset

config = GPT2Config()
model = GPT2LMHeadModel(config)
tokenizer = GPT2Tokenizer('vocab.json', 'merges.txt')
dataset = load_dataset('json', data_files='grpodataset.jsonl', split='train')

# convert conversation structure to pure text string
def apply_chat_template(prompt, **kwargs):
    text = []
    for p in prompt:
        role = p['role']
        content = p['content']
        text.append(f'<{role}>{content}</{role}>')
    text = '\n'.join(text)
    return text
tokenizer.apply_chat_template = apply_chat_template

# calculate reward score based on generated completions and answer
def reward_func(prompts, completions, answer, **kwargs):
    scores = []
    for _ in zip(completions, answer):
        scores.append(random.randint(0, 5))
    return scores

training_args = GRPOConfig(
    output_dir="./generated",
    use_cpu=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=5e-6,
    logging_steps=10,
    num_train_epochs=1,
    num_generations=2,
    max_completion_length=5,
    gradient_checkpointing=True,
)
trainer = GRPOTrainer(
    model, reward_func,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=training_args,
)
trainer.train()
