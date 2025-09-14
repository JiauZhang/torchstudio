import random
import numpy as np
from conippets import json
from PIL.Image import fromarray
from trl import GRPOTrainer, GRPOConfig
from transformers.models.idefics3.modeling_idefics3 import Idefics3ForConditionalGeneration
from transformers.models.idefics3.configuration_idefics3 import Idefics3Config
from transformers.models.idefics3.image_processing_idefics3 import Idefics3ImageProcessor
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers.models.idefics3.processing_idefics3 import Idefics3Processor
from datasets import load_dataset, Features, Image

config = Idefics3Config.from_dict(json.read('config.json'))
config.scale_factor = 1

model = Idefics3ForConditionalGeneration(config)
image_processor = Idefics3ImageProcessor()
tokenizer = GPT2Tokenizer('vocab.json', 'merges.txt')
processor = Idefics3Processor(image_processor, tokenizer)

def make_random_image(example):
    size = config.vision_config.image_size
    random_data = np.random.randint(0, 256, size=(size, size, 3), dtype=np.uint8)
    pil_image = fromarray(random_data)
    example['image'] = pil_image
    return example
dataset = load_dataset('json', data_files='grpo-vlm-dataset.jsonl', split='train')
dataset = dataset.map(make_random_image)
dataset = dataset.cast_column('image', Image(decode=True))

# convert conversation structure
# {
#     "role": "user",
#     "content": [
#         {"type": "image"},
#         {"type": "text", "text": "YOUR_QUESTION"}
#     ]
# }
# to pure text string: <image>YOUR_QUESTION
def apply_chat_template(prompt, **kwargs):
    text = []
    for p in prompt:
        role = p['role']
        content = p['content']
        text.append('')
        for c in content:
            if c['type'] == 'image':
                text[-1] += processor.image_token
            elif c['type'] == 'text':
                text[-1] += c['text']
        text[-1] = f'<|{role}|>\n{text[-1]}'
    text = '\n'.join(text)
    return text
processor.apply_chat_template = apply_chat_template

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
    max_prompt_length=None,
)
trainer = GRPOTrainer(
    model, reward_func,
    train_dataset=dataset,
    processing_class=processor,
    args=training_args,
)
trainer.train()
