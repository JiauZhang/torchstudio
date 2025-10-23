import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

def draw_context(processor, messages, margin=10, font_size=12):
    image = np.empty((512, 512, 3), dtype=np.uint8)
    image.fill(255)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    font = ImageFont.load_default(size=font_size)
    draw.text((margin, margin), text=text, fill="black", font=font)
    return pil_image

model_id = "Qwen/Qwen3-VL-4B-Thinking"
model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'Hi.'}
        ]
    }
]
context = draw_context(processor, messages)

system_instruction = '''
You are an helpful AI assistant, and all your chat histories with users are recorded in images.
You need to respond to users based on the information in the images.
'''
messages = [
    {
        'role': 'system',
        'content': [
            {
                'type': 'text',
                'text': system_instruction,
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": context,
            },
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)