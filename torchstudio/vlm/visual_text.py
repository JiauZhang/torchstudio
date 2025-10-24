import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, TextStreamer
import os, pymupdf

def doc_images(scale=1, num_pages=1):
    pdf_path = 'paper.pdf'
    if not os.path.exists(pdf_path):
        os.system(f'curl https://arxiv.org/pdf/2510.18234 -o {pdf_path}')

    mat = pymupdf.Matrix(scale, scale)
    doc = pymupdf.open('paper.pdf')
    pil_images = []
    doc_texts = []
    for idx, page in enumerate(doc.pages()):
        if idx > num_pages:
            break
        doc_texts.append(page.get_text())
        pil_image = page.get_pixmap(matrix=mat).pil_image()
        pil_images.append(pil_image)
    return doc_texts, pil_images

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
streamer = TextStreamer(processor.tokenizer, skip_prompt=True)

context = [
    {
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'What is the core point of this paper?'}
        ]
    }
]
context = draw_context(processor, context)

system_instruction = '''
You are an helpful AI assistant, and all the chat histories between you and the user are recorded in images.
You must respond to the user based the history information recorded in the images.
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
        "content": [],
    }
]

doc_texts, pil_images = doc_images(num_pages=2)
for pil_image in pil_images + [context]:
    messages[-1]['content'].append({
        'type': 'image',
        'image': pil_image,
    })

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, streamer=streamer, max_new_tokens=4096)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)