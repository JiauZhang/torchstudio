from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, TextStreamer
import os, pymupdf
from torchstudio.plot.text import draw_text

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

def draw_context(processor, messages):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    pil_image = draw_text(text)
    return pil_image

model_id = "Qwen/Qwen3-VL-4B-Thinking"
model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)
streamer = TextStreamer(processor.tokenizer, skip_prompt=True)
context = [
    {'role': 'system', 'content': [{'type': 'text', 'text': 'You are a helpful AI assistant'}]},
    {"role": "user", "content": []},
]

text_messages = []
doc_texts, pil_images = doc_images(num_pages=5)
questions = [
    'What is the core point of this paper?',
    'Your output is too long, make it shorter',
]
llm_messages = [{'role': 'user', 'content': [{'type': 'text', 'text': '\n'.join(doc_texts)}]}]

for idx, question in enumerate(questions):
    text_messages.append({'role': 'user', 'content': [{'type': 'text', 'text': question}]})
    llm_messages.append(text_messages[-1])
    pixel_messages = draw_context(processor, text_messages)
    pixel_messages.save(f'pixel_messages_{idx:02d}.png')
    context[-1]['content'] = [
        {'type': 'text', 'text': 'chat history as below:\n'},
        {'type': 'image', 'image': pixel_messages},
        {'type': 'text', 'text': 'paper content as below:\n'},
    ]
    for pil_image in pil_images:
        context[-1]['content'].append({
            'type': 'image',
            'image': pil_image,
        })

    inputs = processor.apply_chat_template(
        context,
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
    )[0].split('</think>')[-1].lstrip()
    text_messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': output_text}]})
    llm_messages.append(text_messages[-1])

pixel_messages = draw_context(processor, text_messages)
pixel_messages.save(f'pixel_messages_{len(questions):02d}.png')

llm_messages.pop()
llm_tokens = processor.apply_chat_template(
    llm_messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
).input_ids
pixel_tokens = processor.apply_chat_template(
    context, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
).input_ids
ratio = llm_tokens.numel() / pixel_tokens.numel()
print(f'compression ratio: {ratio:.2f}')