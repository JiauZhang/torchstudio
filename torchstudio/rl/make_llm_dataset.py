import random
from conippets import jsonl

def random_text():
    n = random.randint(2, 20)
    text = ''
    for i in range(n):
        text += chr(ord('a') + random.randint(0, 25))
    return text

dataset = []
for i in range(100):
    item = {
        'prompt': [
            {'role': 'system', 'content': random_text()},
            {'role': 'user', 'content': random_text()},
        ],
        'answer': random_text(),
    }
    dataset.append(item)

jsonl.write('grpodataset.jsonl', dataset)