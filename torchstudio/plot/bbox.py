from PIL import Image, ImageDraw, ImageFont

def draw_bbox(image, bbox, label=None, score=None, font=None, font_size=12, color='green'):
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline=color, width=2)

    if label:
        font = ImageFont.truetype(font, size=font_size) if font else ImageFont.load_default(size=font_size) 
        label_text = f"{label}: {score:.2f}" if score else label
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # text background
        x_min, y_min, x_max, y_max = bbox
        draw.rectangle(
            [x_min, y_min - text_height - 4, x_min + text_width + 4, y_min],
            fill=color
        )
        draw.text((x_min + 2, y_min - text_height - 2), label_text, fill="white", font=font)

if __name__ == "__main__":
    import numpy as np

    image = np.random.randint(0, 255, size=(512, 512, 3)).astype(np.uint8)
    image = Image.fromarray(image)
    draw_bbox(image, (100, 50, 300, 250), label='label', score=0.99, color='green')
    image.show()
