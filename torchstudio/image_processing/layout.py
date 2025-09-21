import torch
from PIL import Image
from transformers import DFineForObjectDetection, RTDetrImageProcessor

class LayoutDetector:
    classes_map = {
        0: "Caption", 1: "Footnote", 2: "Formula", 3: "List-item", 4: "Page-footer", 5: "Page-header",
        6: "Picture", 7: "Section-header", 8: "Table", 9: "Text", 10: "Title", 11: "Document Index",
        12: "Code", 13: "Checkbox-Selected", 14: "Checkbox-Unselected", 15: "Form", 16: "Key-Value Region",
    }

    def __init__(self, model_path=None, threshold=0.6):
        self.threshold = threshold
        model_name = model_path if model_path else 'ds4sd/docling-layout-egret-large'
        self.image_processor = RTDetrImageProcessor.from_pretrained(model_name)
        self.model = DFineForObjectDetection.from_pretrained(model_name)

    @torch.no_grad()
    def __call__(self, image):
        inputs = self.image_processor(images=[image], return_tensors="pt")
        outputs = self.model(**inputs)
        result = self.image_processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=self.threshold,
        )[0]

        layout = []
        for score, label_id, bbox in zip(result["scores"], result["labels"], result["boxes"], strict=True):
            layout.append({'score': score.item(), 'label': self.classes_map[label_id.item()], 'bbox': bbox.tolist()})
        return layout

if __name__ == "__main__":
    import argparse
    from torchstudio.plot.bbox import draw_bbox

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--model-path', type=str, default=None)
    args = parser.parse_args()

    detector = LayoutDetector(model_path=args.model_path)
    image = Image.open(args.image)
    image = image.convert("RGB")

    layout = detector(image)
    for item in layout:
        draw_bbox(image, item['bbox'], label=item['label'], score=item['score'])
    image.show()
