# Vendored from DeQA-Score/src/evaluate/scorer.py with relative imports.
# Inference-only wrapper: runs the LLM forward with the prompt
#   "USER: How would you rate the quality of this image?\n<|image|>\nASSISTANT: The quality of the image is"
# pulls logits at the last position over the 5 word ids of
#   ["excellent", "good", "fair", "poor", "bad"]
# and returns softmax @ [5, 4, 3, 2, 1] — a continuous MOS-style score in
# roughly [1, 5], higher is better.
from typing import List

import torch
import torch.nn as nn
from PIL import Image

from .builder import load_pretrained_model
from .constants import (  # noqa: F401  (DEFAULT_IMAGE_TOKEN re-exported for callers)
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from .mm_utils import get_model_name_from_path, tokenizer_image_token  # noqa: F401


class Scorer(nn.Module):
    def __init__(self, pretrained: str = "zhiyuanyou/DeQA-Score-Mix3", device: str = "cuda:0"):
        super().__init__()
        tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, "mplug_owl2", device=device)
        prompt = (
            "USER: How would you rate the quality of this image?\n<|image|>\nASSISTANT: The quality of the image is"
        )

        self.preferential_ids_ = [
            id_[1] for id_ in tokenizer(["excellent", "good", "fair", "poor", "bad"])["input_ids"]
        ]
        self.weight_tensor = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0]).half().to(model.device)

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(model.device)
        )

    def expand2square(self, pil_img: Image.Image, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def forward(self, image: List[Image.Image]) -> torch.Tensor:
        bg = tuple(int(x * 255) for x in self.image_processor.image_mean)
        image = [self.expand2square(img, bg) for img in image]
        with torch.inference_mode():
            image_tensor = (
                self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().to(self.model.device)
            )
            output_logits = self.model(
                input_ids=self.input_ids.repeat(image_tensor.shape[0], 1),
                images=image_tensor,
            )["logits"][:, -1, self.preferential_ids_]

            return torch.softmax(output_logits, -1) @ self.weight_tensor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="zhiyuanyou/DeQA-Score-Mix3")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--img_path", type=str, required=True)
    args = parser.parse_args()

    scorer = Scorer(pretrained=args.model_path, device=args.device)
    print(scorer([Image.open(args.img_path)]).tolist())
