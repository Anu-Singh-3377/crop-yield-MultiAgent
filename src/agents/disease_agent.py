from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "crop_disease_model.pt"


def conv_block(in_channels: int, out_channels: int, pool: bool = False) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class CNNNeuralNet(nn.Module):
    """Architecture used by plant-diseases-detection-pytorch.ipynb."""

    def __init__(self, in_channels: int, num_diseases: int) -> None:
        super().__init__()
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(
            conv_block(128, 128),
            conv_block(128, 128),
        )
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(
            conv_block(512, 512),
            conv_block(512, 512),
        )
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_diseases),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        return self.classifier(out)


class DiseasePredictionRequest(BaseModel):
    crop: Optional[str] = Field(default=None, min_length=1)


class DiseasePredictionResponse(BaseModel):
    crop: str
    disease: str
    confidence: float
    risk_level: str


class DiseaseAgent:
    """Independent crop disease image inference agent."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None) -> None:
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self.model = None
        self.class_names = []
        self.device = None
        self.transform = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Disease model artifact not found: {self.model_path}. "
                "Train it with scripts/train_disease_model.py."
            )

        try:
            from torchvision import transforms
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Disease dependencies are missing. Install torch, torchvision, and pillow."
            ) from exc

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        artifact = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False,
        )
        self.class_names = list(artifact["class_names"])
        if not self.class_names:
            raise ValueError("Disease model artifact has no class names")

        model = CNNNeuralNet(3, len(self.class_names))
        model.load_state_dict(artifact["model_state"])
        model.eval()
        model.to(self.device)
        self.model = model

        image_size = int(artifact.get("image_size", 256))
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def _risk_level(self, disease: str, confidence: float) -> str:
        if disease.lower() in {"healthy", "normal"}:
            return "low"
        if confidence >= 0.85:
            return "high"
        if confidence >= 0.60:
            return "medium"
        return "low"

    def _split_crop_and_disease(self, class_name: str) -> tuple[str, str]:
        if "___" in class_name:
            crop, disease = class_name.split("___", 1)
        elif "__" in class_name:
            crop, disease = class_name.split("__", 1)
        else:
            crop, disease = "Unknown", class_name
        return crop.replace("_", " ").strip() or "Unknown", disease.replace("_", " ").strip()

    def predict(
        self,
        image: Union[str, Path, bytes, BytesIO, Any],
        crop_hint: Optional[str] = None,
    ) -> DiseasePredictionResponse:
        if self.model is None:
            self.load()

        try:
            import torch
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Disease dependencies are missing. Install torch, torchvision, and pillow."
            ) from exc

        if isinstance(image, (bytes, bytearray)):
            source = BytesIO(bytes(image))
        elif isinstance(image, BytesIO):
            source = image
        elif hasattr(image, "read"):
            source = image
        else:
            source = Path(image)

        if isinstance(source, Path):
            pil_image = Image.open(source).convert("RGB")
        else:
            pil_image = Image.open(source).convert("RGB")

        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence, predicted_index = torch.max(probabilities, dim=0)

        class_name = self.class_names[int(predicted_index.item())]
        dataset_crop, disease = self._split_crop_and_disease(class_name)
        crop = crop_hint.strip() if crop_hint else dataset_crop
        confidence_value = round(float(confidence.item()), 6)

        return DiseasePredictionResponse(
            crop=crop,
            disease=disease,
            confidence=confidence_value,
            risk_level=self._risk_level(disease, confidence_value),
        )
