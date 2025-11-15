from typing import Optional

import cv2
import numpy as np
import torch


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer_name: str) -> None:
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.hooks()

    def hooks(self) -> None:
        def backward_hook(
            module: torch.nn.Module, grad_in: torch.Tensor, grad_out: torch.Tensor
        ) -> None:
            self.gradients = grad_out[0]

        def forward_hook(
            module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor
        ) -> None:
            self.activations = output

        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        logit = self.model(input_tensor)
        score = logit[:, target_class]
        self.model.zero_grad()
        score.backward()

        if self.gradients is None:
            raise ValueError("Gradients are None. Backward hook did not run.")
        if self.activations is None:
            raise ValueError("Activations are None. Forward hook did not run.")

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam
