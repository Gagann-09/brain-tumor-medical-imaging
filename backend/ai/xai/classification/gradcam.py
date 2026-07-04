import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional

from ai.xai.shared.base import BaseExplainer
from ai.xai.shared.registry import ExplainerRegistry

@ExplainerRegistry.register("grad_cam", ["classification", "hybrid_cnn"])
class GradCAM(BaseExplainer):
    def __init__(self, model, target_layer: str, device: Optional[torch.device] = None):
        super().__init__(model, device)
        self.target_layer = target_layer

    def generate_explanation(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()
        
        # Ensure input requires grad
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get intermediate features
        features_dict = self.model.get_intermediate_features()
        if self.target_layer not in features_dict:
            raise ValueError(f"Target layer {self.target_layer} not found in FeatureProvider.")
            
        feature_map = features_dict[self.target_layer].tensor
        feature_map.retain_grad()
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        score = output[0, target_class]
        score.backward(retain_graph=True)
        
        gradients = feature_map.grad
        
        # Grad-CAM: Global Average Pooling on gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of feature maps
        cam = torch.sum(weights * feature_map, dim=1, keepdim=True)
        
        # ReLU
        cam = F.relu(cam)
        
        # Interpolate to input size
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        
        # Normalize between 0 and 1
        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam

@ExplainerRegistry.register("grad_cam_plus_plus", ["classification", "hybrid_cnn"])
class GradCAMPlusPlus(BaseExplainer):
    def __init__(self, model, target_layer: str, device: Optional[torch.device] = None):
        super().__init__(model, device)
        self.target_layer = target_layer

    def generate_explanation(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()
        
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_()
        
        output = self.model(input_tensor)
        
        features_dict = self.model.get_intermediate_features()
        if self.target_layer not in features_dict:
            raise ValueError(f"Target layer {self.target_layer} not found in FeatureProvider.")
            
        feature_map = features_dict[self.target_layer].tensor
        feature_map.retain_grad()
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        score = output[0, target_class]
        
        # Need exp(score) for Grad-CAM++
        score = torch.exp(score)
        score.backward(retain_graph=True)
        
        gradients = feature_map.grad
        activations = feature_map
        
        b, k, u, v = gradients.size()
        
        # Alpha computation for Grad-CAM++
        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + activations.mul(gradients.pow(3)).sum(dim=(2, 3), keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alphas = alpha_num.div(alpha_denom)
        
        positive_gradients = F.relu(score.exp() * gradients)
        weights = (alphas * positive_gradients).sum(dim=(2, 3), keepdim=True)
        
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam
