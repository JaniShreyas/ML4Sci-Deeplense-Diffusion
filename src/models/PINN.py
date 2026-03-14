import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedClassifier(nn.Module):
    def __init__(self, backbone_classifier, physics_config, image_size=32):
        super().__init__()
        
        channels = physics_config.get("estimator_channels", [8, 16])
        in_ch = physics_config.get("in_channels", 1)
        
        layers = []
        for out_ch in channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            in_ch = out_ch 
            
        layers.append(nn.Flatten())
        self.feature_extractor = nn.Sequential(*layers)
        
        dummy_input = torch.zeros(1, physics_config.get("in_channels", 1), image_size, image_size)
        flattened_size = self.feature_extractor(dummy_input).shape[1]
        
        self.head = nn.Linear(flattened_size, 3)
        self.param_estimator = nn.Sequential(self.feature_extractor, self.head)
        
        self.classifier = backbone_classifier 
        
    def _create_physics_grid(self, theta_E, x_c, y_c, B, H, W, device):
        """ The Gravitational Lensing Equation: beta = theta - alpha """
        y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
        theta_grid = torch.stack([x, y], dim=-1).to(device)
        theta_grid = theta_grid.unsqueeze(0).repeat(B, 1, 1, 1)
        
        centers = torch.stack([x_c, y_c], dim=-1).view(B, 1, 1, 2)
        
        delta_theta = theta_grid - centers
        r = torch.norm(delta_theta, dim=-1, keepdim=True) + 1e-6
        
        theta_E_expanded = theta_E.view(B, 1, 1, 1)
        alpha = theta_E_expanded * (delta_theta / r)
        
        beta_grid = theta_grid - alpha
        return beta_grid

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        
        params = self.param_estimator(x)
        
        theta_E = F.softplus(params[:, 0]) 
        x_c = torch.tanh(params[:, 1])
        y_c = torch.tanh(params[:, 2])
        
        beta_grid = self._create_physics_grid(theta_E, x_c, y_c, B, H, W, device)
        
        unlensed_x = F.grid_sample(x, beta_grid, align_corners=True, padding_mode='zeros')
        
        combined_features = torch.cat([x, unlensed_x], dim=1) 
        
        logits = self.classifier(combined_features)
        
        return logits