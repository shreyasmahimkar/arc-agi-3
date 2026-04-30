import torch
import torch.nn as nn
import torch.nn.functional as F

class TRMBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Combine input and latent state
        combined = torch.cat([x, z], dim=-1)
        out = F.relu(self.fc1(combined))
        out = self.fc2(out)
        # Residual connection
        return out + z

class TinyRecursiveModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, max_steps: int = 10, threshold: float = 1e-4):
        """
        Tiny Recursive Model (TRM) constrained to a shallow architecture (< 10M params).
        
        Args:
            input_dim: Dimension of input features.
            output_dim: Dimension of output predictions.
            hidden_dim: Hidden dimension size.
            max_steps: Maximum number of recursion cycles (N).
            threshold: Adaptive halting threshold for latent state stabilization.
        """
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        self.trm_block = TRMBlock(hidden_dim)
        
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        self.threshold = threshold
        
        # Verify parameter count is well under 10M
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"TRM Initialized. Total Parameters: {total_params:,}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Iterative forward pass simultaneously taking input x, updating latent state z,
        and refining output prediction y over dynamic recursion cycles.
        """
        batch_size = x.size(0)
        
        # Project input to hidden dimension
        x_encoded = F.relu(self.input_proj(x))
        
        # Initialize latent reasoning state z (zeros)
        z = torch.zeros_like(x_encoded)
        
        step = 0
        for step in range(self.max_steps):
            z_next = self.trm_block(x_encoded, z)
            
            # Adaptive Halting Logic: check if latent state z has stabilized
            diff = torch.norm(z_next - z, p=2, dim=-1).mean()
            z = z_next
            
            if diff < self.threshold:
                # Latent state stabilized, stop early
                break
                
        # Refine output prediction y from final latent state
        y = self.output_proj(z)
        
        return y, step + 1
