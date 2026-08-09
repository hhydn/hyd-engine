import torch


class Euler:
    @torch.inference_mode()
    def step(self, x: torch.Tensor, sigma: torch.Tensor, sigma_next: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
        d: torch.Tensor = (x - denoised) / sigma
        dt: torch.Tensor = sigma_next - sigma
        return x + d * dt