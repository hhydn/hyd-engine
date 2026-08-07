import torch


class Karras:
    def schedule(self, steps: int, sigma_min: float, sigma_max: float, device: torch.device, rho: float = 7.0) -> torch.Tensor:
        ramp: torch.Tensor = torch.linspace(0, 1, steps, device=device)

        min_inv_rho: float = sigma_min ** (1 / rho)
        max_inv_rho: float = sigma_max ** (1 / rho)

        sigmas: torch.Tensor = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho

        return torch.cat([sigmas, sigmas.new_zeros(1)])