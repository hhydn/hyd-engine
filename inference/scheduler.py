import torch

from collections.abc import Callable


_steps: int


def on_steps_changed(steps: int) -> None:
    global _steps

    _steps = steps


def get_schedule() -> Callable[[float, float, torch.device], torch.Tensor]:
    return karras


def karras(sigma_min: float, sigma_max: float, device: torch.device, rho: float = 7.0) -> torch.Tensor:
    ramp: torch.Tensor = torch.linspace(0, 1, _steps, device=device)

    min_inv_rho: float = sigma_min ** (1 / rho)
    max_inv_rho: float = sigma_max ** (1 / rho)

    sigmas: torch.Tensor = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho

    return torch.cat([sigmas, sigmas.new_zeros(1)])