import torch

from collections.abc import Callable

from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler


_euler: EulerDiscreteScheduler = EulerDiscreteScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    prediction_type="epsilon",
)

_sigma_max: float = _euler.sigmas[0].item()
_sigma_min: float = _euler.sigmas[-2].item()


@torch.inference_mode()
def sample(x: torch.Tensor, schedule: Callable[[float, float, torch.device], torch.Tensor], predict: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
    sigmas: torch.Tensor = schedule(_sigma_min, _sigma_max, x.device)

    _euler.set_timesteps(sigmas=sigmas.tolist(), device=x.device)

    x = x * _euler.init_noise_sigma

    for timestep in _euler.timesteps:
        latent_input: torch.Tensor = _euler.scale_model_input(x, timestep)

        noise_prediction: torch.Tensor = predict(latent_input, timestep)

        x = _euler.step(noise_prediction, timestep, x, return_dict=False)[0]

    return x