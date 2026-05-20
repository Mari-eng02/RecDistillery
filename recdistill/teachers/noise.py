from __future__ import annotations

import torch

from recdistill.teachers.state import TeacherState


def _scaled_noise_std(
    teacher_state: TeacherState,
    noise_scale: float,
    target: str,
) -> tuple[float, float]:
    user_embs = teacher_state.user_embeddings
    item_embs = teacher_state.item_embeddings
    if user_embs is None or item_embs is None:
        raise ValueError("Teacher noise injection requires an embedding-based teacher.")

    if target == "user":
        base_std = float(user_embs.std().item())
    elif target == "item":
        base_std = float(item_embs.std().item())
    else:
        all_embs = torch.cat([user_embs, item_embs], dim=0)
        base_std = float(all_embs.std().item())

    return base_std, float(noise_scale) * base_std


def inject_static_noise(
    teacher_state: TeacherState,
    noise_scale: float,
    target: str = "both",
    seed: int | None = None,
) -> tuple[TeacherState, dict[str, float | str | int | None]]:
    """
    Inject Gaussian noise into teacher embeddings.

    The final perturbation std is computed as:
        noise_std = noise_scale * std(teacher_embeddings_target)

    Returns:
        (possibly updated TeacherState, metadata dict)
    """
    if target not in {"both", "user", "item"}:
        raise ValueError("target must be one of: both, user, item")
    if not teacher_state.has_embeddings:
        raise ValueError("Teacher noise injection requires an embedding-based teacher.")

    if noise_scale <= 0.0:
        return teacher_state, {
            "noise_scale": float(noise_scale),
            "noise_target": str(target),
            "base_std": 0.0,
            "scaled_noise_std": 0.0,
            "noise_seed": int(seed) if seed is not None else None,
            "enabled": False,
        }

    base_std, noise_std = _scaled_noise_std(teacher_state=teacher_state, noise_scale=noise_scale, target=target)

    device = teacher_state.user_embeddings.device
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    user_embeddings = teacher_state.user_embeddings.clone()
    item_embeddings = teacher_state.item_embeddings.clone()

    with torch.no_grad():
        if target in {"both", "user"}:
            user_noise = torch.randn(
                user_embeddings.shape,
                device=user_embeddings.device,
                dtype=user_embeddings.dtype,
                generator=generator,
            ) * noise_std
            user_embeddings.add_(user_noise)

        if target in {"both", "item"}:
            item_noise = torch.randn(
                item_embeddings.shape,
                device=item_embeddings.device,
                dtype=item_embeddings.dtype,
                generator=generator,
            ) * noise_std
            item_embeddings.add_(item_noise)

    metadata = dict(teacher_state.metadata)
    metadata["noise_injection"] = {
        "enabled": True,
        "noise_scale": float(noise_scale),
        "noise_target": str(target),
        "base_std": float(base_std),
        "scaled_noise_std": float(noise_std),
        "noise_seed": int(seed) if seed is not None else None,
    }

    updated_state = TeacherState(
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        metadata=metadata,
        scorer=teacher_state.scorer,
    )
    return updated_state, metadata["noise_injection"]
