"""Torch-native B-spline action representation for ABPolicy."""

from __future__ import annotations

import torch
from torch import Tensor


def _normalize_quaternion(quaternion: Tensor) -> Tensor:
    return quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def quaternion_conjugate(quaternion: Tensor) -> Tensor:
    return torch.cat((-quaternion[..., :3], quaternion[..., 3:]), dim=-1)


def quaternion_multiply(lhs: Tensor, rhs: Tensor) -> Tensor:
    lx, ly, lz, lw = lhs.unbind(dim=-1)
    rx, ry, rz, rw = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dim=-1,
    )


def quaternion_to_rotation_vector(quaternion: Tensor) -> Tensor:
    """Map unit xyzw quaternions to the shortest SO(3) logarithm."""
    quaternion = _normalize_quaternion(quaternion)
    quaternion = torch.where(quaternion[..., 3:4] < 0, -quaternion, quaternion)
    vector = quaternion[..., :3]
    vector_norm = vector.norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(vector_norm, quaternion[..., 3:4].clamp_min(0))
    scale = torch.where(vector_norm > 1e-7, angle / vector_norm, 2 * torch.ones_like(vector_norm))
    return vector * scale


def cartesian_quaternion_to_local_rotvec(actions: Tensor, reference_index: int) -> Tensor:
    """Pack xyz, request-local rotation vector, gripper and one zero into 8D."""
    if actions.shape[-1] != 8:
        raise ValueError(f"cartesian_rotvec ABPolicy expects 8D actions, got {actions.shape[-1]}")
    reference = _normalize_quaternion(actions[..., reference_index, 3:7])
    relative = quaternion_multiply(
        quaternion_conjugate(reference).unsqueeze(-2), _normalize_quaternion(actions[..., 3:7])
    )
    rotation_vector = quaternion_to_rotation_vector(relative)
    zero = torch.zeros_like(actions[..., 7:8])
    return torch.cat((actions[..., :3], rotation_vector, actions[..., 7:8], zero), dim=-1)


def clamped_uniform_knots(num_control_points: int, degree: int, *, device=None, dtype=None) -> Tensor:
    if degree < 1 or num_control_points <= degree:
        raise ValueError("num_control_points must be greater than a positive degree")
    interior_count = num_control_points - degree - 1
    interior = (
        torch.linspace(0, 1, interior_count + 2, device=device, dtype=dtype)[1:-1]
        if interior_count
        else torch.empty(0, device=device, dtype=dtype)
    )
    return torch.cat(
        [
            torch.zeros(degree + 1, device=device, dtype=dtype),
            interior,
            torch.ones(degree + 1, device=device, dtype=dtype),
        ]
    )


def bspline_basis(
    trajectory_length: int,
    num_control_points: int,
    degree: int = 3,
    *,
    device=None,
    dtype=torch.float32,
) -> Tensor:
    """Return a ``(trajectory_length, num_control_points)`` basis matrix."""
    if trajectory_length < 2:
        raise ValueError("trajectory_length must be at least two")
    knots = clamped_uniform_knots(num_control_points, degree, device=device, dtype=dtype)
    u = torch.linspace(0, 1, trajectory_length, device=device, dtype=dtype)
    basis = ((u[:, None] >= knots[:-1]) & (u[:, None] < knots[1:])).to(dtype)
    basis[-1].zero_()
    basis[-1, num_control_points - 1] = 1
    for current_degree in range(1, degree + 1):
        columns = []
        for i in range(num_control_points):
            left_denom = knots[i + current_degree] - knots[i]
            right_denom = knots[i + current_degree + 1] - knots[i + 1]
            left = torch.zeros_like(u)
            right = torch.zeros_like(u)
            if left_denom.item() > 0:
                left = (u - knots[i]) / left_denom * basis[:, i]
            if i + 1 < basis.shape[1] and right_denom.item() > 0:
                right = (knots[i + current_degree + 1] - u) / right_denom * basis[:, i + 1]
            columns.append(left + right)
        basis = torch.stack(columns, dim=1)
    basis[-1].zero_()
    basis[-1, -1] = 1
    return basis


def fit_control_points(actions: Tensor, basis: Tensor) -> Tensor:
    if actions.shape[-2] != basis.shape[0]:
        raise ValueError(f"trajectory length {actions.shape[-2]} does not match basis {basis.shape[0]}")
    output_dtype = actions.dtype
    solve_dtype = torch.float64 if output_dtype == torch.float64 else torch.float32
    fitted = torch.einsum(
        "nt,...td->...nd",
        torch.linalg.pinv(basis.to(solve_dtype)),
        actions.to(solve_dtype),
    )
    return fitted.to(output_dtype)


def rebuild_trajectory(control_points: Tensor, basis: Tensor) -> Tensor:
    if control_points.shape[-2] != basis.shape[1]:
        raise ValueError(f"control point count {control_points.shape[-2]} does not match basis {basis.shape[1]}")
    return torch.einsum("tn,...nd->...td", basis, control_points)


def refit_control_point_prefix(
    executed_actions: Tensor,
    predicted_control_points: Tensor,
    basis: Tensor,
    *,
    num_free_control_points: int,
    prefix_start: int = 0,
) -> Tensor:
    """Adjust only the first control points to fit executed action history."""
    prefix_length = executed_actions.shape[-2]
    if not 0 < num_free_control_points <= predicted_control_points.shape[-2]:
        raise ValueError("invalid num_free_control_points")
    prefix_basis = basis[prefix_start : prefix_start + prefix_length]
    free_basis = prefix_basis[:, :num_free_control_points]
    fixed_basis = prefix_basis[:, num_free_control_points:]
    fixed_points = predicted_control_points[..., num_free_control_points:, :]
    output_dtype = executed_actions.dtype
    solve_dtype = torch.float64 if output_dtype == torch.float64 else torch.float32
    residual = executed_actions.to(solve_dtype) - torch.einsum(
        "tn,...nd->...td", fixed_basis.to(solve_dtype), fixed_points.to(solve_dtype)
    )
    free_points = torch.einsum(
        "nt,...td->...nd", torch.linalg.pinv(free_basis.to(solve_dtype)), residual
    ).to(output_dtype)
    result = predicted_control_points.clone()
    result[..., :num_free_control_points, :] = free_points
    return result
