import torch

from lerobot.policies.pi05.bspline import (
    bspline_basis,
    cartesian_quaternion_to_local_rotvec,
    fit_control_points,
    rebuild_trajectory,
    refit_control_point_prefix,
)
from lerobot.policies.pi05.configuration_pi05 import PI05Config


def test_basis_is_partition_of_unity_and_clamped():
    basis = bspline_basis(40, 8, 3)
    torch.testing.assert_close(basis.sum(dim=1), torch.ones(40))
    torch.testing.assert_close(basis[0], torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0]))
    torch.testing.assert_close(basis[-1], torch.tensor([0.0, 0, 0, 0, 0, 0, 0, 1]))


def test_fit_and_rebuild_exact_spline():
    basis = bspline_basis(40, 8, 3)
    expected = torch.randn(2, 8, 7)
    fitted = fit_control_points(rebuild_trajectory(expected, basis), basis)
    torch.testing.assert_close(fitted, expected, atol=1e-5, rtol=1e-5)


def test_fit_supports_bfloat16_actions():
    basis = bspline_basis(40, 8, 3, dtype=torch.bfloat16)
    trajectory = torch.randn(2, 40, 7, dtype=torch.bfloat16)
    fitted = fit_control_points(trajectory, basis)
    assert fitted.dtype == torch.bfloat16
    assert fitted.shape == (2, 8, 7)


def test_refit_changes_only_free_prefix_and_reduces_history_error():
    basis = bspline_basis(40, 8, 3)
    predicted = torch.randn(8, 7)
    desired = torch.randn(8, 7)
    refitted = refit_control_point_prefix(desired, predicted, basis, num_free_control_points=4)
    torch.testing.assert_close(refitted[4:], predicted[4:])
    assert torch.mean((rebuild_trajectory(refitted, basis)[:8] - desired) ** 2) < torch.mean(
        (rebuild_trajectory(predicted, basis)[:8] - desired) ** 2
    )


def test_abpolicy_config_uses_bidirectional_action_window():
    config = PI05Config(abpolicy_enabled=True, chunk_size=8, n_action_steps=32)
    assert config.action_delta_indices == list(range(-8, 32))


def test_cartesian_quaternion_actions_use_request_local_rotation_vectors():
    actions = torch.zeros(1, 3, 8)
    actions[..., 6] = 1.0
    actions[..., 7] = 0.25
    half_angle = torch.tensor(torch.pi / 4)
    actions[0, 2, 5] = torch.sin(half_angle)
    actions[0, 2, 6] = torch.cos(half_angle)
    packed = cartesian_quaternion_to_local_rotvec(actions, reference_index=1)
    torch.testing.assert_close(packed[0, 1, 3:6], torch.zeros(3))
    torch.testing.assert_close(packed[0, 2, 3:6], torch.tensor([0.0, 0.0, torch.pi / 2]))
    torch.testing.assert_close(packed[..., 6], actions[..., 7])
    torch.testing.assert_close(packed[..., 7], torch.zeros_like(packed[..., 7]))
