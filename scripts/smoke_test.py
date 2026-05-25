"""Minimal CI smoke test for the active-contour notebook logic.

Runs a few snake iterations on Tasse2.jpg without plotting. Mirrors the
notebook implementation; fixes the notebook's use of global ``kwargs`` in
``greedy_minimization`` by using the ``neighborhood_size`` argument.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from numpy.linalg import norm
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_ROUNDS = 3


def gradient(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k_y = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])
    k_x = np.transpose(k_y)
    image = np.copy(image).astype("float32")
    gradient_x = cv2.filter2D(image, -1, k_x)
    gradient_y = cv2.filter2D(image, -1, k_y)
    return gradient_x, gradient_y


def wrap_index(index: int, cont: int) -> int:
    if index < 0:
        return cont - 1
    if index > cont - 1:
        return 0
    return index


def average_distance(contour: np.ndarray) -> float:
    n = contour.shape[1]
    total = sum(norm(contour[:, i] - contour[:, i + 1]) for i in range(n - 1))
    return total / n


def greedy_minimization(
    norm_image_gradient: np.ndarray,
    contour: np.ndarray,
    i: int,
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    neighborhood_size: int,
) -> tuple[np.ndarray | None, bool]:
    m, n = norm_image_gradient.shape
    cont = contour.shape[1]
    neighborhood = neighborhood_size
    d = average_distance(contour)
    updated = False

    p_prev = contour[:, wrap_index(i - 1, cont)]
    p_curr = contour[:, i]
    p_next = contour[:, wrap_index(i + 1, cont)]

    candidates: list[np.ndarray] = []
    e_cont: list[float] = []
    e_curv: list[float] = []
    e_imag: list[float] = []

    for p_curr_x in np.arange(
        max(0, p_curr[0] - neighborhood), min(m - 1, p_curr[0] + neighborhood) + 1
    ):
        for p_curr_y in np.arange(
            max(0, p_curr[1] - neighborhood), min(n - 1, p_curr[1] + neighborhood) + 1
        ):
            candidates.append(np.array([int(p_curr_x), int(p_curr_y)]))
            e_cont.append((d * norm(candidates[-1] - p_prev)) ** 2)
            e_curv.append(norm(p_prev - 2 * candidates[-1] + p_next) ** 2)
            e_imag.append(-norm_image_gradient[candidates[-1][0], candidates[-1][1]])

    if not candidates:
        return None, False

    e_cont_arr = np.array(e_cont)
    e_curv_arr = np.array(e_curv)
    e_imag_arr = np.array(e_imag)
    e_cont_arr = e_cont_arr / norm(e_cont_arr)
    e_curv_arr = e_curv_arr / norm(e_curv_arr)
    e_imag_arr = e_imag_arr / norm(e_imag_arr)

    energy = alpha[i] * e_cont_arr + beta[i] * e_curv_arr + gamma[i] * e_imag_arr
    new_p_curr = candidates[int(np.argmin(energy))]

    if (p_curr - new_p_curr).sum() != 0:
        updated = True

    return new_p_curr, updated


def snake_rounds(img: np.ndarray, contour: np.ndarray, max_rounds: int, **kwargs) -> np.ndarray:
    n = contour.shape[1]
    alpha = kwargs["alpha"] * np.ones(n)
    beta = kwargs["beta"] * np.ones(n)
    gamma = kwargs["gamma"] * np.ones(n)

    img = cv2.GaussianBlur(img, (5, 5), 0)
    gradient_x, gradient_y = gradient(img)
    norm_image_gradient = np.sqrt(gradient_x**2 + gradient_y**2)

    for _ in range(max_rounds):
        for i in range(n):
            p, _updated = greedy_minimization(
                norm_image_gradient,
                contour,
                i,
                alpha,
                beta,
                gamma,
                kwargs["neighborhood_size"],
            )
            if p is not None:
                contour[:, i] = p

    return contour


def load_image(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    img = Image.open(path)
    image = np.asarray(img)
    scale_percent = 50
    height = int(img.height * scale_percent / 100)
    width = int(img.width * scale_percent / 100)
    dim = (width, height)
    img_res = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return img_res, dim


def set_contour_own(dim: tuple[int, int]) -> np.ndarray:
    center = (dim[1] / 2, dim[0] / 2)
    vertical = (dim[0] / 3) * 1.75
    horizontal = (dim[1] / 3) * 0.9
    t = np.linspace(0, 2 * np.pi, 50, endpoint=True)
    x_0 = center[1] + 65 + vertical * np.sin(t)
    y_0 = center[0] - 85 + horizontal * np.cos(t)
    return np.array([x_0, y_0])


def main() -> None:
    image_path = REPO_ROOT / "Tasse2.jpg"
    if not image_path.is_file():
        print(f"FAIL: missing test image {image_path}", file=sys.stderr)
        sys.exit(1)

    img_own, dim_own = load_image(image_path)
    contour = set_contour_own(dim_own)
    img_gray = cv2.GaussianBlur(img_own[:, :, 0], (59, 59), 0)

    kwargs = {
        "neighborhood_size": 21,
        "contour_fraction": 0.7,
        "alpha": 0.9,
        "beta": 1.5,
        "gamma": 3.0,
        "begin_corner_elim": 30,
        "C_threshold": 1.0,
    }

    before = contour.copy()
    result = snake_rounds(img_gray, contour, MAX_ROUNDS, **kwargs)

    assert result.shape == (2, 50), f"unexpected contour shape: {result.shape}"
    assert np.isfinite(result).all(), "contour contains non-finite values"
    moved = not np.allclose(before, result)
    if not moved:
        print("WARN: contour unchanged after smoke iterations (may still be valid)")

    print(
        f"OK: active-contour smoke test passed "
        f"({MAX_ROUNDS} rounds, contour moved={moved})"
    )


if __name__ == "__main__":
    main()
