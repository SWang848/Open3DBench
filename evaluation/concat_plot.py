import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def _pad_to_shape(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    if image.ndim == 2:
        canvas = np.ones((target_height, target_width), dtype=image.dtype)
    else:
        channels = image.shape[2]
        canvas = np.ones((target_height, target_width, channels), dtype=image.dtype)

    if np.issubdtype(image.dtype, np.integer):
        canvas *= np.iinfo(image.dtype).max

    y_offset = (target_height - image.shape[0]) // 2
    x_offset = (target_width - image.shape[1]) // 2
    canvas[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1]] = image
    return canvas


def concatenate_images(
    image1_path: Path,
    image2_path: Path,
    output_path: Path,
    layout: str = "horizontal",
) -> None:
    image1 = mpimg.imread(image1_path)
    image2 = mpimg.imread(image2_path)
    target_height = max(image1.shape[0], image2.shape[0])
    target_width = max(image1.shape[1], image2.shape[1])
    image1 = _pad_to_shape(image1, target_height, target_width)
    image2 = _pad_to_shape(image2, target_height, target_width)

    if layout == "vertical":
        fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image1)
    axes[0].axis("off")
    axes[1].imshow(image2)
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concatenate two images with matplotlib.")
    parser.add_argument("image1", type=Path, help="Path to the first image.")
    parser.add_argument("image2", type=Path, help="Path to the second image.")
    parser.add_argument("output", type=Path, help="Path to the output image.")
    parser.add_argument(
        "--layout",
        choices=["horizontal", "vertical"],
        default="horizontal",
        help="How to arrange the two images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    concatenate_images(args.image1, args.image2, args.output, layout=args.layout)


if __name__ == "__main__":
    main()
