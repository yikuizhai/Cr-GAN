"""Generate images from a trained CR-GAN generator checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from torchvision.utils import save_image

from crgan.models import Generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CR-GAN image generation")
    parser.add_argument("--checkpoint", required=True, help="generator state_dict checkpoint")
    parser.add_argument("--output", default="outputs/generated", help="output directory")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index; CPU is used when CUDA is unavailable")
    return parser.parse_args()


def load_generator(args: argparse.Namespace, device: torch.device) -> Generator:
    model_args = SimpleNamespace(
        latent_dim=args.latent_dim,
        img_size=args.img_size,
        channels=args.channels,
        classes=args.classes,
    )
    generator = Generator(model_args).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    generator.load_state_dict(state)
    generator.eval()
    return generator


def main() -> None:
    args = parse_args()
    if args.num_images <= 0 or args.batch_size <= 0:
        raise ValueError("num-images and batch-size must be positive")

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    generator = load_generator(args, device)

    written = 0
    with torch.inference_mode():
        while written < args.num_images:
            count = min(args.batch_size, args.num_images - written)
            latent = torch.randn(count, args.latent_dim, device=device)
            images = generator(latent)
            for image in images:
                save_image(
                    image,
                    output / f"{written:06d}.png",
                    normalize=True,
                    value_range=(-1, 1),
                )
                written += 1

    print(f"Generated {written} images in {output}")


if __name__ == "__main__":
    main()
