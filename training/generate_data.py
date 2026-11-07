from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance


BRICK_DIR = Path("dataset/bricks/no-bg")
BACKGROUND_DIR = Path("dataset/not-bricks/val2014")
OUTPUT_DIR = Path("dataset/generated")

IMAGE_SIZE = (512, 512)
SAMPLES_PER_SPLIT = {
    "train": 4000,
    "val": 500,
    "test": 500,
}

# Fraction of generated samples that contain at least one brick.
BRICK_PROBABILITY = 0.80

# Randomness is repeatable for easier debugging.
random.seed(42)     # The meaning of life :amen:
np.random.seed(42)


def load_brick(image_path: Path) -> Image.Image:
    """
    Load a cutout, crop transparent margins, return RGBA.
    
    :param image_path: The path to the image
    :return: The image cropped perfectly to the brick
    """
    image = Image.open(image_path).convert("RGBA")

    alpha = image.getchannel("A")
    bbox = alpha.getbbox()

    if bbox is None:
        # How did we get here
        raise ValueError(f"Image has no visible pixels: {image_path}")

    return image.crop(bbox)


def transform_brick(brick: Image.Image, max_extent: int) -> Image.Image:
    """
    Randomly resize, rotate, and adjust brightness.
    
    :param brick: The brick image already cropped
    :param max_extent: Relatively the max size of the new, transformed brick
    :return: The transformed brick (resized, rotated, and brightness adjusted)
    """
    # Choose a target maximum dimension.
    # Min size is 32x32 (any smaller and detail is just silly)
    target_extent = random.randint(
        max(32, int(max_extent * 0.10)),
        max(32, int(max_extent * 0.70)),
    )

    scale = target_extent / max(brick.width, brick.height)

    new_size = (
        max(1, round(brick.width * scale)),
        max(1, round(brick.height * scale)),
    )

    brick = brick.resize(new_size, Image.Resampling.LANCZOS)

    # Rotate with expansion so the whole object remains in the image.
    brick = brick.rotate(
        random.uniform(0, 360),
        resample=Image.Resampling.BICUBIC,
        expand=True,
    )

    # Change brightness without changing transparency.
    rgb = Image.new("RGB", brick.size)
    rgb.paste(brick, mask=brick.getchannel("A"))

    # 35% above or below
    rgb = ImageEnhance.Brightness(rgb).enhance(
        random.uniform(0.65, 1.35)
    )

    result = rgb.convert("RGBA")
    result.putalpha(brick.getchannel("A"))

    return result


def place_brick(background: Image.Image, brick: Image.Image) -> list[int] | None:
    """
    Composite a brick and return its COCO [x, y, w, h] box.
    
    :param background: The image to place behind the brick
    :param brick: The brick to place
    :return:    The location and bounding box of the brick in the image (x,y,w,h) (x,y = top-left).
                None if the brick is too large or has no bounding box.
    """
    bg_width, bg_height = background.size
    brick_width, brick_height = brick.size

    # Skip bricks too large to fit on the background.
    if brick_width > bg_width or brick_height > bg_height:
        return None

    # Place the brick randomly
    x = random.randint(0, bg_width - brick_width)
    y = random.randint(0, bg_height - brick_height)

    # Alpha composite the brick over the background.
    background.alpha_composite(brick, (x, y))

    # Use the transformed alpha mask to calculate the tight box.
    alpha_bbox = brick.getchannel("A").getbbox()

    if alpha_bbox is None:
        return None

    left, top, right, bottom = alpha_bbox

    x_min = x + left
    y_min = y + top
    box_width = right - left
    box_height = bottom - top

    return [x_min, y_min, box_width, box_height]


def main() -> None:
    """
    Goes through all the bricks and backgrounds and creates a bunch
    of generated data of bricks over backgrounds.
    """
    brick_paths = sorted(BRICK_DIR.glob("*.png"))

    background_paths = sorted(
        p for p in BACKGROUND_DIR.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not brick_paths:
        raise RuntimeError(f"No brick PNGs found in {BRICK_DIR}")

    if not background_paths:
        raise RuntimeError(f"No backgrounds found in {BACKGROUND_DIR}")

    bricks = [load_brick(path) for path in brick_paths]

    # Split backgrounds before generating images. This avoids having
    # the same original background appear in multiple splits.
    random.shuffle(background_paths)

    n = len(background_paths)
    background_splits = {
        "train": background_paths[: int(n * 0.80)],             # 80%
        "val": background_paths[int(n * 0.80): int(n * 0.90)],  # 10%
        "test": background_paths[int(n * 0.90):],               # 10%
    }

    count = 0
    for split, sample_count in SAMPLES_PER_SPLIT.items():
        split_dir = OUTPUT_DIR / split
        image_dir = split_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        backgrounds = background_splits[split]

        if not backgrounds:
            raise RuntimeError(f"No backgrounds available for {split}")

        images = []
        annotations = []
        annotation_id = 1

        for image_id in range(1, sample_count + 1):
            count += 1
            if (count%100 == 0):
                print(f"Finished {count}")
            bg_path = random.choice(backgrounds)

            background = Image.open(bg_path).convert("RGB")
            background = background.resize(
                IMAGE_SIZE,
                Image.Resampling.LANCZOS,
            ).convert("RGBA")

            boxes = []

            if random.random() < BRICK_PROBABILITY:
                # Start with one brick; increase to 2–3 randomly
                num_bricks = random.choices(
                    [1, 2, 3],
                    weights=[0.75, 0.20, 0.05],
                )[0]

                for _ in range(num_bricks):
                    brick = random.choice(bricks)
                    brick = transform_brick(
                        brick,
                        max_extent=min(IMAGE_SIZE),
                    )

                    box = place_brick(background, brick)

                    if box is not None:
                        boxes.append(box)

            filename = f"{image_id:06d}.jpg"
            output_path = image_dir / filename

            background.convert("RGB").save(
                output_path,
                quality=95,
            )

            images.append({
                "id": image_id,
                "file_name": filename,
                "width": IMAGE_SIZE[0],
                "height": IMAGE_SIZE[1],
            })

            for x, y, width, height in boxes:
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                })

                annotation_id += 1

        dataset = {
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": 1, "name": "brick"}
            ],
        }

        with (split_dir / "annotations.json").open("w") as f:
            json.dump(dataset, f)

        print(
            f"{split}: wrote {len(images)} images and "
            f"{len(annotations)} brick boxes"
        )


if __name__ == "__main__":
    main()