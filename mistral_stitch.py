"""Combine uploaded images into one horizontal image for vision models."""

from PIL import Image


def stitch_images(images):
    """Keep the first image's height and resize the rest proportionally."""
    images = list(images or [])
    if len(images) < 2:
        return images

    target_height = images[0].height
    resized = []
    for image in images:
        width = max(1, round(image.width * target_height / image.height))
        if image.height == target_height and image.width == width:
            resized.append(image.convert("RGB"))
        else:
            resized.append(image.resize((width, target_height), Image.Resampling.LANCZOS).convert("RGB"))

    result = Image.new("RGB", (sum(image.width for image in resized), target_height), "white")
    x = 0
    for image in resized:
        result.paste(image, (x, 0))
        x += image.width
    return [result]
