import os
import re
from PIL import Image, ImageDraw


def extract_image_id(filename):
    """
    Extract the image_id from filenames such as:

      role_dist_epoch_100_image_391895.png
      role_bbox_epoch_100_image_391895.png
    """
    match = re.search(r"_image_(\d+)\.png$", filename)
    if match:
        return int(match.group(1))
    return None


def make_combined_page(heatmap_path, bbox_path, image_id, gap=30, top_margin=50):
    """
    Create one PDF page containing the heatmap and its corresponding
    role-colored bounding-box image side by side.
    """
    heatmap = Image.open(heatmap_path).convert("RGB")
    bbox = Image.open(bbox_path).convert("RGB")

    target_height = max(heatmap.height, bbox.height)

    def resize_to_height(img, height):
        if img.height == height:
            return img
        scale = height / img.height
        new_width = max(1, int(round(img.width * scale)))
        return img.resize((new_width, height), Image.Resampling.LANCZOS)

    heatmap = resize_to_height(heatmap, target_height)
    bbox = resize_to_height(bbox, target_height)

    page_width = heatmap.width + gap + bbox.width
    page_height = target_height + top_margin

    page = Image.new("RGB", (page_width, page_height), "white")
    draw = ImageDraw.Draw(page)

    draw.text((10, 10), f"Image ID {image_id} | Epoch 100", fill="black")
    draw.text((10, 30), "ROLE distribution heatmap", fill="black")
    draw.text(
        (heatmap.width + gap + 10, 30),
        "Role-colored bounding boxes",
        fill="black",
    )

    page.paste(heatmap, (0, top_margin))
    page.paste(bbox, (heatmap.width + gap, top_margin))

    return page


def heatmaps_and_bboxes_to_pdf(epoch_dir, output_pdf):
    """
    Pair ROLE heatmaps and role-colored bounding-box images by image_id
    and save them to one multi-page PDF.

    Each PDF page contains:
      left:  role_dist_epoch_100_image_<image_id>.png
      right: role_bbox_epoch_100_image_<image_id>.png
    """
    filenames = [
        f for f in os.listdir(epoch_dir)
        if f.lower().endswith(".png")
    ]

    heatmaps = {}
    bbox_images = {}

    for filename in filenames:
        image_id = extract_image_id(filename)
        if image_id is None:
            continue

        if filename.startswith("role_dist_epoch_250_image_"):
            heatmaps[image_id] = filename
        elif filename.startswith("role_bbox_epoch_250_image_"):
            bbox_images[image_id] = filename

    paired_ids = sorted(set(heatmaps) & set(bbox_images))

    if not paired_ids:
        raise RuntimeError(
            f"No matching heatmap/bounding-box pairs found in {epoch_dir}.\n"
            "Expected filenames like:\n"
            "  role_dist_epoch_001_image_<image_id>.png\n"
            "  role_bbox_epoch_001_image_<image_id>.png"
        )

    missing_bbox = sorted(set(heatmaps) - set(bbox_images))
    missing_heatmap = sorted(set(bbox_images) - set(heatmaps))

    if missing_bbox:
        print(
            "Warning: heatmaps without corresponding bbox images for image IDs:",
            missing_bbox,
        )

    if missing_heatmap:
        print(
            "Warning: bbox images without corresponding heatmaps for image IDs:",
            missing_heatmap,
        )

    pages = []

    for image_id in paired_ids:
        heatmap_path = os.path.join(epoch_dir, heatmaps[image_id])
        bbox_path = os.path.join(epoch_dir, bbox_images[image_id])

        page = make_combined_page(
            heatmap_path,
            bbox_path,
            image_id=image_id,
        )
        pages.append(page)

    pages[0].save(
        output_pdf,
        save_all=True,
        append_images=pages[1:],
        resolution=200.0,
    )

    print(f"Saved {len(pages)} paired image pages to:")
    print(output_pdf)


if __name__ == "__main__":

    # Change this parent directory if your ROLE_OUT_DIR is different.
    epoch_dir = r"role_plots_bbox_images_reweighted_filler_indep/epoch_250"

    output_pdf = os.path.join(
        epoch_dir,
        "role_heatmaps_and_bbox_images.pdf",
    )

    heatmaps_and_bboxes_to_pdf(epoch_dir, output_pdf)
