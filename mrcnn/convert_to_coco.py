import os
import json
import glob
from datetime import datetime

def polygon_area(points):
    """
    Compute polygon area via the shoelace formula.
    `points` is a list of [x, y] pairs.
    """
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    # close the loop
    x.append(x[0])
    y.append(y[0])
    return 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(len(points))))

def convert_folder_to_coco(anno_dir, out_dir, out_filename):
    images       = []
    annotations  = []
    category_map = {}   # label → category_id
    ann_id       = 1
    img_id       = 1

    json_files = sorted(glob.glob(os.path.join(anno_dir, "*.json")))
    if not json_files:
        raise ValueError(f"No JSON files found in {anno_dir!r}")

    for filepath in json_files:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # --- Image metadata ---
        img_name = data.get("imageName") or os.path.basename(data.get("imagePath", ""))
        width    = data.get("imageWidth")
        height   = data.get("imageHeight")

        if not img_name or width is None or height is None:
            raise KeyError(f"Missing image metadata in {filepath!r}")

        images.append({
            "id": img_id,
            "file_name": img_name,
            "width": width,
            "height": height
        })

        # --- Annotations ---
        for shape in data.get("shapes", []):
            label  = shape.get("label")
            points = shape.get("points")

            if label is None or not points:
                continue

            if label not in category_map:
                category_map[label] = len(category_map) + 1

            # flatten segmentation
            segmentation = [coord for pt in points for coord in pt]

            xs = [pt[0] for pt in points]
            ys = [pt[1] for pt in points]
            x_min, y_min = min(xs), min(ys)
            w_box = max(xs) - x_min
            h_box = max(ys) - y_min

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_map[label],
                "segmentation": [segmentation],
                "area": polygon_area(points),
                "bbox": [x_min, y_min, w_box, h_box],
                "iscrowd": 0
            })
            ann_id += 1

        img_id += 1

    # --- Build categories list ---
    categories = [
        {"id": cid, "name": name, "supercategory": ""}
        for name, cid in sorted(category_map.items(), key=lambda x: x[1])
    ]

    # --- Assemble COCO dict ---
    coco_dict = {
        "info": {
            "description": "",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    # ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, out_filename)
    with open(output_path, "w") as out_f:
        json.dump(coco_dict, out_f, indent=2)

    print(f"\n✅ Wrote COCO JSON:")
    print(f"   Images:      {len(images)}")
    print(f"   Annotations: {len(annotations)}")
    print(f"   Categories:  {len(categories)}")
    print(f"   → {output_path}\n")

if __name__ == "__main__":
    print("\n=== COCO Converter ===")
    anno_dir     = r'C:\Users\jcac\OneDrive - KTH\Journals\03-Synthetic data\03-Code\Mask_RCNN\annotations\val'
    out_dir      = r'C:\Users\jcac\OneDrive - KTH\Journals\03-Synthetic data\03-Code\Mask_RCNN\annotations\annotations'
    out_filename = 'instances_val2017.json'

    # Optionally enforce .json extension
    if not out_filename.lower().endswith(".json"):
        out_filename += ".json"

    try:
        convert_folder_to_coco(anno_dir, out_dir, out_filename)
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
