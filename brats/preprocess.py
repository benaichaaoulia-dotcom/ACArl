import SimpleITK as sitk
from PIL import Image
import os
import argparse
import multiprocessing as mp
from tqdm import tqdm
import numpy as np


def preprocess(args) -> None:
    input_path, output_path = args
    os.makedirs(output_path)
    sample_name = input_path.split(os.sep)[-1]
    image = sitk.GetArrayFromImage(
        sitk.ReadImage(os.path.join(input_path, f"{sample_name}_flair.nii"))
    ).astype(np.float32)
    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    label = sitk.GetArrayFromImage(
        sitk.ReadImage(os.path.join(input_path, f"{sample_name}_seg.nii"))
    ).astype(np.uint8)
    label[label != 0] = 255
    for i, (img, lab) in enumerate(zip(image, label)):
        Image.fromarray(img, mode="L").save(
            os.path.join(output_path, f"img-{str(i).zfill(3)}.jpg"), quality=90
        )
        Image.fromarray(lab, mode="L").save(
            os.path.join(output_path, f"seg-{str(i).zfill(3)}.png")
        )


def generate_splits(output_path: str) -> None:
    """Generate train/val/test splits in output_path/splits.

    Creates train.txt (80%), val.txt (20%), and empty test.txt.
    """
    splits_dir = os.path.join(output_path, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    patients = sorted([d for d in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, d)) and not d.startswith('.')])
    total = len(patients)
    n_train = int(total * 80 / 100)

    with open(os.path.join(splits_dir, "train.txt"), "w") as f:
        for p in patients[:n_train]:
            f.write(p + "\n")

    with open(os.path.join(splits_dir, "val.txt"), "w") as f:
        for p in patients[n_train:]:
            f.write(p + "\n")

    open(os.path.join(splits_dir, "test.txt"), "w").close()
    print(f"Generated splits in {splits_dir}: {n_train} train, {total-n_train} val, 0 test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocessing")
    parser.add_argument("--input-path", type=str, help="input path", required=False)
    parser.add_argument("--output-path", type=str, help="output path", required=True)
    parser.add_argument("--generate-splits", action="store_true", help="Only generate train/val/test splits in the output path")
    parser.add_argument("--workers", type=int, help="preprocess workers", default=4)
    args = parser.parse_args()

    # Ensure output path exists (do not fail if it already exists)
    os.makedirs(args.output_path, exist_ok=True)

    # If only generating splits, skip requiring input
    if args.generate_splits:
        generate_splits(args.output_path)
        raise SystemExit(0)

    # Validate input path exists
    if not os.path.exists(args.input_path):
        print(f"Input path does not exist: {args.input_path}")
        raise SystemExit(1)

    input_paths = [
        os.path.join(args.input_path, tumor_type, sample_name)
        for tumor_type in ["HGG", "LGG"]
        for sample_name in os.listdir(os.path.join(args.input_path, tumor_type))
    ]

    output_paths = [
        os.path.join(args.output_path, sample_name)
        for tumor_type in ["HGG", "LGG"]
        for sample_name in os.listdir(os.path.join(args.input_path, tumor_type))
    ]

    with mp.Pool(args.workers) as pool:
        list(
            tqdm(
                pool.imap(preprocess, zip(input_paths, output_paths)),
                total=len(input_paths),
                ncols=100,
            )
        )

    if args.generate_splits:
        # Only generate splits and exit
        generate_splits(args.output_path)
        raise SystemExit(0)

    # Generate splits after preprocessing by default
    generate_splits(args.output_path)
