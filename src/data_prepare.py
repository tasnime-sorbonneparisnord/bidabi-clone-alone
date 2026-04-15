import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def clear_directory(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def prepare_dataset_structured(
    images_dir: Path, output_dir: Path, val_ratio: float, test_ratio: float, seed: int
) -> None:
    """
    Prépare les splits train/val/test à partir de data/raw/images/<class>/
    ou data/raw/<class>/ contenant les images.
    """
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    if not images_dir.exists():
        raise FileNotFoundError(f"Le dossier images n'existe pas : {images_dir}")

    # Cherche les sous-dossiers de classes
    class_dirs = [d for d in sorted(images_dir.iterdir()) if d.is_dir()]
    if not class_dirs:
        raise ValueError(f"Aucune classe détectée dans {images_dir}. Vérifiez l'organisation du dataset.")

    for subset in ["train", "val", "test"]:
        clear_directory(output_dir / subset)

    random.seed(seed)
    summary = {"train": 0, "val": 0, "test": 0}

    for class_dir in class_dirs:
        image_paths = [p for p in sorted(class_dir.iterdir()) if is_image_file(p)]
        if not image_paths:
            print(f"⚠ Aucune image trouvée dans {class_dir.name}")
            continue

        random.shuffle(image_paths)
        total = len(image_paths)
        val_count = max(1, int(total * val_ratio))
        test_count = max(1, int(total * test_ratio))
        train_count = total - val_count - test_count
        if train_count < 1:
            train_count = max(1, total - val_count - test_count)

        train_images = image_paths[:train_count]
        val_images = image_paths[train_count : train_count + val_count]
        test_images = image_paths[train_count + val_count :]

        for subset_name, subset_images in [
            ("train", train_images),
            ("val", val_images),
            ("test", test_images),
        ]:
            subset_dir = output_dir / subset_name / class_dir.name
            subset_dir.mkdir(parents=True, exist_ok=True)
            for image_path in subset_images:
                shutil.copy2(image_path, subset_dir / image_path.name)
            summary[subset_name] += len(subset_images)

    print("\nDataset préparé :")
    print(f"  train: {summary['train']} images")
    print(f"  val: {summary['val']} images")
    print(f"  test: {summary['test']} images")


def main():
    parser = argparse.ArgumentParser(description="Prépare les splits train/val/test à partir de data/raw/images ou data/raw.")
    parser.add_argument(
        "--images-dir",
        default="data/raw/images",
        help="Dossier contenant les sous-dossiers de classes avec les images",
    )
    parser.add_argument("--output-dir", default="data", help="Dossier de sortie pour data/train, data/val et data/test")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Proportion de validation")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Proportion de test")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire pour la reproductibilité")
    args = parser.parse_args()

    prepare_dataset_structured(
        images_dir=Path(args.images_dir),
        output_dir=Path(args.output_dir),
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
