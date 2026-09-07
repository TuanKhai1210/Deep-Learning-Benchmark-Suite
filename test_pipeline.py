from pathlib import Path
import torch

from dlbench.a1.data.dataset import prepare_data
from dlbench.a1.data.loaders import build_dataloaders
from dlbench.a1.data.eda import generate_eda

def main() -> None:
    print("=== Step 1: Running prepare_data ===")
    config = {
        "data_root": "./data",
        "split_path": "configs/a1/splits/fashion_mnist_split.json",
        "split_seed": 36,
        "validation_size": 10_000,
        "download": True,
        "batch_size": 64,
        "run_seed": 69420,
    }

    # 1. Prepare data & generate split manifest
    metadata = prepare_data(config)
    print("Metadata generated successfully:")
    for k, v in metadata.items():
        print(f"  {k}: {v}")

    # Inject measured normalization stats into config so transforms don't fail
    config["preprocessing"] = {
        "mean": metadata["measured_mean"],
        "std": metadata["measured_std"],
    }

    print("\n=== Step 2: Testing DataLoader Contracts ===")
    # 2. Build smoke dataloaders
    loaders = build_dataloaders(config, smoke=True)
    
    # Fetch one batch from each loader to verify shapes and types
    for name, loader in [("Train", loaders.train), ("Validation", loaders.validation), ("Test", loaders.test)]:
        batch = next(iter(loader))
        images = batch["images"]
        labels = batch["labels"]
        sample_ids = batch["sample_ids"]

        print(f"[{name} Loader]")
        print(f"  Images shape : {images.shape} (dtype: {images.dtype})")
        print(f"  Labels shape : {labels.shape} (dtype: {labels.dtype})")
        print(f"  Sample IDs   : {sample_ids[:2]}... (total: {len(sample_ids)})")

        # Assert contract guarantees
        assert isinstance(images, torch.Tensor)
        assert images.ndim == 4 and images.shape[1:] == (1, 28, 28)
        assert images.dtype == torch.float32
        assert isinstance(labels, torch.Tensor)
        assert labels.ndim == 1
        assert labels.dtype == torch.int64
        assert len(sample_ids) == len(images)

    print("\n=== Step 3: Generating EDA Artifacts ===")
    # 3. Generate figures and stats
    output_dir = Path("runs/eda_smoke")
    generate_eda(config, output_dir=output_dir)
    print(f"EDA generated in '{output_dir}' and 'docs/assets/a1'")

    print("\nAll pipeline and contract checks passed successfully!")

if __name__ == "__main__":
    main()