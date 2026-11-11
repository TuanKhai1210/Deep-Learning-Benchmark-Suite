import sys
import csv
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to PYTHONPATH so dlbench can be imported
sys.path.insert(0, str(Path("src").resolve()))

from dlbench.common.config import load_config
from dlbench.a1.data.dataset import prepare_data
from dlbench.a1.data.eda import generate_eda
from dlbench.a1.data.loaders import build_dataloaders
from dlbench.a1.trainer import fit, evaluate_checkpoint

def main():
    print("Starting Comprehensive E2E Smoke Test Pipeline (Python-native API)\n")
    
    config_file = Path("configs/a1/models/mlp.py")
    
    print("--- Step 0: Load Config ---")
    config = load_config(config_file)
    print("Config loaded successfully.")

    # 1. Prepare Data
    print("\n--- Step 1: Data Preparation ---")
    metadata = prepare_data(config)
    print("Metadata generated successfully:")
    for k, v in metadata.items():
        print(f"  {k}: {v}")
    
    # 2. Generate EDA Artifacts
    print("\n--- Step 2: Generating EDA Artifacts ---")
    output_dir = Path("runs/eda_smoke")
    generate_eda(config, output_dir=output_dir)
    print(f"EDA generated in {output_dir}")

    # 3. Verify Dataloaders & Transformations
    print("\n--- Step 3: Verifying Transformations & Dataloaders ---")
    loaders = build_dataloaders(config, smoke=True)
    batch = next(iter(loaders.train))
    print(f"Train batch images shape: {batch['images'].shape}")
    print(f"Train batch labels shape: {batch['labels'].shape}")
    print("Transformations applied successfully!")
    
    # 4. Train (Smoke Mode)
    print("\n--- Step 4: Training (Smoke Mode) ---")
    fit_result = fit(config, smoke=True)
    run_dir = fit_result.run_dir
    print(f"Completed run: {run_dir}")
    print(f"Detected run directory: {run_dir}")
    
    checkpoint_path = fit_result.best_checkpoint
    if not checkpoint_path.exists():
        print(f"Expected checkpoint not found at {checkpoint_path}")
        sys.exit(1)
        
    # 5. Evaluate (Validation Split)
    print("\n--- Step 5: Evaluation (Validation Split) ---")
    eval_result = evaluate_checkpoint(config, checkpoint_path, split="validation", smoke=True)
    print(eval_result.metrics)

    # 6. Makeshift Plot of History File
    print("\n--- Step 6: Plotting History (Makeshift) ---")
    history_file = run_dir / "history.csv"
    if not history_file.exists():
        print(f"No history file found at {history_file}")
        sys.exit(1)

    epochs, val_loss, val_acc, val_f1 = [], [], [], []
    train_loss, train_acc, train_f1 = [], [], []
    with history_file.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            val_loss.append(float(row["val_loss"]))
            val_acc.append(float(row["val_accuracy"]))
            val_f1.append(float(row["val_macro_f1"]))
            train_loss.append(float(row["train_loss"]))
            train_acc.append(float(row["train_accuracy"]))
            train_f1.append(float(row["train_macro_f1"]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(epochs, train_loss, label='Train Loss')
    axes[0].plot(epochs, val_loss, label='Validation Loss')
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label='Train Accuracy')
    axes[1].plot(epochs, val_acc, label='Validation Accuracy')
    axes[1].set_title("Accuracy")
    axes[1].legend()

    axes[2].plot(epochs, train_f1, label='Train Macro F1')
    axes[2].plot(epochs, val_f1, label='Validation Macro F1')
    axes[2].set_title("Macro F1")
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True)

    out_path = run_dir / "makeshift_learning_curves.png"
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Makeshift learning curves saved to {out_path}")
    
    print("\nAll End-to-End Smoke Test steps completed successfully!")

if __name__ == "__main__":
    main()
