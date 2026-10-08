# train MobileNetV2 from scratch, weights=None (part 3)
import json
import os
import random

os.environ["KERAS_BACKEND"] = "torch"

import keras
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, random_split

# --- CONSTANTS ---
RANDOM_SEED = 42
IMG_SIZE = 96
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def seed_everything(seed=42):
    """Fixes seeds to reproduce outcomes. Returns a NumPy random generator."""
    # fmt: off
    random.seed(seed)                                       # seeds Python's built-in "random" module
    np.random.seed(seed)                                    # legacy global stream (some libs use it)
    keras.utils.set_random_seed(seed)                       # python + numpy + torch backend
    os.environ["PYTHONHASHSEED"] = str(seed)                # seeds child processes of DataLoaders with multiple background workers (num_workers > 0)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)                        # if using multi-GPU

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"       # gives cuBLAS (the matrix-multiplication library) a chunk of memory to store intermediate sums
    torch.backends.cudnn.benchmark = False                  # forces to use only one hard-coded CNN algorithm instead of searching for the fastest every time
    torch.backends.cudnn.deterministic = True               # makes the previous line choose the algorithm that doesn't rely on atomic addition (slows training by ~14%; scales even more for heavy nets)
    torch.use_deterministic_algorithms(True)                # forces every other single PyTorch function to use a deterministic path (scatter, gather, interpolations, pooling etc.)

    rng = np.random.default_rng(seed)                       # # isolated Generator to pass around explicitly
    return rng


def plot_loss_acc_per_ep(train_losses, valid_losses, train_accuracies, valid_accuracies):
    """Plots the metrics for training and validation performance comparison."""
    print("Smoothed scores over epochs grouped by folds:")
    for ep, (tl, vl, ta, va) in enumerate(zip(train_losses, valid_losses, train_accuracies, valid_accuracies)):
        print(f"Epoch {ep:>2}: train_loss={tl:.4f}  val_loss={vl:.4f}  train_acc={ta:.4f}  val_acc={va:.4f}")

    print()
    _fix, axes = plt.subplots(ncols=2, figsize=(15, 4))

    # --- first plot: train and validation losses ---
    axes[0].plot(np.arange(len(train_losses)), train_losses, ".-")
    axes[0].plot(np.arange(len(valid_losses)), valid_losses, ".-")
    axes[0].legend(["train", "validation"])
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))  # Forces integer
    axes[0].grid()

    # --- second plot: train and validation accuracies ---
    axes[1].plot(np.arange(len(train_accuracies)), train_accuracies, ".-")
    axes[1].plot(np.arange(len(valid_accuracies)), valid_accuracies, ".-")
    axes[1].legend(["train", "validation"])
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))  # Forces integer
    axes[1].grid()


# --- one-time input prep: (N, 28, 28) -> (N, 96, 96, 1) float32 ---
def prepare_inputs(X, img_size=IMG_SIZE, chunk=2000):
    out = []
    for i in range(0, len(X), chunk):
        x = keras.ops.convert_to_tensor(X[i : i + chunk], dtype="float32")
        x = keras.ops.reshape(x, (-1, 28, 28, 1))
        x = keras.ops.image.resize(x, (img_size, img_size), interpolation="bilinear")
        out.append(keras.ops.convert_to_numpy(x))
    return np.concatenate(out, axis=0)


class FashionMNISTDataset(Dataset):
    def __init__(self, data: np.ndarray, targets: np.ndarray, img_size: int | None = None):
        if img_size is not None:  # one-time resize (N,28,28) -> (N,H,W,1)
            data = prepare_inputs(data, img_size)
        self.data = torch.from_numpy(data).clone().float()
        self.targets = torch.from_numpy(targets).clone().int()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def build_model(steps, torch_gen, normalizer):
    keras.utils.set_random_seed(RANDOM_SEED)
    torch_gen.manual_seed(RANDOM_SEED)

    net = keras.applications.MobileNetV2(
        weights=None,
        include_top=True,
        input_shape=(IMG_SIZE, IMG_SIZE, 1),
        classes=10,
    )

    for layer in net.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.momentum = 0.9

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = normalizer(inputs)
    outputs = net(x)
    model = keras.Model(inputs, outputs)

    base = 0.001
    optimizer_obj = keras.optimizers.RMSprop(learning_rate=keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[steps * 8, steps * 12], values=[base, base * 0.1, base * 0.01]))

    model.compile(
        optimizer=optimizer_obj,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    # --- SEEDING FUNCTIONALITY ---
    _rng = seed_everything(RANDOM_SEED)
    torch_gen = torch.Generator()
    keras.utils.set_random_seed(RANDOM_SEED)
    torch_gen.manual_seed(RANDOM_SEED)
    keras.backend.clear_session()
    torch.cuda.empty_cache()

    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

    X_train_writable = X_train.copy()
    y_train_writable = y_train.copy()

    X_test_writable = X_test.copy()
    y_test_writable = y_test.copy()

    full_train_ds_96 = FashionMNISTDataset(X_train_writable, y_train_writable, img_size=IMG_SIZE)
    test_ds_96 = FashionMNISTDataset(X_test_writable, y_test_writable, img_size=IMG_SIZE)

    full_train_ds = FashionMNISTDataset(X_train_writable, y_train_writable)

    # Split train into train/validation (90/10)
    train_size = int(0.9 * len(X_train))
    val_size = len(full_train_ds) - train_size

    train_ds, _valid_ds = random_split(
        dataset=full_train_ds,
        lengths=[train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, generator=torch_gen)

    # Calculate the whole dataset's mean and std in batches (not needed to do it in batches for such a small dataset though)
    # (z-score normalization)
    n = 0
    s = 0.0
    ss = 0.0
    for x, _ in train_loader:
        s += x.sum().item()
        ss += (x**2).sum().item()
        n += x.numel()

    mean_ds = s / n
    std_ds = (ss / n - mean_ds**2) ** 0.5
    variance_ds = std_ds**2

    # Hardcode the stats into Keras
    normalizer = keras.layers.Normalization(
        axis=None,
        mean=mean_ds,
        variance=variance_ds,
    )

    # Fit on full train before assessing on test:
    keras.utils.set_random_seed(RANDOM_SEED)
    torch_gen.manual_seed(RANDOM_SEED)
    keras.backend.clear_session()
    torch.cuda.empty_cache()

    full_train_loader = DataLoader(full_train_ds_96, batch_size=125, shuffle=True, generator=torch_gen)
    test_loader_96 = DataLoader(test_ds_96, batch_size=125, shuffle=False)

    steps = len(full_train_loader)
    model = build_model(steps, torch_gen, normalizer)

    history = model.fit(
        x=full_train_loader,
        epochs=9,
        verbose=1,
        shuffle=False,
    )

    plot_loss_acc_per_ep(history.history["loss"], [], history.history["accuracy"], [])

    test_loss, test_acc = model.evaluate(test_loader_96, verbose=1)

    print("\n" + "=" * 40)
    print(f"FINAL TEST ACCURACY: {test_acc * 100:.2f}%")
    print(f"FINAL TEST LOSS:     {test_loss:.4f}")
    print("=" * 40)

    # --- Save (Keras 3 native: architecture + weights + optimizer state in one file) ---
    model.save("task3_final_full_train.keras")

    # --- Save the training history + test scores (for the Dash app to plot) ---
    with open("task3_final_full_train_history.json", "w") as f:
        json.dump(
            {**{k: [float(v) for v in vals] for k, vals in history.history.items()}, "test_loss": float(test_loss), "test_accuracy": float(test_acc)},
            f,
            indent=2,
        )

    # --- Load ---
    # loaded = keras.models.load_model("task3_final_full_train.keras")

    # --- 1. Predictions + true labels out of the DataLoader ---
    y_prob = model.predict(test_loader_96, verbose=0)  # (10000, 10) softmax probabilities
    y_pred = y_prob.argmax(axis=1)  # most probable class indices
    y_true = np.concatenate([yb.numpy() for _, yb in test_loader_96])  # convert each y_batch from tensor to numpy and concatenate into 1D array

    # --- 2. Per-class metrics ---
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    # --- 3. Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred)

    _fig, ax = plt.subplots(figsize=(9, 9))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d",  # integer format
        xticks_rotation=45,
    )
    ax.set_title("Fashion-MNIST — Raw Confusion Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
