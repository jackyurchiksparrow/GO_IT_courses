# IMPORT LIBRARIES

import os

os.environ["KERAS_BACKEND"] = "torch"

import base64
import io
import json
from functools import cache, lru_cache
from pathlib import Path

import keras
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from PIL import Image, ImageOps
from plotly.subplots import make_subplots

# --- CONSTANTS ---
from train_model_part_1 import CLASS_NAMES
from train_model_part_3 import IMG_SIZE, prepare_inputs  # the exact (N,28,28) -> (N,96,96,1) bilinear resize used in training

BASE_DIR = Path(__file__).resolve().parent

# Training curves copied from the module 13 notebook outputs (goit-ds-hw-13/main.ipynb). They only change when the model is retrained.
# "valid_run" - the run on the 90/10 split that picked the number of epochs
# "full_train" - the final fit on the full train set (= how the saved model was produced), plus its test scores.
NOTEBOOK_HISTORY = {
    "part1": {
        # (best epoch 15)
        "valid_run": {
            "loss":         [0.4505, 0.3012, 0.2597, 0.2323, 0.2129, 0.1959, 0.1800, 0.1685, 0.1546, 0.1417, 0.0945, 0.0839, 0.0780, 0.0735, 0.0689, 0.0626, 0.0606, 0.0606, 0.0598, 0.0593, 0.0587, 0.0579, 0.0573, 0.0576],
            "val_loss":     [0.4224, 0.3227, 0.3352, 0.3124, 0.3596, 0.2543, 0.2768, 0.2914, 0.2492, 0.2129, 0.1816, 0.1784, 0.1799, 0.1787, 0.1827, 0.1764, 0.1771, 0.1767, 0.1765, 0.1767, 0.1767, 0.1772, 0.1772, 0.1774],
            "accuracy":     [0.8383, 0.8913, 0.9066, 0.9163, 0.9228, 0.9304, 0.9352, 0.9392, 0.9442, 0.9494, 0.9699, 0.9745, 0.9766, 0.9781, 0.9800, 0.9824, 0.9839, 0.9837, 0.9836, 0.9839, 0.9842, 0.9850, 0.9853, 0.9844],
            "val_accuracy": [0.8437, 0.8838, 0.8807, 0.8895, 0.8767, 0.9117, 0.9040, 0.8908, 0.9130, 0.9268, 0.9375, 0.9360, 0.9358, 0.9375, 0.9363, 0.9395, 0.9393, 0.9397, 0.9398, 0.9402, 0.9398, 0.9393, 0.9395, 0.9380],
        },
        "full_train": {
            "loss":     [0.4425, 0.2935, 0.2554, 0.2288, 0.2090, 0.1910, 0.1785, 0.1638, 0.1501, 0.1372, 0.0931, 0.0809, 0.0748, 0.0707, 0.0657, 0.0592],
            "accuracy": [0.8399, 0.8949, 0.9086, 0.9170, 0.9250, 0.9317, 0.9354, 0.9416, 0.9458, 0.9512, 0.9693, 0.9753, 0.9775, 0.9788, 0.9809, 0.9834],
            "test_loss": 0.1877,
            "test_accuracy": 0.9344,
        },
    },
    # (best epoch 8)
    "part3": {
        "valid_run": {
            "loss":         [0.5616, 0.2977, 0.2455, 0.2134, 0.1894, 0.1719, 0.1554, 0.1362, 0.0754, 0.0480, 0.0311, 0.0189, 0.0088, 0.0072],
            "val_loss":     [0.3734, 0.2870, 0.2797, 0.2515, 0.2174, 0.2261, 0.2349, 0.2005, 0.1821, 0.2035, 0.2287, 0.2634, 0.2754, 0.2772],
            "accuracy":     [0.7981, 0.8905, 0.9099, 0.9219, 0.9304, 0.9366, 0.9430, 0.9493, 0.9724, 0.9835, 0.9897, 0.9943, 0.9984, 0.9988],
            "val_accuracy": [0.8632, 0.8928, 0.9012, 0.9062, 0.9210, 0.9203, 0.9117, 0.9298, 0.9412, 0.9383, 0.9360, 0.9365, 0.9367, 0.9370],
        },
        "full_train": {
            "loss":     [0.5379, 0.2879, 0.2385, 0.2075, 0.1846, 0.1664, 0.1487, 0.1348, 0.0763],
            "accuracy": [0.8037, 0.8943, 0.9117, 0.9236, 0.9319, 0.9397, 0.9449, 0.9506, 0.9728],
            "test_loss": 0.1922,
            "test_accuracy": 0.9406,
        },
    },
}  # fmt: skip

# Models to choose from.
MODELS = {
    "part1": {
        "label": "Part 1 - own CNN (4 conv blocks, 28×28 input, ~1.3M params)",
        "path": BASE_DIR / "task1_final_full_train.keras",
        "history_path": BASE_DIR / "task1_final_full_train_history.json",  # written by train_model_part_1.py
        "upscale_to_96": False,  # whether the model expects the (96, 96, 1) input prepared by `prepare_inputs`
    },
    "part3": {
        "label": "Part 3 - MobileNetV2 from scratch (96×96 input, ~2.3M params)",
        "path": BASE_DIR / "task3_final_full_train.keras",
        "history_path": BASE_DIR / "task3_final_full_train_history.json",  # written by train_model_part_3.py
        "upscale_to_96": True,  # whether the model expects the (96, 96, 1) input prepared by `prepare_inputs`
    },
}


# ----- model / data helpers -----
@cache  # only execute the function once and keep the results in RAM. If called again - returns from cache instead of executing the function again; entries aren't limited for scalability, but basically maxsize=2 here
def load_model(key: str):
    """Loads a model (Keras 3 native format)."""
    spec = MODELS[key]
    return keras.models.load_model(spec["path"], safe_mode=spec.get("safe_mode", True))


@lru_cache(maxsize=1)  # explicitly allocate only 1 cache entry (it only needs to save the data once, no less no more)
def load_test_set():
    """Fashion-MNIST test split - for the "random test image" button."""
    (_X_train, _y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    return X_test, y_test


def load_history(key: str):
    """Training history of the saved model: the JSON written by the training script if it exists, otherwise the notebook values."""
    nb = NOTEBOOK_HISTORY[key]
    path = MODELS[key]["history_path"]
    if path.exists():
        with open(path) as f:
            return nb["valid_run"], json.load(f), f"training script ({path.name})"
    return nb["valid_run"], nb["full_train"], "main.py (pre-saved from module 13 notebook)"


def to_fashion_mnist(img: Image.Image, invert_mode: str) -> np.ndarray:
    """Turns an arbitrary picture into a Fashion-MNIST-like sample: 28x28 grayscale, light item on a black background, float32 in [0, 255] (no scaling - the Normalization layer inside the model does it)."""
    img = ImageOps.exif_transpose(img)  # phone photos may be stored rotated
    if img.mode in ("RGBA", "LA", "P"):  # transparent background -> white (as on a typical catalog picture)
        rgba = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
    gray = img.convert("L")

    # Fashion-MNIST items are light on black; most photos are dark-ish items on a light background -> invert those
    arr = np.asarray(gray, dtype=np.float32)
    border_mean = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]]).mean()
    if invert_mode == "always" or (invert_mode == "auto" and border_mean > 127):
        gray = ImageOps.invert(gray)

    # Keep the aspect ratio: fit into 28x28 and pad with black, the item stays centered as in the dataset
    gray = ImageOps.pad(gray, (28, 28), method=Image.Resampling.LANCZOS, color=0)
    return np.asarray(gray, dtype=np.float32)


def predict_proba(key: str, x28: np.ndarray) -> np.ndarray:
    """Softmax probabilities (10,) for one 28x28 image."""
    x = x28[None, ...]  # (1, 28, 28)
    if MODELS[key]["upscale_to_96"]:
        x = prepare_inputs(x, IMG_SIZE)  # (1, 96, 96, 1) - same preprocessing as FashionMNISTDataset(img_size=96)
    return np.asarray(load_model(key).predict(x, verbose=0)[0])


def decode_upload(contents: str) -> Image.Image:
    """dcc.Upload gives 'data:image/png;base64,....' -> PIL image."""
    _header, b64 = contents.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(b64)))


# ----- figures -----
def history_figure(key: str):
    """Plotly version of my plot_loss_acc_per_ep(): Loss and Accuracy side by side, train vs validation."""
    valid_run, full_train, source = load_history(key)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"))
    for col, metric in ((1, "loss"), (2, "accuracy")):
        show = col == 1  # one legend entry per line
        fig.add_trace(go.Scatter(y=valid_run[metric], mode="lines+markers", name="train (90/10 split)", line_color="#1f77b4", legendgroup="t", showlegend=show), 1, col)
        fig.add_trace(go.Scatter(y=valid_run[f"val_{metric}"], mode="lines+markers", name="validation (90/10 split)", line_color="#ff7f0e", legendgroup="v", showlegend=show), 1, col)
        fig.add_trace(go.Scatter(y=full_train[metric], mode="lines+markers", name="final model: full train", line={"color": "#2ca02c", "dash": "dash"}, legendgroup="f", showlegend=show), 1, col)
    fig.update_xaxes(title_text="Epoch", dtick=1 if len(valid_run["loss"]) <= 16 else 2)  # integer epochs (as MaxNLocator(integer=True))
    fig.update_layout(height=380, margin={"l": 40, "r": 20, "t": 40, "b": 40}, legend={"orientation": "h", "y": -0.25})

    test_txt = f"Test accuracy: {full_train['test_accuracy'] * 100:.2f}%  ·  Test loss: {full_train['test_loss']:.4f}  ·  source: {source}"
    return fig, test_txt


def to_png_uri(x28: np.ndarray) -> str:
    """The 28x28 array exactly as the model receives it -> upscaled PNG (NEAREST keeps the pixels sharp) for an <img>."""
    buf = io.BytesIO()
    Image.fromarray(x28.astype(np.uint8)).resize((224, 224), Image.Resampling.NEAREST).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def proba_figure(proba: np.ndarray):
    """Horizontal bar chart of the class probabilities, the predicted class highlighted."""
    pred = int(proba.argmax())
    colors = ["#4f46e5" if i == pred else "#d9dcf7" for i in range(len(CLASS_NAMES))]
    fig = go.Figure(go.Bar(x=proba * 100, y=CLASS_NAMES, orientation="h", marker_color=colors, text=[f"{p * 100:.1f}%" for p in proba], textposition="outside", cliponaxis=False))
    fig.update_layout(
        template="plotly_white",
        height=360,
        margin={"l": 10, "r": 50, "t": 10, "b": 30},
        xaxis={"range": [0, 100], "showticklabels": False, "showgrid": False, "zeroline": False},
        yaxis={"autorange": "reversed"},
        bargap=0.35,
        font={"family": "Segoe UI, system-ui, Arial", "size": 13},
    )
    return fig


# ----- layout -----
# Styling lives in assets/style.css (Dash serves and links it automatically); here only class names are assigned
available = [k for k, spec in MODELS.items() if spec["path"].exists()]
PLACEHOLDER = html.Div("Upload an image or take a random one from the test set", className="placeholder")


def model_option(spec):
    """Card-like label for the model selector: "Part 1 - own CNN (details)" -> bold title + muted details line."""
    title, _, details = spec["label"].partition(" (")
    return html.Div([html.Div(title, className="model-title"), html.Div(details.rstrip(")"), className="model-sub")])


app = Dash(__name__, title="Fashion-MNIST classifier")
app.layout = html.Div(
    className="page",
    children=[
        html.H1("Fashion-MNIST classifier"),
        html.P("Module 13 models: upload a clothing image and see what each network thinks it is", className="subtitle"),
        # --- model choice + training curves ---
        html.Div(
            className="card",
            children=[
                html.Div("Model", className="section-title"),
                dcc.RadioItems(
                    id="model",
                    className="model-cards",
                    options=[{"label": model_option(spec), "value": k, "disabled": k not in available} for k, spec in MODELS.items()],
                    value=available[0] if available else None,
                ),
                dcc.Graph(id="history-graph"),
                html.Div(id="test-metrics", className="muted"),
            ],
        ),
        # --- input image (left) + prediction (right) ---
        html.Div(
            className="card classify",
            children=[
                html.Div(
                    children=[
                        html.Div("Input", className="section-title"),
                        dcc.Upload(
                            id="upload",
                            accept="image/*",
                            className="dropzone",
                            children=[html.Div("🖼️", className="dropzone-icon"), html.Div(["Drag & drop or ", html.B("browse")]), html.Div("PNG, JPG…", className="muted")],
                        ),
                        html.Button("🎲  Random test-set image", id="sample-btn", n_clicks=0, className="btn"),
                        html.Div(
                            className="row",
                            children=[
                                html.Span("Invert colors", className="muted", title="Fashion-MNIST has light items on a black background"),
                                dcc.RadioItems(
                                    id="invert",
                                    className="segmented",
                                    options=[{"label": "Auto", "value": "auto"}, {"label": "Always", "value": "always"}, {"label": "Never", "value": "never"}],
                                    value="auto",
                                    inline=True,
                                ),
                            ],
                        ),
                        html.Div(
                            className="previews",
                            children=[
                                html.Div([html.Div(html.Img(id="original-img"), className="tile"), html.Div("Your image", className="caption")]),
                                html.Div([html.Div(html.Img(id="input-img"), className="tile tile-dark"), html.Div("Model input, 28×28", className="caption")]),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    children=[
                        html.Div("Prediction", className="section-title"),
                        html.Div(id="prediction-text", className="prediction", children=PLACEHOLDER),
                        dcc.Loading(dcc.Graph(id="proba-graph", config={"displayModeBar": False}, style={"display": "none"}), color="#4f46e5"),
                    ],
                ),
            ],
        ),
        dcc.Store(id="image-store"),  # what is currently shown: an uploaded picture or a test sample
    ],
)


# ----- callbacks -----
@app.callback(Output("history-graph", "figure"), Output("test-metrics", "children"), Input("model", "value"))
def update_history(key):
    if key is None:
        return go.Figure(), "No model files found next to main.py"
    return history_figure(key)


@app.callback(
    Output("image-store", "data"),
    Input("upload", "contents"),
    Input("sample-btn", "n_clicks"),
    State("upload", "filename"),
    prevent_initial_call=True,
)
def set_image(contents, _n_clicks, filename):
    if ctx.triggered_id == "sample-btn":
        X_test, y_test = load_test_set()
        i = int(np.random.default_rng().integers(len(X_test)))  # deliberately unseeded: a new image on every click
        return {"kind": "sample", "index": i, "true": int(y_test[i])}
    if contents is None:
        return no_update
    return {"kind": "upload", "contents": contents, "filename": filename}


@app.callback(
    Output("original-img", "src"),
    Output("input-img", "src"),
    Output("proba-graph", "figure"),
    Output("proba-graph", "style"),
    Output("prediction-text", "children"),
    Input("image-store", "data"),
    Input("model", "value"),
    Input("invert", "value"),
    prevent_initial_call=True,
)
def classify(data, key, invert_mode):
    if not data or key is None:
        return no_update, no_update, no_update, no_update, no_update

    true_label = None
    if data["kind"] == "sample":  # already in the dataset format -> no preprocessing
        X_test, _y = load_test_set()
        x28 = X_test[data["index"]].astype(np.float32)
        true_label = CLASS_NAMES[data["true"]]
        src = to_png_uri(x28)
    else:
        try:
            x28 = to_fashion_mnist(decode_upload(data["contents"]), invert_mode)
        except Exception as e:  # not an image / corrupted file
            return None, None, no_update, {"display": "none"}, html.Div(f"Can't read the file: {e}", className="placeholder")
        src = data["contents"]

    proba = predict_proba(key, x28)
    pred = int(proba.argmax())
    headline = [html.Span(CLASS_NAMES[pred], className="pred-class"), html.Span(f"{proba[pred] * 100:.1f}%", className="badge")]
    if true_label is not None:  # test-set sample: show whether the model got it right
        ok = true_label == CLASS_NAMES[pred]
        headline.append(html.Div(["True label: ", html.B(true_label), html.Span("correct" if ok else "wrong", className=f"badge {'ok' if ok else 'bad'}")], className="muted"))
    return src, to_png_uri(x28), proba_figure(proba), {"display": "block"}, headline


# Run:  python main.py  ->  open http://127.0.0.1:8050
if __name__ == "__main__":
    print("Launching...")
    # use_reloader=False means the app doesn't restart when the code is saved
    app.run(debug=True, use_reloader=False)
