# Author: Simon M. Hofmann
# Contact: simon.hofmann@cbs.mpg.de

# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#     "marimo>=0.14.13,<0.15",
#     "numpy<=2.3,>=2.2",
#     "torch==2.7.1",
#     "zennit==0.5.1",
#     "opencv-python==4.11.0.86",
#     "matplotlib==3.10.3",
#     "pillow==11.3.0",
#     "torchvision==0.22.1",
#     "pandas==2.3.1",
#     "rsatoolbox==0.3.0",
#     "umap-learn==0.5.9.post2",
#     "altair==5.5.0",
#     "pyarrow==21.0.0",
# ]
# ///

import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from datetime import datetime
    import io
    import re
    import math
    import matplotlib.pyplot as plt
    import altair as alt
    import numpy as np
    from numpy import typing as npt
    import pandas as pd
    import rsatoolbox
    import torch
    from torch import nn
    from pathlib import Path
    from collections import OrderedDict
    import cv2
    from PIL import Image
    import urllib
    import shutil
    import zipfile
    import umap
    from sklearn.decomposition import PCA
    from zennit.attribution import Gradient
    from zennit.torchvision import VGGCanonizer
    from zennit.composites import (
        COMPOSITES,  # dict of all composite classes
    )
    from zennit.image import imgify, imsave

    return (
        COMPOSITES,
        Gradient,
        Image,
        OrderedDict,
        PCA,
        Path,
        VGGCanonizer,
        alt,
        cv2,
        datetime,
        imgify,
        imsave,
        io,
        math,
        mo,
        nn,
        np,
        npt,
        pd,
        plt,
        re,
        rsatoolbox,
        shutil,
        torch,
        umap,
        urllib,
        zipfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Exploring predictions and latent features of deep neural networks

    [Simon M. Hofmann](#contact) :: @Cognition Academy :: `July 25, 2025`
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    # Collect relevant paths
    ROOT_DIR = mo.notebook_dir()
    RESULT_DIR = ROOT_DIR / "results"
    DATA_DIR = ROOT_DIR / "data"
    MODEL_DIR = DATA_DIR / "vgg_face_torch"
    FACES_DIR = DATA_DIR / "faces"
    DOWNLOAD_DIR = ROOT_DIR / "download"

    # Create folders in case they aren't there yet
    for p in [RESULT_DIR, DATA_DIR]:
        p.mkdir(exist_ok=True)
    return DATA_DIR, DOWNLOAD_DIR, FACES_DIR, MODEL_DIR, RESULT_DIR, ROOT_DIR


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Download image data and model weights""")
    return


@app.cell(hide_code=True)
def _(FACES_DIR, MODEL_DIR):
    REQUIRED_FILES = [
        MODEL_DIR / "vggface.pth",
        MODEL_DIR / "names.txt",
        FACES_DIR / "ak.png",
    ]

    def found_required_files(verbose: bool = False) -> bool:
        if not all(f.exists() for f in REQUIRED_FILES):
            if verbose:
                print("Not all necessary files are there")
            return False
        else:
            if verbose:
                print("Found all necessary files")
            return True

    return (found_required_files,)


@app.cell(hide_code=True)
def _(
    DATA_DIR,
    DOWNLOAD_DIR,
    ROOT_DIR,
    cprint,
    found_required_files,
    shutil,
    urllib,
    zipfile,
):
    DOWNLOAD_URL = "https://keeper.mpdl.mpg.de/f/cb07841a00db43e88b32/?dl=1"
    path_to_zipped_files = DOWNLOAD_DIR / "data.zip"
    _downloaded_and_unzipped = False

    # Download only if necessary
    if not (found_required_files(verbose=False) or path_to_zipped_files.exists()):
        try:
            print("Downloading model and image files ...")
            DOWNLOAD_DIR.mkdir(exist_ok=True)
            _, _ = urllib.request.urlretrieve(
                url=DOWNLOAD_URL, filename=path_to_zipped_files
            )  # noqa: S310
            _downloaded_and_unzipped = True
        except urllib.error.HTTPError as e:
            cprint(string=f"HTTP Error: {e.code} - {DOWNLOAD_URL} {e.reason}", col="r")

    # Unzip downloaded files and move them to the data dir
    if not found_required_files(verbose=False) and path_to_zipped_files.exists():
        print("Unzipping files ...")
        with zipfile.ZipFile(path_to_zipped_files, "r") as zip_ref:
            zip_ref.extractall(DOWNLOAD_DIR)

        print(f"Moving files to ./{DATA_DIR.relative_to(ROOT_DIR)}/")

        # Ensure destination exists (done above)
        _unzipped_data_dir = DOWNLOAD_DIR / "data"

        # Traverse all files in source (recursively)
        for src_file in _unzipped_data_dir.rglob("*"):
            if src_file.is_file():
                if src_file.suffix == ".zip":
                    # Do not move the zip file itself
                    continue

                # Compute destination path
                relative_path = src_file.relative_to(_unzipped_data_dir)
                dst_file = DATA_DIR / relative_path

                # Create destination parent directories
                dst_file.parent.mkdir(parents=True, exist_ok=True)

                # Only move if the file doesn't already exist
                if not dst_file.exists():
                    shutil.move(str(src_file), str(dst_file))
                else:
                    print(f"🔁 Skipped existing file: {dst_file.relative_to(ROOT_DIR)}")

        _downloaded_and_unzipped = True

    # Remove download folder
    if found_required_files(verbose=False) and _downloaded_and_unzipped:
        print(f"Clean up and delete ./{DOWNLOAD_DIR.relative_to(ROOT_DIR)}/")
        shutil.rmtree(DOWNLOAD_DIR)
    return DOWNLOAD_URL, path_to_zipped_files


@app.cell(hide_code=True)
def _(DATA_DIR, DOWNLOAD_URL, found_required_files, mo, path_to_zipped_files):
    download_button = mo.download(
        data=DOWNLOAD_URL,
        filename=str(path_to_zipped_files),
        label="Download required files",
        disabled=found_required_files(verbose=False),
    )

    def _alternative_download():
        if not found_required_files(verbose=False):
            return mo.vstack(
                [
                    mo.md(
                        f"""
                        ### Automatic download above does not work for me ?!

                        Click the download button below!

                        Then move the unpacked files to:

                        *{DATA_DIR}*
                        """
                    ),
                    download_button,
                ],
                gap=2,
            )
        else:
            return mo.md("")

    def _check_data():
        if not found_required_files():
            return mo.stop(
                predicate=not found_required_files(verbose=False),
                output=mo.vstack(
                    [
                        mo.md("""## Please download all required files to continue"""),
                        _alternative_download(),
                    ],
                    gap=3,
                    align="center",
                ),
            )
        else:
            return mo.md(
                """### Ready to go!

                All required image and model files were found.
                """
            )

    _check_data()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Get the model

    ### Load VGG-Face

    [VGG-Face](https://www.robots.ox.ac.uk/~vgg/software/vgg_face/) is a convolutional neural network (CNN) that has
    been trained to predict face identities (i.e., celebrities; $N_{identities} = 2,622$) from a large image dataset.

    **Question**: Could VGG-Face be a model of human face perception? Why (not)?
    """
    )
    return


@app.cell(hide_code=True)
def _(MODEL_DIR, OrderedDict, nn, torch):
    class VGGFace(nn.Module):
        """
        VGGFace class.

        This is an reimplementation of the original `VGG-Face` model in `PyTorch`.

        *Source: https://github.com/chi0tzp/PyVGGFace/blob/master/lib/vggface.py.*
        """

        def __init__(self, save_layer_output: bool = False) -> None:
            """
            Initialize VGGFace model.

            :param save_layer_output: If True, save the output of each layer in a list.
            :return: None
            """
            super().__init__()

            self.save_layer_output = save_layer_output
            self.layer_output = []
            self._layer_names = []

            self.features = nn.ModuleDict(
                OrderedDict(
                    {
                        # === Block 1 ===
                        "conv_1_1": nn.Conv2d(
                            in_channels=3, out_channels=64, kernel_size=3, padding=1
                        ),
                        "relu_1_1": nn.ReLU(inplace=True),
                        "conv_1_2": nn.Conv2d(
                            in_channels=64, out_channels=64, kernel_size=3, padding=1
                        ),
                        "relu_1_2": nn.ReLU(inplace=True),
                        "maxp_1_2": nn.MaxPool2d(kernel_size=2, stride=2),
                        # === Block 2 ===
                        "conv_2_1": nn.Conv2d(
                            in_channels=64, out_channels=128, kernel_size=3, padding=1
                        ),
                        "relu_2_1": nn.ReLU(inplace=True),
                        "conv_2_2": nn.Conv2d(
                            in_channels=128, out_channels=128, kernel_size=3, padding=1
                        ),
                        "relu_2_2": nn.ReLU(inplace=True),
                        "maxp_2_2": nn.MaxPool2d(kernel_size=2, stride=2),
                        # === Block 3 ===
                        "conv_3_1": nn.Conv2d(
                            in_channels=128, out_channels=256, kernel_size=3, padding=1
                        ),
                        "relu_3_1": nn.ReLU(inplace=True),
                        "conv_3_2": nn.Conv2d(
                            in_channels=256, out_channels=256, kernel_size=3, padding=1
                        ),
                        "relu_3_2": nn.ReLU(inplace=True),
                        "conv_3_3": nn.Conv2d(
                            in_channels=256, out_channels=256, kernel_size=3, padding=1
                        ),
                        "relu_3_3": nn.ReLU(inplace=True),
                        "maxp_3_3": nn.MaxPool2d(
                            kernel_size=2, stride=2, ceil_mode=True
                        ),
                        # === Block 4 ===
                        "conv_4_1": nn.Conv2d(
                            in_channels=256, out_channels=512, kernel_size=3, padding=1
                        ),
                        "relu_4_1": nn.ReLU(inplace=True),
                        "conv_4_2": nn.Conv2d(
                            in_channels=512, out_channels=512, kernel_size=3, padding=1
                        ),
                        "relu_4_2": nn.ReLU(inplace=True),
                        "conv_4_3": nn.Conv2d(
                            in_channels=512, out_channels=512, kernel_size=3, padding=1
                        ),
                        "relu_4_3": nn.ReLU(inplace=True),
                        "maxp_4_3": nn.MaxPool2d(kernel_size=2, stride=2),
                        # === Block 5 ===
                        "conv_5_1": nn.Conv2d(
                            in_channels=512, out_channels=512, kernel_size=3, padding=1
                        ),
                        "relu_5_1": nn.ReLU(inplace=True),
                        "conv_5_2": nn.Conv2d(
                            in_channels=512, out_channels=512, kernel_size=3, padding=1
                        ),
                        "relu_5_2": nn.ReLU(inplace=True),
                        "conv_5_3": nn.Conv2d(
                            in_channels=512, out_channels=512, kernel_size=3, padding=1
                        ),
                        "relu_5_3": nn.ReLU(inplace=True),
                        "maxp_5_3": nn.MaxPool2d(kernel_size=2, stride=2),
                    }
                )
            )

            self.fc = nn.ModuleDict(
                OrderedDict(
                    {
                        "fc6": nn.Linear(in_features=512 * 7 * 7, out_features=4096),
                        "fc6-relu": nn.ReLU(inplace=True),
                        "fc6-dropout": nn.Dropout(p=0.5),
                        "fc7": nn.Linear(in_features=4096, out_features=4096),
                        "fc7-relu": nn.ReLU(inplace=True),
                        "fc7-dropout": nn.Dropout(p=0.5),
                        "fc8": nn.Linear(in_features=4096, out_features=2622),
                    }
                )
            )

        def reset_layer_output(self):
            """Reset the layer output list (i.e., set it to an empty list)."""
            self.layer_output = []

        @property
        def layer_names(self):
            """Return list of layer names in `VGGFace`."""
            if self._layer_names:
                return self._layer_names

            for child in self.children():
                for layer in child:
                    self._layer_names.append(str(layer))
            return self._layer_names

        def forward(self, x):
            """Run forward pass through the model `VGGFace`."""
            if self.save_layer_output:
                self.reset_layer_output()

            # Forward through feature layers
            for layer in self.features.values():
                x = layer(x)
                # Append layer output to list
                if self.save_layer_output:
                    self.layer_output.append(x)

            # Flatten convolution outputs
            x = x.view(x.size(0), -1)

            # Forward through FC layers
            if hasattr(self, "fc"):  # check for later cut-off models
                for layer in self.fc.values():
                    x = layer(x)
                    if self.save_layer_output:
                        self.layer_output.append(x)

            return x

    def load_trained_vgg_weights_into_model(model: VGGFace) -> VGGFace:
        """Load trained weights into the original `VGG-Face` model."""
        model_dict = torch.load(
            MODEL_DIR / "vggface.pth",
            map_location=lambda storage, loc: storage,  # noqa: ARG005
        )
        model.load_state_dict(model_dict)
        return model

    def get_vgg_face_model(save_layer_output: bool) -> VGGFace:
        """Get the originally trained `VGGFace` model."""
        vgg_model = VGGFace(save_layer_output=save_layer_output).double()
        return load_trained_vgg_weights_into_model(vgg_model)

    # Load model
    vgg_face = get_vgg_face_model(save_layer_output=True)

    # Tell mode that we use it only for predictions (no training)
    _ = vgg_face.eval()
    return VGGFace, vgg_face


@app.cell(hide_code=True)
def _(mo):
    # Get the trained VGG-Face model
    show_model = mo.ui.switch(label="Show model")
    show_model

    return (show_model,)


@app.cell(hide_code=True)
def _(show_model, vgg_face):
    def _show_model():
        if show_model.value:
            return vgg_face

    _show_model()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Load image data for the model

    For the following analysis, we choose one example image.
    You can load / use your own image(s) – that makes this a bit more fun.
    """
    )
    return


@app.cell(hide_code=True)
def _(Path, cv2, mo, np, torch):
    IMG_SIZE = (224, 224)
    FILETYPES = [".png", ".jpg", ".jpeg"]

    def load_image_for_model(
        image_path: str | Path, dtype: torch.float64, subtract_mean: bool = True
    ) -> torch.Tensor:
        """Load an image for the `VGG` model."""
        image = cv2.imread(str(image_path))
        mo.stop(
            predicate=image.shape[0] != image.shape[1],
            output=mo.md("""### Please submit only square images!"""),
        )
        # if image.shape[0] != image.shape[1]:
        #     raise ValueError("Please submit only square images.")

        image = cv2.resize(image, dsize=IMG_SIZE)
        image = torch.Tensor(image).permute(2, 0, 1).view(1, 3, *IMG_SIZE).to(dtype)
        if subtract_mean:
            # this subtraction should be the average pixel value of the training set of the original VGGFace
            image -= (
                torch.Tensor(np.array([129.1863, 104.7624, 93.5940]))
                .to(dtype)
                .view(1, 3, 1, 1)
            )
        return image

    return FILETYPES, IMG_SIZE, load_image_for_model


@app.cell(hide_code=True)
def _(FILETYPES, mo):
    upload_pic = mo.ui.file(
        filetypes=FILETYPES,
        kind="area",  # "button",
        label="Provide **your own picture**",
        multiple=False,
    )
    return (upload_pic,)


@app.cell(hide_code=True)
def _(FACES_DIR, FILETYPES, ROOT_DIR, mo, upload_pic):
    file_browser = mo.ui.file_browser(
        initial_path=FACES_DIR,
        filetypes=FILETYPES,
        multiple=False,
        label="Choose an image",
        restrict_navigation=True,
    )

    mo.vstack(
        [
            mo.md("""### Load or choose a (square) image"""),
            mo.hstack([upload_pic, file_browser], align="center", widths=[1, 2]),
            mo.md(
                f"""
                Take a picture of you or a picture from the internet and crop it, such that: **height == width**.
                Load / drop the cropped picture in the left panel, or you place it in the folder
                './{FACES_DIR.relative_to(ROOT_DIR)}/'.
                For the latter, you might have to restart the notebook to see the image(s) in the right list.

                Allowed file types: {FILETYPES}

                **Note** : Images are *only* shared among the workshop members, and only if you actively do so
                later during the session. At this stage, all images exclusively lie on your machine.
                """
            ),
        ],
        align="center",
        gap=3,
    )
    return (file_browser,)


@app.cell(hide_code=True)
def _(mo, upload_pic):
    def _name_image_upload():
        if upload_pic.value:
            return mo.ui.text_area(
                # value=f"{datetime.now():%Y-%m-%d_%H-%M-%S}_upload.png",
                placeholder="unique_image_name.png",
                label="#### Name the uploaded image",
                max_length=25,
                full_width=False,
                rows=1,
            )
        else:
            return mo.md("...")

    image_name = _name_image_upload()
    mo.center(image_name)
    return (image_name,)


@app.cell(hide_code=True)
def _(
    FACES_DIR,
    IMG_SIZE,
    Image,
    file_browser,
    image_name,
    io,
    load_image_for_model,
    mo,
    re,
    torch,
    upload_pic,
):
    path_to_image = None

    def sanitize_filename(filename: str) -> str:
        """
        Sanitizes a filename by replacing spaces with hyphens, removing special characters,
        and then replacing any double hyphens with single hyphens.

        Args:
            filename (str): The original filename.

        Returns:
            str: The sanitized filename.
        """
        # 1. Replace spaces with hyphens
        sanitized = filename.replace(" ", "-")

        # 2. Remove special characters (keep alphanumeric, hyphens, and dots)
        # This also ensures characters like '!' or '#' adjacent to spaces don't create extra hyphens initially
        sanitized = re.sub(r"[^\w\-\.]", "", sanitized)

        # 3. Replace any sequence of multiple hyphens with a single hyphen
        sanitized = re.sub(r"-+", "-", sanitized)

        # Optional: Remove leading/trailing hyphens if any
        sanitized = sanitized.strip("-")

        return sanitized

    def _get_image():
        if upload_pic.value:
            # mo.image(src=upload_pic.contents())
            up_img = Image.open(io.BytesIO(upload_pic.contents()))
            # path_to_image = FACES_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_upload.png"
            mo.stop(
                predicate=not image_name.value,
                output=mo.center(
                    mo.md("### Provide a name for the uploaded image above!")
                ),
            )

            path_to_image = (
                FACES_DIR / sanitize_filename(image_name.value)
            ).with_suffix(".png")
            up_img.save(fp=path_to_image)
            img = load_image_for_model(
                image_path=path_to_image, dtype=torch.double
            )  # reload and prep
            print("Image shape after prep for model:", img.shape)
            # up_img = up_img.convert("RGB")  # not necessary
            # return mo.image(up_img, height=IMG_SIZE[0])
            return img, path_to_image

        elif file_browser.value:
            path_to_image = file_browser.path(index=0)
            img = load_image_for_model(image_path=path_to_image, dtype=torch.double)
            print("Image shape after prep for model:", img.shape)

            return img, path_to_image
        else:
            return mo.stop(
                predicate=not (file_browser.value or upload_pic.value),
                output=mo.center(
                    mo.md("""### Please load or choose a face image to continue ...""")
                ),
            )

    img, path_to_image = _get_image()
    mo.center(mo.image(path_to_image, height=IMG_SIZE[0]))
    return img, path_to_image


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Get image labels of the original VGG-Face dataset""")
    return


@app.cell(hide_code=True)
def _(MODEL_DIR):
    # Read class labels
    path_to_labels = MODEL_DIR / "names.txt"

    with open(path_to_labels, "r") as labels:
        list_of_labels = labels.read().splitlines()
    return (list_of_labels,)


@app.cell(hide_code=True)
def _(mo):
    show_labels = mo.ui.switch(label="Show labels (i.e., names of celebrities)")
    show_labels
    return (show_labels,)


@app.cell(hide_code=True)
def _(list_of_labels, mo, show_labels):
    # Show labels
    def _show_labels():
        if show_labels.value:
            return mo.as_html(list_of_labels)  # e.g., "Abigail_Breslin"
        else:
            return mo.md("")

    _show_labels()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Predict the face identity in an image

    Use the selected image and predict the person in the image.

    Note that if you chose an image / identity, which is not part of the original data distribution,
    the network will make **a guess that should rely on similarities to data from the training distribution**. 
    However, **unexpected predictions / classifications can occur**, too!
    That is, the person in the provided image has little or no resemblance to the predicted celebrity (i.e., class).
    """
    )
    return


@app.cell(hide_code=True)
def _():
    def display_name(label: str) -> str:
        """Display a label as a name."""
        return label.replace("_", " ")

    def get_search_url(name: str) -> str:
        """Get a Google image search URL for the given name."""
        if "_" in name:
            # The label wasn't transformed to a name
            name = display_name(label=name)
        return f"https://www.google.com/search?q={name.replace(' ', '+')}&tbm=isch"

    return display_name, get_search_url


@app.cell(hide_code=True)
def _(display_name, get_search_url, img, list_of_labels, torch, vgg_face):
    # Push image through the model
    out = vgg_face(img)

    # Check the output vector
    n_labels = len(list_of_labels)
    assert len(out[0]) == n_labels  # 2622

    # Find index with the highest value in output vector, that is, the prediction
    idx_pred = torch.argmax(out)

    # Extract name of person
    pred_name = display_name(label=list_of_labels[idx_pred])
    img_search_url = get_search_url(name=pred_name)
    return idx_pred, img_search_url, n_labels, pred_name


@app.cell(hide_code=True)
def _(idx_pred, img_search_url, list_of_labels, mo, pred_name):
    neuron_stat = mo.stat(
        value=idx_pred.item(),
        label="predicted",
        caption="index of output neuron",
        bordered=True,
    )

    total_stat = mo.stat(
        value=len(list_of_labels),
        label="total",
        bordered=True,
        caption="number of faces",
    )

    mo.md(
        f"""
    ### The predicted person is: [**{pred_name}**]({img_search_url})
    ---

    {mo.hstack([neuron_stat, total_stat])}


    *Note: The total number of face identities is equal to the number of output neurons (classes)*
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Analyse the model decision""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Explain the prediction using explainable artificial intelligence (XAI)

    Use the post-hoc XAI-method [*Layer-wise relevance propagation* (LRP)](https://doi.org/10.1038/s41467-019-08987-4)
    to analyze the model decision.
    """
    )
    return


@app.cell(hide_code=True)
def _(VGGCanonizer):
    COMPOSITE_NAMES = [
        "epsilon_gamma_box",  # 0
        "epsilon_plus",  # 1
        "epsilon_alpha2_beta1",  # 2
        "epsilon_plus_flat",  # 3
        "epsilon_alpha2_beta1_flat",  # 4
        "excitation_backprop",  # 5
    ]

    canonizers = [VGGCanonizer()]  # == [SequentialMergeBatchNorm()]

    COMPOSITE_KWARGS = {
        "epsilon_gamma_box": dict(low=-3.0, high=3.0, canonizers=canonizers),
        "epsilon_plus": dict(canonizers=canonizers),
        "epsilon_alpha2_beta1": dict(canonizers=canonizers),
        "epsilon_plus_flat": dict(canonizers=canonizers),
        "epsilon_alpha2_beta1_flat": dict(canonizers=canonizers),
        "excitation_backprop": dict(canonizers=canonizers),
    }

    return COMPOSITE_KWARGS, COMPOSITE_NAMES


@app.cell(hide_code=True)
def _(display_name, idx_pred, list_of_labels, mo, n_labels):
    select_neuron = mo.ui.slider(
        start=0,
        stop=n_labels,
        step=1,
        value=int(idx_pred),
        label=f"Select the output neuron (`default`: **{int(idx_pred):,}** [{display_name(list_of_labels[idx_pred])}], "
        f"i.e., the neuron with the highest activation) to compute its relevance map:",
        show_value=True,
        full_width=True,
        include_input=True,
        debounce=False,
    )

    mo.vstack(
        [
            mo.md(
                f"""The network has {n_labels} output neurons,
                representing the face identities in the original dataset."""
            ),
            select_neuron,
        ]
    )

    return (select_neuron,)


@app.cell(hide_code=True)
def _(COMPOSITE_NAMES, mo):
    comp_name = mo.ui.dropdown(
        COMPOSITE_NAMES,
        allow_select_none=False,
        label="Choose an XAI-algorithm"
        "[`zennit` [composite name](https://zennit.readthedocs.io/en/latest/getting-started.html#composites)]",
        value="epsilon_alpha2_beta1",
    )

    mo.hstack(
        [
            mo.md(
                """
                There are many post-hoc explainable AI (XAI) methods.

                Here, we choose a small subset provided by the [`zennit`](https://zennit.readthedocs.io/en/latest/)
                toolbox.
                """
            ),
            comp_name,
        ],
        gap=3,
        widths=[1.5, 2],
    )

    return (comp_name,)


@app.cell(hide_code=True)
def _(
    COMPOSITES,
    COMPOSITE_KWARGS,
    Gradient,
    comp_name,
    img,
    mo,
    n_labels,
    select_neuron,
    torch,
    vgg_face,
):
    # Load zennit tools
    # composite = EpsilonGammaBox(low=-3.0, high=3.0, canonizers=canonizers)
    composite = COMPOSITES[comp_name.value](**COMPOSITE_KWARGS[comp_name.value])

    with Gradient(model=vgg_face, composite=composite) as attributor:
        outp, relevance = attributor(
            img, torch.eye(n_labels)[[select_neuron.value]]
        )  # outp == idx_pred

    mo.md(
        f"""
        The computed **relevance map has the same shape as the model input** (i.e., the face image): {relevance.shape}
        """
    )
    return (relevance,)


@app.cell(hide_code=True)
def _(relevance):
    # Compute the heatmap from the relevance map; That is, sum over color channels
    heatmap = relevance.sum(1)
    amax = heatmap.abs().numpy().max((1, 2))  # not necessary here
    return amax, heatmap


@app.cell(hide_code=True)
def _(amax, mo):
    # UI for displaying heatmaps
    CMAP = {
        "bwr": "bwr",
        "coldnhot": "coldnhot",
    }

    # CMAP.update(CMAPS.__dict__["_sources"].items())

    xai_cmap = mo.ui.dropdown(
        CMAP.keys(),
        value="coldnhot",  # "coldnhot",
        label="Choose a colormap",
        allow_select_none=False,
    )

    select_symmetric = mo.ui.switch(value=True, label="symmetric")

    slider_level = mo.ui.slider(
        start=0,
        stop=5,
        step=0.1,
        value=1,
        label="level",
        show_value=True,
        orientation="vertical",
    )

    rslider_vminmax = mo.ui.range_slider(
        show_value=True,
        start=-amax.item(),
        stop=amax.item(),
        step=0.00005,
        orientation="vertical",
        label="vmin,vmax",
        full_width=False,
        disabled=False,  # maybe disable
    )

    mo.vstack(
        [
            mo.md("### Visualize the relevance map"),
            mo.hstack(
                [xai_cmap, select_symmetric, slider_level, rslider_vminmax],
                # mo.vstack([xai_cmap, select_symmetric]),
                justify="start",
                align="start",
                gap=2,
            ),
            mo.md("To manipulate `vmin,vmax`, switch off `symmetric`"),
        ],
        gap=3,
    )
    return CMAP, rslider_vminmax, select_symmetric, slider_level, xai_cmap


@app.cell(hide_code=True)
def _(
    CMAP,
    IMG_SIZE,
    display_name,
    get_search_url,
    heatmap,
    imgify,
    list_of_labels,
    mo,
    path_to_image,
    rslider_vminmax,
    select_neuron,
    select_symmetric,
    slider_level,
    xai_cmap,
):
    heatmap_image = imgify(
        obj=heatmap[0].detach(),
        level=slider_level.value,  # 1.0,
        grid=False,  # only one image in "batch"
        symmetric=select_symmetric.value,
        vmin=None
        if select_symmetric.value
        else rslider_vminmax.value[0],  # -amax, # not necessary here, if symmetric=True
        vmax=None if select_symmetric.value else rslider_vminmax.value[1],  # amax,
        cmap=CMAP[xai_cmap.value],  # cmap="bwr"
    )

    selected_name = display_name(list_of_labels[select_neuron.value])

    mo.vstack(
        [
            mo.md(f"""
            ### Relevance map for output neuron: **{select_neuron.value:,}**

            This neuron represents: [{selected_name}]({get_search_url(selected_name)})
            """),
            mo.hstack(
                [
                    mo.image(path_to_image, width=IMG_SIZE[0]),
                    mo.image_compare(
                        before_image=path_to_image, after_image=heatmap_image
                    ),
                    mo.image(heatmap_image),
                ],
                widths=(IMG_SIZE[0],) * 3,
            ),
            mo.md(
                "Check out the `zennit` visualization "
                "[docs](https://zennit.readthedocs.io/en/latest/how-to/visualize-results.html) for more information"
            ),
        ],
        align="center",
        gap=3,
    )
    return


@app.cell(hide_code=True)
def _(comp_name, mo, path_to_image, select_neuron, slider_level, xai_cmap):
    path_to_heatmap = (
        mo.notebook_dir()
        / "results"
        / f"{path_to_image.stem}_{comp_name.value}-{select_neuron.value:04}_{xai_cmap.value}_{slider_level.value}.png"
    )

    save_button = mo.ui.button(
        label="Save the heatmap?",
        value=False,
        on_click=lambda value: not value,
        kind="success",
        full_width=False,
        # keyboard_shortcut="ctrl+s",
        tooltip=f"Save to './{path_to_heatmap.relative_to(mo.notebook_dir())}'",
    )
    mo.center(save_button)
    return path_to_heatmap, save_button


@app.cell(hide_code=True)
def _(
    CMAP,
    IMG_SIZE,
    heatmap,
    imsave,
    mo,
    path_to_heatmap,
    rslider_vminmax,
    save_button,
    select_symmetric,
    slider_level,
    xai_cmap,
):
    def _save_heatmap():
        if path_to_heatmap.exists():
            return mo.hstack(
                [
                    mo.md(
                        f"""#### **Saved heatmap**

                        Saved in: {path_to_heatmap.relative_to(mo.notebook_dir())}
                        """
                    ),
                    mo.image(path_to_heatmap, height=IMG_SIZE[0], width=IMG_SIZE[1]),
                ],
                widths=[2, 1],
            )

        if save_button.value:
            # sum over the color channels
            # get the absolute maximum, to center the heat map around 0
            # amax = heatmap.abs().numpy().max((1, 2))  # we do this now via a range-slider

            # save the heatmap as a color map
            imsave(
                path_to_heatmap,
                heatmap[0],  # or relevance[0].permute(1,2,0)
                symmetric=select_symmetric.value,
                vmin=None if select_symmetric.value else rslider_vminmax.value[0],
                vmax=None if select_symmetric.value else rslider_vminmax.value[1],
                cmap=CMAP[xai_cmap.value],  # "coldnhot"
                level=slider_level.value,
                grid=False,
            )

            return mo.hstack(
                [
                    mo.md(
                        f"""#### **Saved heatmap**

                        Saved in: {path_to_heatmap.relative_to(mo.notebook_dir())}
                        """
                    ),
                    mo.image(path_to_heatmap, height=IMG_SIZE[0], width=IMG_SIZE[1]),
                ],
                widths=[2, 1],
            )

        else:
            return mo.center(mo.md("""#### **Heatmap not saved**"""))

    _save_heatmap()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Extract latent activation maps

    We feed our selected image to the network and extract the neural activation in one selected layer,
    aka the activation maps.
    """
    )
    return


@app.cell(hide_code=True)
def _(Path, RESULT_DIR, ROOT_DIR, VGGFace, load_image_for_model, np, torch):
    path_to_activation_map_dir = RESULT_DIR / "activation_maps"

    def create_name_of_activation_map(layer_name: str, image_path: str | Path) -> str:
        """Create a name for the activation map based on the layer name and image path."""
        return f"{layer_name}_{Path(image_path).stem}.npy"

    def extract_activation_maps(
        model: VGGFace,
        image_path: str | Path,
        layer_name: str | None = None,
        cache: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> np.ndarray | list[np.ndarray]:
        """Extract activation map(s) from model layer(s)."""

        name_of_activation_map = create_name_of_activation_map(
            layer_name=layer_name, image_path=image_path
        )
        path_to_activation_map = path_to_activation_map_dir / name_of_activation_map
        if path_to_activation_map.exists():
            # Load the cached activation map
            if verbose:
                print(
                    f"Load activation map from: '{path_to_activation_map.relative_to(ROOT_DIR)}'"
                )
            return np.load(path_to_activation_map)

        model.eval()
        image = load_image_for_model(
            image_path=image_path, dtype=torch.double, **kwargs
        )

        # Forward image through VGGFace
        with torch.no_grad():
            _ = model(image)

        if layer_name is None:
            return [l_out.data.cpu().numpy() for l_out in model.layer_output]

        layer_name = layer_name.lower()
        if layer_name not in model.layer_names:
            msg = f"Layer name '{layer_name}' not in model.layer_names !"
            raise ValueError(msg)

        if cache:
            activation_map = (
                model.layer_output[model.layer_names.index(layer_name)]
                .data.cpu()
                .numpy()
            )
            path_to_activation_map_dir.mkdir(parents=True, exist_ok=True)
            np.save(arr=activation_map, file=path_to_activation_map)
            if verbose:
                print(
                    f"Activation map saved in: '{path_to_activation_map.relative_to(ROOT_DIR)}'"
                )

        return (
            model.layer_output[model.layer_names.index(layer_name)].data.cpu().numpy()
        )

    return (
        create_name_of_activation_map,
        extract_activation_maps,
        path_to_activation_map_dir,
    )


@app.cell(hide_code=True)
def _(ROOT_DIR, mo, path_to_activation_map_dir, vgg_face):
    select_layer = mo.ui.dropdown(
        options=[
            lay
            for lay in vgg_face.layer_names
            if not ("dropout" in lay or "relu" in lay)
        ],  # ignore relu & dropout layers
        value="fc7",
        label="Select a layer",
    )

    mo.hstack(
        [
            select_layer,
            mo.image(
                "https://miro.medium.com/v2/resize:fit:720/format:webp/0*QdiRfeMqo5D29riX.png",
                alt="Original VGG architecture",
                rounded=True,
                caption="Original VGG architecture – mainly the number of output differs to the VGG-Face",
                # width=400,
            ),
            mo.md(
                f"""
                Note, activation maps are cached in: *'./{path_to_activation_map_dir.relative_to(ROOT_DIR)}'*

                Also, especially early layers in the network take much more disk space.

                For the analysis here, later layers are more relevant: **'`conv5_3`' <= layer**
                """
            ),
        ],
        widths=[0.6, 1.5, 0.6],
        gap=2,
        align="center",
        # justify="start",
    )
    return (select_layer,)


@app.cell(hide_code=True)
def _(extract_activation_maps, mo, path_to_image, select_layer, vgg_face):
    # with mo.persistent_cache(name="activation_maps", save_path=RESULT_DIR / "cache"):
    amap = extract_activation_maps(
        model=vgg_face,
        image_path=path_to_image,
        layer_name=select_layer.value,
        verbose=False,
    )
    mo.md(
        f"""The activation maps of the `VGG-Face` layer '`{select_layer.value}`' are of shape: **{amap.shape}**"""
    )
    return


@app.cell(hide_code=True)
def _(FACES_DIR, ROOT_DIR, mo):
    all_amap_button = mo.ui.run_button(
        kind="success",
        tooltip=f"Compute / load activation maps for all face images in './{FACES_DIR.relative_to(ROOT_DIR)}/'",
        label="Compute / load all activation maps",
        full_width=True,
    )
    all_amap_button
    return (all_amap_button,)


@app.cell(hide_code=True)
def _(
    FACES_DIR,
    all_amap_button,
    create_name_of_activation_map,
    extract_activation_maps,
    get_df_vgg_activation_maps,
    mo,
    path_to_activation_map_dir,
    select_layer,
    vgg_face,
):
    # Compute activation maps of a layer for all faces
    def get_list_of_images():
        """Get a list of all images in the FACES_DIR."""
        list_of_images = [
            f_img
            for f_img in FACES_DIR.glob("*")
            if (
                ("_rect" not in f_img.name)
                and (f_img.suffix in [".jpg", ".png", ".jpeg"])
            )
        ]
        return list_of_images

    def check_amaps() -> bool:
        """Check if all activation maps are there."""
        list_of_images = get_list_of_images()
        list_of_amaps = []
        for img_path in list_of_images:
            list_of_amaps.append(
                create_name_of_activation_map(
                    layer_name=select_layer.value, image_path=img_path
                )
            )
        return all(
            [(path_to_activation_map_dir / amap).exists() for amap in list_of_amaps]
        )

    def _comput_load_amaps():
        if all_amap_button.value:
            list_of_images = get_list_of_images()
            for face_img_path in mo.status.progress_bar(
                list_of_images,
                title=f"Compute activations maps of layer '{select_layer.value}' for all faces",
                subtitle="Please wait",
                show_eta=True,
                show_rate=True,
                completion_subtitle=f"All activations maps of layer '{select_layer.value}' were computed",
                completion_title="Done",
                remove_on_exit=False,
                total=None,
            ):
                if face_img_path.suffix not in [".jpg", ".png", ".jpeg"]:
                    continue
                # list_of_images.append(face_img_path.name)
                extract_activation_maps(
                    model=vgg_face,
                    image_path=face_img_path,
                    layer_name=select_layer.value,
                    cache=True,
                    verbose=False,
                )
        else:
            return mo.stop(
                predicate=not all_amap_button.value,
                output=mo.center(mo.md("###Please compute / load all activation maps")),
            )

    _comput_load_amaps()
    feat_tab = get_df_vgg_activation_maps(layer_name=select_layer.value)
    return check_amaps, feat_tab


@app.cell(hide_code=True)
def _(ROOT_DIR, all_amap_button, mo, path_to_activation_map_dir):
    all_amap_button  # so the cell will update if the button is pressed above
    mo.md(
        f"""
    **These activation maps were found**:
    {mo.as_html([p.relative_to(ROOT_DIR) for p in path_to_activation_map_dir.glob("*.npy")])}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Analysis of feature / activation maps

    *using `PCA` and `UMAP`*

    Each face is represented as an activation map (or feature map) of the selected layer.
    These maps are of high dimensionality.
    Here, we explore the space these activation maps span,
    and ideally learn something about how the model *perceives* the different faces.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, np):
    select_pca = mo.ui.slider(
        steps=np.round(np.append([0], np.arange(0.80, 1.0, 0.01)), 2),
        value=None,
        label="""Apply `PCA`""",
        show_value=True,
        full_width=False,
    )
    mo.vstack(
        [
            mo.md(
                "We can apply `PCA` on the set of activation maps for dimensionality reduction"
            ),
            select_pca,
            mo.md("( 0: *No `PCA`* | 0.80 – 0.99: *% variance explained* )"),
        ],
        gap=2,
        align="center",
        justify="end",
    )
    return (select_pca,)


@app.cell(hide_code=True)
def _(PCA, feat_tab, mo, pd, select_pca):
    def _apply_pca(feature_table: pd.DataFrame | None):
        if feature_table is None:
            mo.stop(
                predicate=feature_table is None,  # == not all_amap_button.value,
                output=mo.center(
                    mo.md("#### Please compute / load all activation maps")
                ),
            )

        if select_pca.value:
            pca_model = PCA(
                n_components=select_pca.value,
                svd_solver="full",  # must be 0 < pca < 1 for svd_solver="full" (see docs)
            )  # .set_output(transform="pandas")
            # this finds n components such that pca*100 % of variance is explained
            feature_table = pca_model.fit_transform(
                feature_table.to_numpy()
            )  # (n_faces, n_features)
            # pca_model.components_
            explained_variance = pca_model.explained_variance_ratio_.sum()
            return (
                mo.center(
                    mo.md(
                        f"First {pca_model.n_components_} `PCA` components explain {explained_variance * 100:.1f}%"
                        f"of variance."
                    )
                ),
                feature_table,
            )
        else:
            return (mo.center(mo.md("*No `PCA` applied*")), feature_table)

    pca_md, feat_tab_t = _apply_pca(feat_tab)
    pca_md
    return (feat_tab_t,)


@app.cell(hide_code=True)
def _(feat_tab_t, umap):
    reducer = umap.UMAP(
        n_neighbors=5,
        min_dist=0.15,
        n_components=3,
        # random_state=42,
    )
    embedding = reducer.fit_transform(feat_tab_t)

    return (embedding,)


@app.cell(hide_code=True)
def _(alt, embedding, get_list_of_face_names, mo, pd, select_pca):
    df = pd.DataFrame(
        {
            "Dim-1": embedding[:, 0],
            "Dim-2": embedding[:, 1],
            "face": get_list_of_face_names(),
        }
    )

    title_txt = "UMAP projection of the activation maps"
    if select_pca.value:
        title_txt += f" (PCA-{select_pca.value:.0%})"

    points = (
        alt.Chart(df)
        .mark_circle(
            size=100,
            color="#FDA881",
        )
        .encode(
            x=alt.X("Dim-1", scale=alt.Scale(zero=False)),
            y=alt.Y("Dim-2", scale=alt.Scale(zero=False)),
            tooltip=["face"],  # Add more fields for richer tooltips
        )
    )

    # Overlay text next to each point
    alt_labels = (
        alt.Chart(df)
        .mark_text(
            align="left",
            baseline="middle",
            dx=7,  # Horizontal offset from the point
            color="lightgray",
            fontSize=14,
        )
        .encode(x="Dim-1", y="Dim-2", text="face")
    )

    chart = (
        (points + alt_labels)
        .properties(
            width=600,
            height=400,
            title=title_txt,
        )
        .configure_view(
            strokeWidth=0,
            fill="#222222",  # dark gray background
        )
        .configure(
            background="#222222",  # overall chart background
        )
        .configure_axis(
            grid=False,
        )
    )  # .interactive()

    # mo.center(chart)
    mo_chart = mo.ui.altair_chart(chart)

    _umap_text = (
        "Here, we apply [`UMAP`](https://umap-learn.readthedocs.io/en/latest/) on the set of activation maps "
        "across faces"
    )
    _umap_text += " after applying `PCA`." if select_pca.value else "."
    _umap_text += """

    Use your mouse to select an area of points – representing the faces – and explore them below.

    ---
    **Question**: Does proximity in the map represent the similarities between faces? Why (not)?

    ___
    """

    mo.hstack(
        [mo.md(_umap_text), mo.center(mo_chart)],
        align="center",
    )
    return (mo_chart,)


@app.cell(hide_code=True)
def _(mo, mo_chart):
    selection_table = mo.ui.table(mo_chart.value)
    return (selection_table,)


@app.cell(hide_code=True)
def _(Image, find_image_path, mo, mo_chart, selection_table):
    # show images: either the first 10 from the selection or the first ten
    # selected in the table
    mo.stop(not len(mo_chart.value))

    def show_images(faces: list[str], max_images: int = 10):
        import matplotlib.pyplot as plt

        faces = faces[:max_images]

        images = [Image.open(find_image_path(f)) for f in faces]

        fig, axes = plt.subplots(1, len(faces))
        fig.set_size_inches(12.5, 1.5)
        if len(faces) > 1:
            for im, ax in zip(images, axes.flat):
                ax.imshow(im, cmap="gray")
                ax.set_yticks([])
                ax.set_xticks([])
        else:
            axes.imshow(images[0], cmap="gray")
            axes.set_yticks([])
            axes.set_xticks([])
        plt.tight_layout()
        return fig

    selected_images = (
        mo.center(show_images(list(mo_chart.value.face)))
        if not len(selection_table.value)
        else mo.center(show_images(list(selection_table.value.face)))
    )

    mo.md(
        f"""
        **Images of the selected points (faces) in the UMAP**:

        {mo.as_html(selected_images)}

        Here's all the data you've selected.

        {selection_table}
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Compute similarities between faces

    We now use the feature maps – representing the faces – and compute their pair-wise similarities.
    """
    )
    return


@app.cell(hide_code=True)
def _(math, np, path_to_activation_map_dir, pd, select_layer):
    def chop_layer_name(face_name: str, layer_name: str) -> str:
        return face_name.split(f"{layer_name}_")[-1]

    def get_list_of_face_names():
        list_of_face_names = [
            chop_layer_name(fn.stem, select_layer.value)
            for fn in path_to_activation_map_dir.glob(f"{select_layer.value}*.npy")
        ]
        return sorted(list_of_face_names)

    def get_df_vgg_activation_maps(
        layer_name: str = select_layer.value,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Get a dataframe containing the available VGG-Face activation maps."""
        df_activation_maps = None  # init
        list_of_face_names = get_list_of_face_names()
        if verbose:
            print("Extract feature maps for:", list_of_face_names)
        for amap_path in path_to_activation_map_dir.glob(f"{select_layer.value}*.npy"):
            amap = np.load(amap_path)
            amap = amap.flatten()
            if df_activation_maps is None:
                m_len = len(amap)
                df_activation_maps = pd.DataFrame(
                    index=list_of_face_names,
                    columns=[
                        f"{layer_name.upper()}-{i: 0{math.floor(math.log(m_len, 10)) + 1}d}"
                        for i in range(m_len)
                    ],
                )
            df_activation_maps.loc[
                chop_layer_name(amap_path.stem, select_layer.value), :
            ] = amap.astype("float32")

        return df_activation_maps

    return get_df_vgg_activation_maps, get_list_of_face_names


@app.cell(hide_code=True)
def _(PCA, StandardScaler, get_df_vgg_activation_maps, np, npt, pd):
    def normalize(
        array: npt.ArrayLike,
        lower_bound: float,
        upper_bound: float,
        global_min: float | None = None,
        global_max: float | None = None,
    ) -> npt.ArrayLike:
        """
        Min-Max-Scaling: Normalizes the input-array to lower and upper bound.

        :param array: To be transformed array
        :param lower_bound: lower-bound a
        :param upper_bound: upper-bound b
        :param global_min: if the array is part of a larger tensor, normalize w.r.t. global min and ...
        :param global_max: ... global max (i.e., tensor min/max)
        :return: normalized array
        """
        if not lower_bound < upper_bound:
            msg = "lower_bound must be < upper_bound"
            raise AssertionError(msg)

        array = np.array(array)
        a, b = lower_bound, upper_bound

        if global_min is not None:
            if global_min > np.nanmin(array):
                msg = "global_min must be <= np.nanmin(array)"
                raise AssertionError(msg)
            mini = global_min
        else:
            mini = np.nanmin(array)

        if global_max is not None:
            if global_max < np.nanmax(array):
                msg = "global_max must be >= np.nanmax(array)"
                raise AssertionError(msg)
            maxi = global_max
        else:
            maxi = np.nanmax(array)

        return (b - a) * ((array - mini) / (maxi - mini)) + a

    def compute_feature_similarity_matrix(
        feature_table: pd.DataFrame,
        pca: bool | float = False,
        z_score: bool = True,
    ) -> npt.NDArray[np.float64]:
        """
        Compute the similarity matrix of a given feature table.

        :param feature_table: table with features of heads
        :param pca: False OR provide (0.< pca < 1.) if PCA should be run on feature table with n components such that
                    pca [float] *100 % of variance is explained
        :param z_score: True: z-score features before computing the similarity matrix
        :return: similarity matrix
        """
        if pca < 0.0 or pca >= 1.0:
            msg = "pca must be between 0 and 1!"
            raise ValueError(msg)

        # Scale features (i.e., z-transform per dimension / column) before computing similarity matrix
        if z_score:
            scaler = StandardScaler().set_output(transform="pandas")
            feature_table = scaler.fit_transform(X=feature_table)
            # this is the same as: scipy.stats.zscore(feature_table.to_numpy(), axis=0)

        # Run PCA (if requested)
        pca_feat_tab = None  # init
        if pca:
            pca_model = PCA(
                n_components=pca,
                svd_solver="full",  # must be 0 < pca < 1 for svd_solver="full" (see docs)
            )  # .set_output(transform="pandas")
            # this finds n components such that pca*100 % of variance is explained
            pca_feat_tab = pca_model.fit_transform(
                feature_table.to_numpy()
            )  # (n_faces, n_features)
            # pca_model.components_
            explained_variance = pca_model.explained_variance_ratio_.sum()
            print(
                f"First {pca_model.n_components_} PCA components explain {explained_variance * 100:.1f}% of variance."
            )

        feat_sim_mat = compute_cosine_similarity_matrix_from_features(
            features=feature_table.to_numpy().astype(np.float64)
            if not pca
            else pca_feat_tab.astype(np.float64)
        )
        # Take care with interpretation of cosine sim, since 1 == identical, -1 == opposite, 0 == orthogonal.
        # After normalization, this is not the case anymore.
        return normalize(feat_sim_mat, lower_bound=0.0, upper_bound=1.0)

    def compute_cosine_similarity_matrix_from_features(
        features: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the cosine similarity matrix according to a given feature matrix.

        :param features: (n_items, m_features)
        :return: cosine similarity matrix (n_items, n_items)
        """
        # Take the magnitude of over model dimensions per face
        magnitude_per_vice_dim = np.linalg.norm(features, axis=1)  # (n_faces, )

        # Outer product of the magnitude
        magnitude_per_cell = np.outer(
            magnitude_per_vice_dim, magnitude_per_vice_dim
        )  # (n_faces, n_faces)

        # Dot product of the weights (compare weights/dimensions of each face pair)
        similarity_matrix = features @ features.T  # (n_faces, n_faces)

        # Normalize to get the cosine similarity for each face pair
        return similarity_matrix / magnitude_per_cell  # (n_faces, n_faces)

    def compute_vgg_feature_map_similarity_matrix(
        layer_name: str,
        pca: bool | float = False,
    ) -> npt.NDArray[np.float64]:
        """
        Compute a similarity matrix from `VGGFace` feature maps.

        !!! note "To extend computational efficiency"
            Intermediate results are saved to disk, such that they do not have to be recomputed each time
            (time-consuming).

        :param layer_name: name of the VGG layer to use
        :param pca: False OR provide (0.< pca < 1.) if PCA should be run on feature table with n components such that
                    pca [float] *100 % of variance is explained
        :return: similarity matrix based on VGGFace feature maps
        """
        feat_tab = get_df_vgg_activation_maps(layer_name=layer_name)
        # Compute similarity matrix
        return compute_feature_similarity_matrix(
            feature_table=feat_tab, pca=pca, z_score=False
        )

    return (compute_vgg_feature_map_similarity_matrix,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Compute the similarity matrix""")
    return


@app.cell
def _(
    all_amap_button,
    check_amaps,
    compute_vgg_feature_map_similarity_matrix,
    select_layer,
):
    sim_mat = None  # init
    if all_amap_button.value or check_amaps():
        sim_mat = compute_vgg_feature_map_similarity_matrix(
            layer_name=select_layer.value,
        )
    return (sim_mat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##### Visualize the similarity matrix""")
    return


@app.cell(hide_code=True)
def _(get_list_of_face_names, np, plt, rsatoolbox):
    def vectorize_similarity_matrix(face_sim_mat: np.ndarray) -> np.ndarray:
        """
        Take the upper triangle of a given similarity matrix and return it as vector.

        ??? note "Ways to vectorize a matrix"
            ```python
                a = [[1 2 3]
                     [4 5 6]
                     [7 8 9]]

                # This is how the diagonal can be excluded (as it is required here)
                print(a[np.triu_indices(n=3, k=1)])
                # > array([2, 3, 6])

                # In contrast, see how the diagonal can be included (as it is not done here):
                print(a[np.triu_indices(n=3, k=0)])
                # > array([1, 2, 3, 5, 6, 9])
            ```

        :param face_sim_mat: face similarity matrix
        :return: 1d vector of upper triangle
        """
        return face_sim_mat[np.triu_indices(n=face_sim_mat.shape[0], k=1)]

    def visualise_matrix(
        face_sim_mat: np.ndarray,
        **kwargs,
    ) -> str | plt.Figure:
        """
        Visualize face similarity judgments.

        :param face_sim_mat: matrix of face similarities
        :return: figure object
        """
        # Get the name of the figure
        fig_name = kwargs.pop("fig_name", "Similarity matrix")

        # Compute size of the figure
        figsize = kwargs.pop(
            "figsize",
            (
                round(
                    face_sim_mat.shape[1] / min(face_sim_mat.shape) * 10
                ),  # keep x-axis longer since we add colorbar
                round(face_sim_mat.shape[0] / min(face_sim_mat.shape) * 9),
            ),
        )
        # This is not ideal for our case, since it works with data with shape of (observations x channels).
        rdms = rsatoolbox.rdm.RDMs(
            dissimilarities=vectorize_similarity_matrix(face_sim_mat=face_sim_mat)
        )

        if "pattern_descriptor" in kwargs:
            rdms.pattern_descriptors.update({"labels": get_list_of_face_names()})
            rdms.pattern_descriptors.update({"index": range(face_sim_mat.shape[0])})

        fig, ax_array, _ = rsatoolbox.vis.show_rdm(
            rdms=rdms,
            show_colorbar="panel",
            vmin=kwargs.pop("vmin", 0.0),
            vmax=kwargs.pop("vmax", 1.0),
            figsize=figsize,
            rdm_descriptor=fig_name,
            num_pattern_groups=face_sim_mat.shape[0] / 2
            if face_sim_mat.shape[0] % 2 == 0
            else None,
            pattern_descriptor=kwargs.pop(
                "pattern_descriptor", None
            ),  # labels OR index
            **kwargs,
            # cmap="viridis",
        )

        # Set labels and title
        if "xlabel" in kwargs:
            ax_array[0][0].set_xlabel(kwargs.pop("xlabel"))
        if "ylabel" in kwargs:
            ax_array[0][0].set_ylabel(kwargs.pop("ylabel"))
        # plt.show(block=False)
        return fig

    return (visualise_matrix,)


@app.cell(hide_code=True)
def _(mo, sim_mat, visualise_matrix):
    fig_sim_mat = None  # init
    if sim_mat is not None:
        fig_sim_mat = visualise_matrix(
            face_sim_mat=sim_mat,
            pattern_descriptor="labels",  # "index" or "labels"
            figsize=(8, 7),
            # cmap="bwr"
        )
    mo.center(fig_sim_mat)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Examples of who looks dis-/similar""")
    return


@app.cell(hide_code=True)
def _(all_amap_button, get_list_of_face_names, mo, path_to_image):
    all_amap_button  # reload when button is pressed
    select_face = mo.ui.dropdown(
        get_list_of_face_names(),
        label="Explore similarities for the selected face image:",
        value=path_to_image if path_to_image is None else path_to_image.stem,
    )
    select_face
    return (select_face,)


@app.cell(hide_code=True)
def _(FACES_DIR, Path, get_list_of_face_names, mo, np, select_face, sim_mat):
    # for idx_face, face_name in enumerate(get_list_of_face_names()):

    def find_image_path(face_name: str) -> Path:
        return list(FACES_DIR.glob(f"{face_name}.*"))[0]  # there should be only one

    def _display_similarity():
        if select_face.value:
            idx_face = get_list_of_face_names().index(select_face.value)
            i_similarities = sim_mat[idx_face, :]
            i_similarities[idx_face] = -np.inf
            idx_most_similar = i_similarities.argmax()
            i_similarities[idx_face] = np.inf
            idx_most_dissimilar = i_similarities.argmin()
            name_most_similar = get_list_of_face_names()[idx_most_similar]
            name_most_dissimilar = get_list_of_face_names()[idx_most_dissimilar]

            max_height_width = 250
            return mo.hstack(
                [
                    mo.md(
                        f"""##### **Which face is dis-/similar to face '{select_face.value}'?**"""
                    ),
                    mo.hstack(
                        [
                            mo.vstack(
                                [
                                    mo.center(mo.md("Selected image")),
                                    mo.image(
                                        find_image_path(select_face.value),
                                        height=max_height_width,
                                        width=max_height_width,
                                    ),
                                    mo.center(
                                        mo.md(
                                            f"**{select_face.value}** | idx: {idx_face}"
                                        )
                                    ),
                                ]
                            ),
                            mo.vstack(
                                [
                                    mo.center(mo.md("**Most similar**")),
                                    mo.image(
                                        find_image_path(name_most_similar),
                                        height=max_height_width,
                                        width=max_height_width,
                                    ),
                                    mo.center(
                                        mo.md(
                                            f"**{name_most_similar}** | idx: {idx_most_similar}"
                                        )
                                    ),
                                ]
                            ),
                            mo.vstack(
                                [
                                    mo.center(mo.md("**Most dissimilar**")),
                                    mo.image(
                                        find_image_path(name_most_dissimilar),
                                        height=max_height_width,
                                        width=max_height_width,
                                    ),
                                    mo.center(
                                        mo.md(
                                            f"**{name_most_dissimilar}** | idx: {idx_most_dissimilar}"
                                        )
                                    ),
                                ]
                            ),
                        ]
                    ),
                ],
                gap=2,
                align="center",
            )

    _display_similarity()
    return (find_image_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""## Discussion – How to combine the previous analysis with cognitive & neuroscientific data"""
    )
    return


@app.cell(hide_code=True)
def _(RESULT_DIR, ROOT_DIR, mo):
    discussion_notes = mo.ui.text_area(
        placeholder="Take notes in markdown ...", rows=20
    )
    save_notes = mo.ui.run_button(
        # on_click=save_notes_to_file(),
        label="Save notes",
        kind="success",
        full_width=True,
        tooltip=f"Press to save notes in './{RESULT_DIR.relative_to(ROOT_DIR)}/'",
        keyboard_shortcut="Ctrl-s",
    )
    mo.vstack(
        [
            discussion_notes,
            save_notes,
        ]
    )
    return discussion_notes, save_notes


@app.cell(hide_code=True)
def _(RESULT_DIR, ROOT_DIR, datetime, discussion_notes, mo, save_notes):
    def _save_notes_to_file():
        if save_notes.value:
            path_to_notes = RESULT_DIR / "discussion_notes.md"
            path_to_notes.write_text(discussion_notes.value)
            return mo.md(
                f"""
                ### Saved notes 

                at *{datetime.now()}* | to './{path_to_notes.relative_to(ROOT_DIR)}'
                """
            )
        else:
            return mo.center(
                mo.md("### Press the 'Save notes' button to save changes!")
            )

    _save_notes_to_file()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Further material on XAI""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(
                """
                ### Advanced versions of `LRP`

                See the [**CBS CoCoNUT**](https://www.cbs.mpg.de/cbs-coconut/past-meetings) talk by
                Sebastian Lapuschkin.
                """
            ),
            mo.video(
                src="https://streaming-eu.mpg.de/de/institute/cbs/CoCoNUT/2024-11-27_Sebastian-Lapuschkin_CBS-CoCoNUT.mp4",
                autoplay=False,
                controls=True,
                loop=False,
                muted=False,
                rounded=True,
                height=None,
                width=None,
            ),
        ],
        gap=0,
    )

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Housekeeping

    Some recommendations after being done: 


    ### Clean up with `uv`

    The (great) Python package manager `uv` likes to cache packages to reuse them across projects.
    If you do not intend to use `uv` as Python-package manager,
    consider deleting the cache of `uv`. 

    Run the following in your terminal:

    ```
    uv cache clean
    ```

    ### Clean up remaining folders

    You only need the files `facexai.py`, `README.md`, and the `./data/` folder to reproduce the pipeline above.
    Consider deleting the `./results/` folder to save some disk space after running the script.

    In case the automatic data download has been interrupted, delete the folder `./download/` manually.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.hstack(
        [
            mo.md(
                r"""
                ## Contact

                * Simon M. Hofmann: simon.hofmann[ät]cbs.mpg.de
                * GitHub: [@SHEscher](https://github.com/SHEscher)
                * Bluesky: [@smnhfmnn.bsky.social](https://bsky.app/profile/smnhfmnn.bsky.social/)
                * Website: [@MPI CBS](https://www.cbs.mpg.de/employees/simon-m-hofmann)
                """
            ),
            mo.image(
                src="https://www.cbs.mpg.de/employee_images/94915-1681740655?t=eyJ3aWR0aCI6NDI2LCJoZWlnaHQiOjU0OCwiZml0IjoiY3JvcCIsImZpbGVfZXh0ZW5zaW9uIjoid2VicCJ9--27646ab4f30e7fedcf3f03ebd360565617825a1c",
                height=200,
                rounded=True,
                caption="@ MPI CBS",
            ),
        ]
    )

    return


if __name__ == "__main__":
    app.run()
