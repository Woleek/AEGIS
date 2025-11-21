### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) installed

### Setup
1. Run `uv sync`
1. Open new terminal with `aegis` environment
1. Run `uv pip install -e .`
1. Download Gaussian Avatars repo using `git clone https://github.com/Woleek/GaussianAvatars.git --recursive`
1. Run `bash scripts/download_models.sh` and `bash scripts/download_datasets.sh`
1. Go to `./GaussianAvatars/chumpy` and run `uv pip install .`
1. Download Insightface repo using `git clone https://github.com/deepinsight/insightface.git`

<!-- Alternative to download GaussianAvatars from original repo -->
<!-- 1. Download `FLAME_masks.pkl`, `flame2023.pkl` following `https://github.com/ShenhanQian/GaussianAvatars/blob/main/doc/download.md` -->
<!-- 1. Inside `./GaussianAvatars/flame_model/flame.py` change paths of `FLAME_MESH_PATH`, `FLAME_LMK_PATH`, `FLAME_MODEL_PATH`, `FLAME_PARTS_PATH` to absolute paths in your system -->

<!-- ```
git clone --depth 1 https://github.com/mattloper/chumpy.git && \
    cd chumpy && \
    poetry init \
        --name "chumpy" \
        --description "chumpy (poetry build)" \
        --author "Matthew Loper" \
        --license "MIT" \
        --no-interaction && \
    poetry version $(python -c 'import runpy;print(runpy.run_path("./chumpy/version.py")["version"])') && \
    poetry add numpy@* scipy@* six@* && \
    poetry build
``` -->