# Download CelebA
mkdir -p datasets/CelebA
uv run gdown https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM -O "datasets/CelebA/img_align_celeba.zip"
uv run gdown https://drive.google.com/uc?id=0B7EVK8r0v71pOXBhSUdJWU1MYUk -O "datasets/CelebA/README.md"
uv run gdown https://drive.google.com/uc?id=1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS -O "datasets/CelebA/identity_CelebA.txt"
unzip datasets/CelebA/img_align_celeba.zip -d datasets/CelebA/
rm datasets/CelebA/img_align_celeba.zip

# Download LFW
mkdir -p datasets/lfw
wget https://www.kaggle.com/api/v1/datasets/download/jessicali9530/lfw-dataset -O datasets/lfw/lfw-dataset.zip
unzip datasets/lfw/lfw-dataset.zip -d datasets/lfw/
rm datasets/lfw/lfw-dataset.zip