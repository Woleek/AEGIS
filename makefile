run_experiments:
	uv run bash scripts/run_experiments.sh

render_avatar:
	uv run GaussianAvatars/local_viewer.py \
		--point_path ./datasets/NeRSembleMasked_adaface_all/eps0.100/avatars/306/point_cloud.ply

eval_rank_k:
	uv run python scripts/evaluate.py \
		--gallery-dataset CelebA \
		--celeba-test-set-only \
		--gallery-dataset NeRSembleGT \
		--anonymized-dataset NeRSembleGT \
		--anonymized-path ./datasets/NeRSembleMasked_adaface_all/eps0.100/renders \
		--evaluation-method rank_k \
		--embedder adaface

eval_verification:
	uv run python scripts/evaluate.py \
		--gallery-dataset lfw \
		--gallery-dataset NeRSembleGT \
		--verification-threshold-dataset lfw \
		--anonymized-dataset NeRSembleGT \
		--anonymized-path ./datasets/NeRSembleMasked_adaface_all/eps0.100/renders \
		--evaluation-method verification \
		--embedder adaface

eval_utility:
	uv run python scripts/evaluate.py \
		--anonymized-dataset NeRSembleReconst \
		--anonymized-path ./datasets/NeRSembleMasked_adaface_all/eps0.100/renders \
		--evaluation-method utility

mask_avatar:
	uv run python scripts/mask_avatar.py \
		--avatar-dir ./datasets/NeRSembleReconst/avatars/306 \
		--target-image ./datasets/NeRSembleReconst/renders/306.png \
		--camera-boundary-angles -0.5 0.5 -0.5 0.5 0.0 0.0 \
		--angle-aggregation mean \
		--seed 42 \
		--epsilons 0.1 \
		--attack-steps 300 \
		--target-features DC \
		--adv-attack linfpgd \
		--embedder adaface
		
# ver-threshold: 0.1720 for AdaFace | 0.1840 for ArcFace | None
# select-regions: eyes, lips, nose, ears, forehead
# epsilons: 0.05, 0.1, 0.2, 0.3