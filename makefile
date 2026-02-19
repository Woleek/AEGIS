run_experiments:
	uv run bash scripts/run_experiments.sh

render_avatar:
	uv run GaussianAvatars/local_viewer.py \
		--point_path ./datasets/NeRSembleReconst/avatars/074/point_cloud.ply

robust_test:
	uv run python scripts/robustness_test.py \
		--input-dir ./datasets/CombinedReconst/renders \
		--output-base ./datasets/CombinedReconst/robustness \
		--degradations all

eval_robust:
	@for deg in $$(ls ./datasets/CombinedReconst/robustness); do \
		echo "=== Evaluating $$deg ==="; \
		uv run python scripts/evaluate.py \
			--gallery-dataset lfw \
			--gallery-dataset CombinedGT \
			--verification-threshold-dataset lfw \
			--anonymized-dataset CombinedGT \
			--anonymized-path ./datasets/CombinedReconst/robustness/$$deg \
			--evaluation-method verification \
			--embedder adaface; \
	done
# $$(ls ./datasets/seed42/NeRSembleMasked_adaface_all/eps_0.200/robustness)

eval_rank_k:
	uv run python scripts/evaluate.py \
		--gallery-dataset CelebA \
		--celeba-test-set-only \
		--gallery-dataset CombinedGT \
		--anonymized-dataset CombinedGT \
		--anonymized-path "./datasets/CombinedReconst_lowkey" \
		--evaluation-method rank_k \
		--embedder adaface

eval_verification:
	uv run python scripts/evaluate.py \
		--gallery-dataset lfw \
		--gallery-dataset CombinedGT \
		--verification-threshold-dataset lfw \
		--anonymized-dataset CombinedGT \
		--anonymized-path "./datasets/CombinedReconst_lowkey" \
		--evaluation-method verification \
		--embedder adaface

eval_utility:
	uv run python scripts/evaluate.py \
		--anonymized-dataset CombinedReconst \
		--anonymized-path "./datasets/CombinedReconst_lowkey" \
		--evaluation-method utility

mask_avatar:
	uv run python scripts/mask_avatar.py \
		--avatar-dir ./datasets/FaceScapeReconst/avatars/527 \
		--target-image ./datasets/FaceScapeReconst/renders/527.png \
		--camera-boundary-angles -0.5 0.5 -0.5 0.5 0.0 0.0 \
		--angle-aggregation mean \
		--seed 42 \
		--epsilons 0.1 \
		--attack-steps 300 \
		--target-features DC \
		--adv-attack linfpgd \
		--embedder adaface \
		--output-name FaceScapeMasked

# ver-threshold: 0.1720 for AdaFace | 0.1840 for ArcFace | None
# select-regions: eyes, lips, nose, ears, forehead
# epsilons: 0.05, 0.1, 0.2, 0.3

# Adaptive Regional Epsilon Budgets
# Uses lower epsilon for skin regions (reduces artifacts), higher for identity-critical regions
mask_avatar_adaptive:
	uv run python scripts/mask_avatar.py \
		--avatar-dir ./datasets/NeRSembleReconst/avatars/306 \
		--target-image ./datasets/NeRSembleReconst/renders/306.png \
		--camera-boundary-angles -0.5 0.5 -0.5 0.5 0.0 0.0 \
		--angle-aggregation mean \
		--seed 42 \
		--epsilons 0.2 \
		--attack-steps 300 \
		--target-features DC \
		--adv-attack linfpgd \
		--embedder adaface \
		--adaptive-epsilon \
		--output-name NeRSembleMasked_adaptive