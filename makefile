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

# Embedders: arcface adaface swinface transface facenet cosface ir152 irse50 mobileface
# Override the embedder on any eval/mask target, e.g.: make eval_rank_k EMBEDDER=ir152
eval_rank_k:
	uv run python scripts/evaluate.py \
		--gallery-dataset CelebA \
		--celeba-test-set-only \
		--gallery-dataset CombinedGT \
		--anonymized-dataset CombinedGT \
		--anonymized-path "./datasets/CombinedReconst_lowkey" \
		--evaluation-method rank_k \
		--embedder $(or $(EMBEDDER),adaface)

# Verification protocol defaults to TAR@FAR (target FAR 1e-3); EER still computed/reported.
# Override: make eval_verification PROTOCOL=eer  /  make eval_verification TARGET_FAR=1e-2
eval_verification:
	uv run python scripts/evaluate.py \
		--gallery-dataset lfw \
		--gallery-dataset CombinedGT \
		--verification-threshold-dataset lfw \
		--anonymized-dataset CombinedGT \
		--anonymized-path "./datasets/CombinedReconst_lowkey" \
		--evaluation-method verification \
		--verification-protocol $(or $(PROTOCOL),tar_at_far) \
		--target-far $(or $(TARGET_FAR),1e-3) \
		--embedder $(or $(EMBEDDER),adaface)

# Utility reports SSIM, PSNR and FID (set-level, unmasked vs masked).
eval_utility:
	uv run python scripts/evaluate.py \
		--anonymized-dataset CombinedReconst \
		--anonymized-path "./datasets/CombinedReconst_lowkey" \
		--evaluation-method utility

# Single-view 1:1 verification (renders a frontal view, compares to the first GT image
# at the model's calibrated threshold). Pass SUBJECT, EMBEDDER, RADIUS as needed.
#   make verify_simple SUBJECT=527 EMBEDDER=ir152 RADIUS=20
verify_simple:
	uv run python scripts/verify_avatar_simple.py \
		--gt-images datasets/CombinedGT/images/$(or $(SUBJECT),527) \
		--avatar-dir datasets/FaceScapeReconst/avatars/$(or $(SUBJECT),527) \
		--embedder $(or $(EMBEDDER),adaface) \
		--radius $(or $(RADIUS),20) \
		--device cuda

# Variance: pass CSV= and TYPE= on the command line, e.g.:
#   make variance CSV=output/evaluations/combined/aegis_arcface_all_eps0.2/rank_k.csv TYPE=rank_k
variance:
	uv run python scripts/compute_variance.py \
		--csv $(CSV) \
		--type $(TYPE)

# CMC: pass CURVES= (space-separated "Label:path" pairs) and OUT= on the command line, e.g.:
#   make cmc CURVES="Unmasked:output/.../rank_k.csv AEGIS:output/.../rank_k.csv" OUT=output/figures/cmc.pdf
cmc:
	uv run python scripts/plot_cmc.py \
		$(foreach c,$(CURVES),--csv "$(c)") \
		--k-max 200 \
		--output $(or $(OUT),output/figures/cmc.pdf)

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
		--embedder $(or $(EMBEDDER),adaface) \
		--output-name FaceScapeMasked

# Ensemble masking: optimize against multiple FR surrogates at once (opt-in multi-FR mode).
# Override the ensemble with SURROGATES="model:variant ...", e.g.:
#   make mask_avatar_ensemble SURROGATES="ir152:r152 irse50:ir_se50 facenet:vggface2"
mask_avatar_ensemble:
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
		--embedder $(or $(EMBEDDER),adaface) \
		--surrogate-keys $(or $(SURROGATES),arcface:r50 facenet:vggface2 swinface:swin_t) \
		--cross-model-aggregation mean \
		--radius 20 \
		--output-name FaceScapeMasked_ensemble

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

# Temporal consistency: SUBJECT=306 MASKED_DIR=... RADIUS=1
temporal_consistency:
	uv run python scripts/temporal_consistency.py \
		--unmasked-ply datasets/NeRSembleReconst/avatars/$(or $(SUBJECT),306)/point_cloud.ply \
		--masked-ply $(or $(MASKED_DIR),datasets/seed42/CombinedMasked_adaptive_adaface_all/eps_0.200)/avatars/$(or $(SUBJECT),306)/point_cloud.ply \
		--reference-image datasets/CombinedGT/images/$(or $(SUBJECT),306)/0_00001_08.png \
		--radius $(or $(RADIUS),1) \
		--camera-boundary-angles -0.5 0.5 -0.5 0.5 0.0 0.0 \
		--output-dir output/temporal_consistency/$(or $(SUBJECT),306)


# SH identity attack: tests whether unperturbed higher-order SH can reconstruct identity
sh_identity_attack:
	uv run python scripts/sh_identity_attack.py \
		--masked-dir $(or $(MASKED_DIR),datasets/seed42/CombinedMasked_adaptive_adaface_all/eps_0.200)

color_diff:
	uv run scripts/plot_color_diff.py \
		--original datasets/NeRSembleReconst/renders/306.png \
		--masked datasets/seed42/CombinedMasked_adaptive_adaface_all/eps_0.200/renders/306.png \
		--labels "AEGIS RA" \
		--amplify 10 \
		--output output/figures/color_diff_306.pdf

tradeoff:
	uv run scripts/plot_tradeoff.py \
		--eval-dir output/evaluations/seed42 --embedder adaface \
		--renders-dir datasets/NeRSembleReconst/renders \
		--masked-dirs \
			0.05:datasets/seed42/CombinedMasked_adaface_all/eps_0.050/renders \
			0.1:datasets/seed42/CombinedMasked_adaface_all/eps_0.100/renders \
			0.2:datasets/seed42/CombinedMasked_adaface_all/eps_0.200/renders \
			0.3:datasets/seed42/CombinedMasked_adaface_all/eps_0.300/renders \
			base:datasets/NeRSembleReconst/renders \
		--ra-masked-dirs \
			0.2:datasets/seed42/CombinedMasked_adaptive_adaface_all/eps_0.200/renders \
		--subject 306 --output-dir output/figures


diffprivacy:
	uv run python scripts/mask_2d_diffprivacy_pytorch.py \
      --input-dir ./datasets/NeRSembleReconst/renders \
      --output-dir ./datasets/NeRSembleReconst_diffprivacy \
      --embedder adaface

face_anon_simple:
	uv run python scripts/mask_2d_faceanonsimple.py \
      --input-dir ./datasets/NeRSembleReconst/renders \
      --output-dir ./datasets/NeRSembleReconst_faceanon

fawkes:
	uv run python scripts/mask_2d_fawkes_pytorch.py \
		--input-dir ./datasets/CombinedReconst/renders \
		--output-dir ./datasets/CombinedReconst_fawkes \
		--mode high \
		--embedder adaface

lowkey:
	uv run python scripts/mask_2d_lowkey_pytorch.py \
		--input-dir ./datasets/CombinedReconst/renders \
		--output-dir ./datasets/CombinedReconst_lowkey \
		--eps 0.05 \
		--embedder adaface

identitydp_eps100:
	uv run python scripts/mask_2d_identitydp.py \
		--input-dir ./datasets/CombinedReconst/renders \
		--output-dir ./datasets/CombinedReconst_identitydp_eps100 \
		--dp-epsilon 100

identitydp_eps1:
	uv run python scripts/mask_2d_identitydp.py \
		--input-dir ./datasets/CombinedReconst/renders \
		--output-dir ./datasets/CombinedReconst_identitydp_eps1 \
		--dp-epsilon 1

identitydp: identitydp_eps100 identitydp_eps1

pixeldp_eps20:
	uv run python scripts/mask_2d_pixeldp.py \
		--input-dir ./datasets/CombinedReconst/renders \
		--output-dir ./datasets/CombinedReconst_pixeldp_eps20 \
		--dp-epsilon 20

pixeldp_eps5:
	uv run python scripts/mask_2d_pixeldp.py \
		--input-dir ./datasets/CombinedReconst/renders \
		--output-dir ./datasets/CombinedReconst_pixeldp_eps5 \
		--dp-epsilon 5

pixeldp: pixeldp_eps20 pixeldp_eps5

metricsvd_eps20:
	uv run python scripts/mask_2d_metricsvd.py \
		--input-dir ./datasets/CombinedReconst/renders \
		--output-dir ./datasets/CombinedReconst_metricsvd_eps20 \
		--dp-epsilon 20

metricsvd_eps1:
	uv run python scripts/mask_2d_metricsvd.py \
		--input-dir ./datasets/CombinedReconst/renders \
		--output-dir ./datasets/CombinedReconst_metricsvd_eps1 \
		--dp-epsilon 1

metricsvd: metricsvd_eps20 metricsvd_eps1

# Face++ rank-k evaluation (black-box)
# Requires FACEPP_API_KEY and FACEPP_API_SECRET env vars
# Override QUERY_DIR, GALLERY_DIR, or OUTPUT on the command line, e.g.:
#   make facepp_rank_k QUERY_DIR=datasets/seed42/CombinedMasked_arcface_all/eps_0.200/renders
facepp_rank_k:
	uv run python scripts/facepp_rank_k.py \
		--query-dir $(or $(QUERY_DIR),datasets/seed42/CombinedMasked_adaptive_adaface_all/eps_0.200/renders) \
		--gallery-dir $(or $(GALLERY_DIR),datasets/CombinedGT/images) \
		--output $(or $(OUTPUT),output/evaluations/combined/blackbox/facepp_aegis_ra_adaface_eps0.2/rank_k.csv)