# !/bin/bash
embedder=adaface # original embedder used for masking (arcface or adaface)
select_regions=("") # regions to mask; empty string means all regions ("", "eyes", "nose", "lips", "forehead" or combinations)
ce=false # cross-evaluation flag (true or false)
seed=42 # random seed used for masking

# Trap SIGINT (Ctrl+C) to exit gracefully
trap 'echo "Interrupted by user"; exit 1' INT

for regions in "${select_regions[@]}"; do

    if [ "$ce" == "false" ]; then # only run masking if not cross-eval
        for avatar_dir in ./datasets/NeRSembleReconst/avatars/*; do
            if [ -d "$avatar_dir" ]; then
                avatar_id=$(basename "$avatar_dir")
                target_image="./datasets/NeRSembleReconst/renders/${avatar_id}.png"

                echo "Running mask_avatar.py for avatar: $avatar_id with regions: $regions"
                uv run scripts/mask_avatar.py \
                    --avatar-dir "$avatar_dir" \
                    --target-image "$target_image" \
                    $([ -n "$regions" ] && echo "--select-regions $regions") \
                    --epsilons 0.05 0.1 0.2 0.3 \
                    --attack-steps 300 \
                    --camera-boundary-angles -0.5 0.5 -0.5 0.5 0.0 0.0 \
		            --angle-aggregation mean \
                    --seed $seed \
                    --adv-attack linfpgd \
                    --embedder $embedder
            fi
        done
    fi

    masked_path="./datasets/seed${seed}/NeRSembleMasked_${embedder}_"
    if [ -n "$regions" ]; then
        # add regions separated by underscores, sorted by name
        regions_str=$(echo $regions | tr ' ' '\n' | sort | tr '\n' '_' | sed 's/_$//')
        masked_path="${masked_path}${regions_str}"
    else
        regions_str="all"
        masked_path="${masked_path}all"
    fi

    # Determine embedder for eval
    if [ "$ce" == "false" ]; then
        eval_embedder=$embedder
        label_prefix=""
    else
        label_prefix="ce_"
        if [ "$embedder" == "arcface" ]; then
            eval_embedder="adaface"
        else
            eval_embedder="arcface"
        fi
    fi

    for eps in 0.050 0.100 0.200 0.300; do
        temp_masked_path="${masked_path}/eps_${eps}/renders"
        echo "Running evaluate.py for path: $temp_masked_path with epsilon: $eps"

        # Rank K
        uv run python scripts/evaluate.py \
            --gallery-dataset CelebA \
            --celeba-test-set-only \
            --gallery-dataset NeRSembleGT \
            --anonymized-dataset NeRSembleGT \
            --anonymized-path "$temp_masked_path" \
            --evaluation-method rank_k \
            --embedder $eval_embedder \
            --label "seed${seed}/${label_prefix}${embedder}_${regions_str}_eps${eps}"

        # Verification
        uv run python scripts/evaluate.py \
            --gallery-dataset lfw \
            --gallery-dataset NeRSembleGT \
            --verification-threshold-dataset lfw \
            --anonymized-dataset NeRSembleGT \
            --anonymized-path "$temp_masked_path" \
            --evaluation-method verification \
            --embedder $eval_embedder \
            --label "seed${seed}/${label_prefix}${embedder}_${regions_str}_eps${eps}"

        if [ "$ce" == "false" ]; then # only run utility if not cross-eval
            # Utility
            uv run python scripts/evaluate.py \
                --anonymized-dataset NeRSembleReconst \
                --anonymized-path "$temp_masked_path" \
                --evaluation-method utility \
                --label "seed${seed}/${embedder}_${regions_str}_eps${eps}"
        fi
    done
done