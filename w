python train.py \
    --data_json data/dendrite_v5/transforms_00_to_29.json \
    --max_iter 15000 \
    --monotonicity_weight 1.0 \
    --tv_weight 1e-5 \
    --background_loss_weight 2.0 \
    --foreground_loss_weight 1.0 \
    --silhouette_threshold 0.95 \
    --output_dir ./output \
    --render_output_dir ./renders


python export.py \
    --checkpoint ./output/checkpoints/best_psnr.ckpt \
    --export_volume \
    --times 0.0 0.25 0.5 0.75 1.0 \
    --resolution 128 \
    --gt_dir data/dendrite_v5/ground_truth \
    --gt_data_range 3.0 \
    --output_dir ./exports