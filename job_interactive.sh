python3 src/finetune.py \
    --data /data/jaygala/ssl-finetune-exps/data \
    --model ViT-B/32 \
    --lr 40 \
    --wd 0 \
    --epochs 10 \
    --save /data/jaygala/ssl-finetune-exps/prompt_exp \
    --use_wandb --run_name prompt_exp
