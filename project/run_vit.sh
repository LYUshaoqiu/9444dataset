# model efficientnet_b0, mobilenet_v2, vgg16, vit_b_16
# ----------------------------------------
# Model Name  | lr (simple) | lr (complex)
# ----------------------------------------
# vgg_origin  | 2e-5        | 5e-5 or 1e-4
# ----------------------------------------
# vgg_complex | 0.0001      | 5e-5 or 1e-4

export CUDA_VISIBLE_DEVICES=7

# vit baseline simple
# python3 main.py \
#     --model vit_b_16 \
#     --dataset simple \
#     --pretrained False \
#     --batch_norm False \
#     --lr 2e-5 \
#     --weight_decay 5e-4 \
#     --num_epochs 50 \
#     --patience 10 \
#     --transform simple

# vit baseline complex
python3 main.py \
    --model vit_b_16 \
    --dataset complex \
    --pretrained False \
    --batch_norm False \
    --lr 1e-4 \
    --weight_decay 5e-3 \
    --num_epochs 50 \
    --patience 10 \
    --transform simple
