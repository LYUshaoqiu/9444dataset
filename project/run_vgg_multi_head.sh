export CUDA_VISIBLE_DEVICES=2

# vgg multi simple transform
python3 main.py \
    --model vgg_multi_head_attention \
    --dataset complex \
    --pretrained False \
    --batch_norm False \
    --lr 5e-5 \
    --weight_decay 1e-3 \
    --num_epochs 50 \
    --patience 10 \
    --transform simple

# vgg multi complex transform
# python3 main.py \
#     --model vgg_multi_head_attention \
#     --dataset complex \
#     --pretrained False \
#     --batch_norm False \
#     --lr 5e-5 \
#     --weight_decay 1e-3 \
#     --num_epochs 50 \
#     --patience 10 \
#     --transform complex
