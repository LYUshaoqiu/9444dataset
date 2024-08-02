export CUDA_VISIBLE_DEVICES=1

# vgg attention simple transform
# python3 main.py \
#     --model vgg_attention \
#     --dataset complex \
#     --pretrained False \
#     --batch_norm False \
#     --lr 5e-5 \
#     --weight_decay 1e-3 \
#     --num_epochs 50 \
#     --patience 10 \
#     --transform simple

# vgg attention complex transform
# python3 main.py \
#     --model vgg_attention \
#     --dataset complex \
#     --pretrained False \
#     --batch_norm False \
#     --lr 1e-4 \
#     --weight_decay 2e-4 \
#     --num_epochs 50 \
#     --patience 10 \
#     --transform complex
