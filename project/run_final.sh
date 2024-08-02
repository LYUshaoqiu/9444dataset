export CUDA_VISIBLE_DEVICES=5

python3 main_data.py \
    --model vgg_attention \
    --dataset complex \
    --pretrained False \
    --batch_norm False \
    --lr 1e-4 \
    --weight_decay 5e-4 \
    --num_epochs 120 \
    --patience 60 \
    --transform complex