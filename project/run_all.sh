# model efficientnet_b0, mobilenet_v2, vgg16, vit_b_16
# my_vgg lr 2e-5/simple 5e-5 or 1e-4/complex

for lr in 2e-5 5e-5 1e-4
do
    for weight_decay in 0 1e-4 5e-4 1e-3
    do
        python3 main.py \
            --model vgg_multi_head_attention \
            --dataset complex \
            --pretrained False \
            --batch_norm False \
            --lr $lr \
            --weight_decay $weight_decay \
            --num_epochs 50
    done
done