# Hyperparameters

## Baseline

### Simple Dataset

#### vgg simple dataset simple transform
dataset='simple', model='vgg_origin', pretrained=False, lr=2e-05, weight_decay=0.0005, num_epochs=50, patience=10, batch_norm=False, transform='simple'

#### vit simple dataset simple transform
dataset='simple', model='vit_b_16', pretrained=False, lr=2e-05, weight_decay=0.0005, num_epochs=50, patience=10, batch_norm=False, transform='simple'

#### resnet simple dataset simple transform
dataset='simple', model='resnet50', pretrained=False, lr=1e-05, weight_decay=0.0005, num_epochs=50, patience=10, batch_norm=False, transform='simple'

### Complex Dataset

#### vgg simple transform
dataset='complex', model='vgg_origin', pretrained=False, lr=5e-05, weight_decay=0.0005, num_epochs=50, patience=10, batch_norm=False, transform='simple'
















## Enhanced Model

#### vgg complex transform
dataset='complex', model='vgg_attention', pretrained=False, lr=0.0001, weight_decay=0.0002, num_epochs=50, patience=10, batch_norm=False, transform='complex'

#### vgg attention complex transform

#### vgg multi head complex transform
