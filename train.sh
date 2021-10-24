ARCH=vgg16_bn
LOSS=sat
DATASET=cifar10
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_$1
GPU_ID=0

mkdir -p ./log

### train
python -u train.py --arch ${ARCH} --gpu-id ${GPU_ID} --pretrain 0 --sat-momentum 0.99 \
       --loss ${LOSS} \
       --dataset ${DATASET} --save ${SAVE_DIR}  \
       2>&1 | tee -a ${SAVE_DIR}.log
