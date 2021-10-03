ARCH=vgg16_bn
LOSS=sat
DATASET=cifar10
REWARD=2.2
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_$1
GPU_ID=1


### eval
python -u train.py --arch ${ARCH} --gpu-id ${GPU_ID} \
       --loss ${LOSS} --reward ${REWARD} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       >> ${SAVE_DIR}.log  2>&1
