CURRENT_DIR=`pwd`
export ROOT_PATH=$CURRENT_DIR/../

python train.py \
    --root_path=$ROOT_PATH \
    --epoch_nums=20 \
    --gpu=0 \
    --task=train