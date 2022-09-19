CURRENT_DIR=`pwd`
export ROOT_PATH=$CURRENT_DIR/../

python test.py \
    --root_path=$ROOT_PATH \
    --gpu=0 \
    --task=test \
    --model_name=...