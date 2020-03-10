#!/usr/bin/sh

TXT_FILE="sample/pretrain_sample.txt"
NPY_FILE="sample/pretrain_sample.non_text_feat.npy"
SAVE_PATH="save_dir/my_vilbert_sample"
LOG_FILE="results/log/log.txt"

FROM_PRETRAINED="/path/to/pretrained/bert/dir"
VOCAB_FILE="/path/to/pretrained/bert/dir/vocab.txt"
CONFIG="config/my_bert_sample_6layer_6conect.json"
WEIGHT_NAME_FILE="config/bert-base-uncased_weight_name.json"

GPU=0
EPOCH=30
BATCH=16

python src/main.py --mode pretrain --config $CONFIG --from_pretrained $FROM_PRETRAINED --vocab_file $VOCAB_FILE --pretrain_text_file $TXT_FILE --pretrain_non_text_feature_file $NPY_FILE --bert_weight_name_file $WEIGHT_NAME_FILE --train_batch_size $BATCH --gpu $GPU --num_training_epochs $EPOCH --save_path $SAVE_PATH --log_file $LOG_FILE
