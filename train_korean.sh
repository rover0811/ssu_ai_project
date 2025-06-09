expdir=finetune_korean.0005_KB200_drop0.1_GPThid
mkdir -p exp/${expdir}
cd korean_processed2 && mkdir -p exp/${expdir} && python ../train.py \
    --modeltype medium \
    --train_json korean_lecture_bias.json \
    --dev_json korean_lecture_bias.json \
    --lr 0.0005 \
    --batch_size 1 \
    --log_interval 20 \
    --nepochs 10 \
    --warmup_pct 0.0 \
    --decay_pct 0.2 \
    --expdir exp/${expdir} \
    --logfile exp/${expdir}/log.txt \
    --accumgrad 10 \
    --biasing \
    --biasinglist korean_all_rare_words.txt \
    --dropentry 0.1 \
    --maxKBlen 100 \
    # --useGPT \
    # --GNNtype gcn2 \
    # --GNNdim 256 \
    # --loadfrom exp/finetune_librispeech_lr0.0005_KB200_drop0.1/model.acc.best \
