cd korean_processed && \
expdir=exp/finetune_korean.0007_KB200_drop0.1_GPThid && \
testset=other && \
decodedir=decode_no_lm_b50_KB1000_${testset}_50best && \
mkdir -p $expdir/$decodedir && \
python ../decode.py \
    --test_json korean_lecture_bias_small.json \
    --beamsize 5 \
    --expdir $expdir/$decodedir \
    --loadfrom $expdir/model.acc.best \
    --biasing \
    --biasinglist korean_rareword_error.txt \
    --dropentry 0.0 \
    --maxKBlen 100 \
    --save_nbest \
    # --use_gpt2 \
    # --lm_weight 0.01 \
    # --ilm_weight 0.005 \
