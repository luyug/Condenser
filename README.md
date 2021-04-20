# Condenser
Code for converting a pre-trained Transformer encoder LM into Condenser, a Transformer architecture specialized in bi-encoder training. Details can be found in our preprint, [Is Your Language Model Ready for Dense Representation Fine-tuning?](https://arxiv.org/abs/2104.08253).


Currently supported architectures include all models with BERT or RoBERTa architecture.


## Dependencies
The code uses the following packages,
```
pytorch
transformers
datasets
nltk
```

## Pre-processing
We first tokenize all the training text before running pre-training. The pre-processor expects one-paragraph per-line format. It will then run for each line sentence tokenizer to construct the final training data instances based on passed in `--max_len`. The output is a json file. We recommend first break the full corpus into shards.
```
for s in shard1, shard2, shardN
do
 python data/create_train.py \
  --tokenizer_name bert-base-uncased \
  --file $s \
  --save_to $JSON_SAVE_DIR \
  --max_len $MAX_LENGTH
done
```

## Pre-training
The following code lauch training on 4 gpus and train BERT variant `bert-base-uncased` into Condenser.
```
python -m torch.distributed.launch --nproc_per_node 4 run_pre_training.py \
  --output_dir $OUTDIR \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --save_steps 20000 \
  --per_device_train_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $ACCUMULATION_STEPS \
  --fp16 \
  --warmup_ratio 0.1 \
  --learning_rate 1e-4 \
  --num_train_epochs 8 \
  --overwrite_output_dir \
  --dataloader_num_workers 32 \
  --n_head_layers 2 \
  --skip_from 6 \
  --max_seq_length $MAX_LENGTH \
  --train_dir $JSON_SAVE_DIR \
  --weight_decay 0.01 \
  --late_mlm
```

## Fine-tuning
The saved model can be loaded directly using huggingface interface and fine-tuned,
```
from transformers import AutoModel
model = AutoModel.from_pretrained('path/to/train/output')
```
