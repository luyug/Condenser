# Condenser
Code for Condenser family, Transformer architectures for dense retrieval pre-training. Details can be found in our papers, [Condenser: a Pre-training Architecture for Dense Retrieval](https://arxiv.org/abs/2104.08253) and [Unsupervised Corpus Aware Language Model Pre-training for Dense Passage Retrieval
](https://arxiv.org/abs/2108.05540).


Currently supports all models with BERT or RoBERTa architecture.

## Resource
### Pre-trained Models
Headless Condenser can be retrived from Huggingface Hub using the following identifier strings.
- `Luyu/condenser`: Condenser pre-trained on BookCorpus and Wikipedia 
- `Luyu/co-condenser-wiki`: coCondenser pre-trained on Wikipedia 
- `Luyu/co-condenser-marco`: coCondenser pre-trained on MS-MARCO collection

For example, to load Condenser weights,
```
from transformers import AutoModel
model = AutoModel.from_pretrained('Luyu/condenser')
```

*Models with head will be adde soon after we decided where to host them.*

## Fine-tuning
The saved model can be loaded directly using huggingface interface and fine-tuned,
```
from transformers import AutoModel
model = AutoModel.from_pretrained('path/to/train/output')
```
The head will then be automatically omitted in fine-tuninig.

- For reproducing open QA experiments on NQ/TriviaQA, you can use the DPR toolkit and set `--pretrained_model_cfg` to a Condenser checkpoint. If GPU memory is an issue running DPR, you can alternatively use our [GC-DPR](https://github.com/luyug/GC-DPR) toolkit, which allows limited memory setup to train DPR without performance sacrifice.
- For supervised IR on MS-MARCO, you can use our [Tevatron](https://github.com/texttron/tevatron/tree/main/examples/coCondenser-marco) toolkit (an official version of our Dense prototype toolkit). We will also add open QA examples and pre-processing code to Dense soon.

## Dependencies
The code uses the following packages,
```
pytorch
transformers
datasets
nltk
```

## Condenser Pre-training
### Pre-processing
We first tokenize all the training text before running pre-training. The pre-processor expects one-paragraph per-line format. It will then run for each line sentence tokenizer to construct the final training data instances based on passed in `--max_len`. The output is a json file. We recommend first break the full corpus into shards.
```
for s in shard1, shard2, shardN
do
 python helper/create_train.py \
  --tokenizer_name bert-base-uncased \
  --file $s \
  --save_to $JSON_SAVE_DIR \
  --max_len $MAX_LENGTH
done
```
### Pre-training
The following code lauch training on 4 gpus and train Condenser warm starting from BERT (`bert-base-uncased`) .
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

## coCondenser Pre-training
### Pre-processing
First tokenize all the training text before running pre-training. The pre-processor expects one training document per line, with document broken into spans, e.g.
```
{'spans': List[str]}
...
```
We recommend breaking the full corpus into shards. Then run tokenization script,
```
for s in shard1, shard2, shardN
do
 python helper/create_train_co.py \
  --tokenizer_name bert-base-uncased \
  --file $s \
  --save_to $JSON_SAVE_DIR
done
```
### Pre-training
Launch training with the following script. Our experiments in the paper warm start the coCondenser (both head and backbone) from a Condenser checkpoint.
```
python -m torch.distributed.launch --nproc_per_node $NPROC run_co_pre_training.py \
  --output_dir $OUTDIR \
  --model_name_or_path /path/to/pre-trained/condenser/model \
  --do_train \
  --save_steps 20000 \
  --model_type bert \
  --per_device_train_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps 1 \
  --fp16 \
  --warmup_ratio 0.1 \
  --learning_rate 1e-4 \
  --num_train_epochs 8 \
  --dataloader_drop_last \
  --overwrite_output_dir \
  --dataloader_num_workers 32 \
  --n_head_layers 2 \
  --skip_from 6 \
  --max_seq_length $MAX_LENGTH \
  --train_dir $JSON_SAVE_DIR \
  --weight_decay 0.01 \
  --late_mlm
```
Having `NPROC x BATCH_SIZE` to be large is critical for effective contrastive pre-training. It is set to roughly 2048 in our experiments.
*Warning: gradient_accumulation_steps should be kept at 1 as accumulation cannot emulate large batch for contrative loss.*

If total GPU memory is bottlnecking, you may consider using gradient cached update. Download and install `GradCache` package from its [repo](https://github.com/luyug/GradCache). Then set additional command line argument `--cache_chunk_size` to be the desired sub-batch size. More about grad cache can be found in its paper, [Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup](https://arxiv.org/abs/2101.06983).
```
@inproceedings{gao2021scaling,
     title={Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup},
     author={Luyu Gao, Yunyi Zhang, Jiawei Han, Jamie Callan},
     booktitle ={Proceedings of the 6th Workshop on Representation Learning for NLP},
     year={2021},
}
```
