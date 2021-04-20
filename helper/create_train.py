import argparse
import os
import json
from multiprocessing import Pool
import random

from tqdm import tqdm
from transformers import AutoTokenizer
import nltk
nltk.download("punkt", quiet=True)

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str)
parser.add_argument('--save_to', type=str)
parser.add_argument('--column', type=int, help="take specified column")
parser.add_argument('--tokenizer_name', required=True)
parser.add_argument('--max_len', type=int, default=512)
parser.add_argument('--chunksize', type=int, default=500)
parser.add_argument('--short_sentence_prob', type=float, default=0.1)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
file_name = os.path.split(args.file)[1]

target_length = args.max_len - tokenizer.num_special_tokens_to_add(pair=False)


def encode_one_line(text: str):
    if args.column is not None:
        text = text.split('\t')[args.column]
    blocks = []
    sentences = nltk.sent_tokenize(text)
    ids = [
        tokenizer(
            sent,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"] for sent in sentences
    ]
    curr_len = 0
    curr_block = []

    curr_tgt_len = target_length if random.random() > args.short_sentence_prob else random.randint(1, target_length)

    for sent in ids:
        if curr_len + len(sent) > curr_tgt_len and curr_len > 0:
            blocks.append(curr_block)
            curr_block = []
            curr_len = 0
            curr_tgt_len = target_length if random.random() > args.short_sentence_prob \
                else random.randint(1, target_length)
        curr_len += len(sent)
        curr_block.extend(sent)
    if len(curr_block) > 0:
        blocks.append(curr_block)
    return blocks


with open(args.file, 'r') as corpus_file:
    lines = corpus_file.readlines()

with open(os.path.join(args.save_to, file_name + '.json'), 'w') as tokenized_file:
    with Pool() as p:
        all_blocks = p.imap_unordered(
            encode_one_line,
            tqdm(lines),
            chunksize=args.chunksize,
        )
        for blocks in all_blocks:
            for block in blocks:
                tokenized_file.write(json.dumps({'text': block}) + '\n')
