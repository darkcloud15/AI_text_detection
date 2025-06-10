
import os
import argparse

import numpy as np

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

import webdataset as wds

from tqdm import tqdm

def make_texts(sample, _type):
    sample["text"] = sample[_type + "_text"]
    if _type == "human":
        sample["labels"] = 0
    else :
        sample["labels"] = 1
    return sample


def make_shards(dataset, maxcount, split, tokenizer):
    with wds.ShardWriter(split + "_%05d.tar", maxcount = maxcount) as sink:
        for idx, sample in enumerate(tqdm(dataset)):
            input_ids = tokenizer(sample["text"], return_tensors = 'np', truncation = True).input_ids.squeeze()
            sink.write({
                "__key__" : f"sample_{sample['idx']}",
                "npy" : input_ids,
                "cls" : str(sample["labels"]),
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str)
    args = parser.parse_args()

    os.makedirs("tokenized_texts", exist_ok = True)

    ds = load_dataset("dmitva/human_ai_generated_text", split = "train[:100000]", num_proc = 4)
    human_texts = ds.map(make_texts, fn_kwargs = {"_type" : "human"},remove_columns = ds.column_names)
    ai_texts = ds.map(make_texts, fn_kwargs = {"_type" : "ai"}, remove_columns = ds.column_names)

    data = concatenate_datasets([human_texts, ai_texts])
    data = data.add_column("idx", list(range(len(data))))

    train_test_split = data.train_test_split(test_size = 0.2)
    train = train_test_split["train"]
    val = train_test_split["test"]

    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    make_shards(train, maxcount = 20_000, split = "train", tokenizer = tokenizer)
    make_shards(val, maxcount = 5_000, split = "eval", tokenizer = tokenizer)

if __name__ == "__main__":
    main()
