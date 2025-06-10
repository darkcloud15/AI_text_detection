

import os

from transformers import AutoTokenizer,AutoModelForSequenceClassification, DataCollatorWithPadding
from accelerate import Accelerator

import torch
from torch.utils.data import DataLoader
import numpy as np

import webdataset as wds

import argparse
from tqdm import tqdm


def sample_data(sample):
    return {"input_ids": sample[0], "attention_mask" : np.ones_like(sample[0]), "labels" : int(sample[1])}


def train_model(accelerator, model, optimizer, train_dataloader, eval_dataloader, args):
    device = accelerator.device
    
    train_acc = 0.0
    train_loss = 0.0
    eval_metrics = [0]
    for train_step, batch in enumerate(tqdm(train_dataloader)):
        optimizer.step()

        outputs = model(**batch.to(device))
        
        loss = outputs.loss
        train_loss += loss.item()
        train_acc += (outputs.logits.argmax(-1) == batch["labels"]).sum()
        
        accelerator.backward(loss)
        optimizer.step()

        if train_step % args.eval_steps == 0:
            eval_acc = 0.0
            eval_loss = 0.0
            with torch.no_grad():
                for eval_step, batch in enumerate(tqdm(eval_dataloader)):
                    
                    outputs = model(**batch.to(device))
                    loss = outputs.loss
                    
                    eval_loss += loss.item()
                    eval_acc += (outputs.logits.argmax(-1) == batch["labels"]).sum()


                if accelerator.is_main_process:
    
                    train_loss /= train_step
                    eval_loss /= eval_step
    
                    train_acc /= (train_step * args.bs) 
                    eval_acc /= (eval_step * args.bs)

                    if eval_acc > eval_metrics[-1]:
                        torch.save(model.state_dict(), "best_model.pt")
                        eval_metrics.append(evaL_acc)
                    print(f"step {train_step} train_acc {train_acc} eval_acc {eval_acc} train_loss {train_loss} eval_loss {eval_loss} ")




def main ():    
    parser = argparse.ArgumentParser()
   
    parser.add_argument("--model", type = str)
    parser.add_argument("--eval_steps", type = int)
    parser.add_argument("--bs", type = int)
    parser.add_argument("--lr", type = float)
    
    args = parser.parse_args()

    
    tokenizer  = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

    train_dataset = wds.WebDataset("train_{00000..00003}.tar", shardshuffle =  False, resampled = True, workersplitter = wds.split_by_worker, nodesplitter = wds.split_by_node).decode().to_tuple("npy", "cls").map(sample_data)
    eval_dataset = wds.WebDataset("eval_{00000..00003}.tar", shardshuffle =  False, resampled = False, workersplitter = wds.split_by_worker, nodesplitter = wds.split_by_node).decode().to_tuple("npy", "cls").map(sample_data)
        
    train_dataloader =  DataLoader(train_dataset, batch_size = args.bs, num_workers = 2, collate_fn = data_collator)
    eval_dataloader =  DataLoader(eval_dataset, batch_size = 32, num_workers = 2, collate_fn = data_collator)

    accelerator = Accelerator()
    # eval_dataloader = accelerator.prepare(eval_dataloader)
    
    # for idx, sample in enumerate(eval_dataloader):
    #     print(idx, sample)
    #     if idx == 3:
    #         break

    
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels = 2)
    model.config.pad_token_id = model.config.eos_token_id

    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)
    
    model, optimizer = accelerator.prepare(model, optimizer)
    train_model(accelerator, model, optimizer, train_dataloader, eval_dataloader, args)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    
if __name__ == "__main__":
    main()
