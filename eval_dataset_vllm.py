import argparse
import os
import numpy as np
import torch
from vllm import LLM
from tqdm import tqdm
import time
import json

def find_first_zero_row(arr):
    is_row_all_zero = (arr == 0).all(axis=1)
    first_zero_index = np.argmax(is_row_all_zero)
    if first_zero_index == 0 and not is_row_all_zero[0]:
        return arr.shape[0]
    else:
        return first_zero_index

def main(args):
    merged_model_path = args.merged_model_path
    test_data_path = args.test_data_path
    batch_size = args.batch_size
    max_len = args.max_model_len
    npy_path = args.embedding_file_name
    output_dir = 'dir for saving node embedding'
    embedding_output_path = os.path.join(output_dir, npy_path)

    print("--- init vLLM model... ---")
    llm = LLM(
        model=merged_model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=max_len+1,
        task="reward"
    )
    tokenizer = llm.get_tokenizer()
    
    print("--- load test data... ---")
    with open(test_data_path, 'r') as r:
        test_dataset = json.load(r)
    
    def format_and_truncate_prompt(instruction, input_str, tokenizer, max_len):
        prompt = f"<s>[INST] {instruction}\n{input_str} [/INST]"
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) > max_len:
            truncated_token_ids = token_ids[:max_len]
            prompt = tokenizer.decode(truncated_token_ids, skip_special_tokens=True)
        return prompt

    all_prompts = [
        format_and_truncate_prompt(item['instruction'], item['input'], tokenizer, max_len)
        for item in tqdm(test_dataset, desc="preprocess test data")
    ]
    num_total_prompts = len(all_prompts)
    print(f"--- preprocess finish, have {num_total_prompts} prompts ---")

    hidden_size = llm.llm_engine.model_config.get_hidden_size()
    print(f"--- hidden size: {hidden_size} ---")
    
    start_index = 0
    if os.path.exists(embedding_output_path):
        print(f"--- has found file {embedding_output_path} ---")
        final_embeddings_memmap = np.memmap(embedding_output_path, dtype='float16', mode='r+', shape=(num_total_prompts, hidden_size))
        
        if final_embeddings_memmap.shape != (num_total_prompts, hidden_size):
            print(f"warning: existing file shape {final_embeddings_memmap.shape} different from { (num_total_prompts, hidden_size)}. Remove and create a new one.")
            del final_embeddings_memmap
            os.remove(embedding_output_path)
            final_embeddings_memmap = np.memmap(embedding_output_path, dtype='float16', mode='w+', shape=(num_total_prompts, hidden_size))
        else:
            print("--- find breakpoint... ---")
            start_index = find_first_zero_row(final_embeddings_memmap)
            print(f"--- find break point: {start_index}  ---")
            if start_index >= num_total_prompts:
                print("--- The file is complete, program will return ---")
                return
    else:
        print(f"--- not file {embedding_output_path}, crate a new one ---")
        final_embeddings_memmap = np.memmap(embedding_output_path, dtype='float16', mode='w+', shape=(num_total_prompts, hidden_size))
    
    print(f"--- memmap is ready, the shape is: {final_embeddings_memmap.shape} ---")
    
    print(f"--- start feature extraction, batch size: {batch_size} ---")
    start_time = time.time()
    
    saved_count = start_index 
    
    progress_bar = tqdm(
        range(start_index, num_total_prompts, batch_size), 
        desc="total batch num:",
        initial=start_index // batch_size,
        total=num_total_prompts // batch_size
    )

    for i in progress_bar:
        batch_prompts = all_prompts[i : i + batch_size]
        
        if not batch_prompts:
            continue

        batch_outputs = llm.encode(batch_prompts, use_tqdm=False)

        batch_embeddings = []
        valid_indices = [] 
        for idx, request_output in enumerate(batch_outputs):
            try:
                target_index = len(request_output.prompt_token_ids) - 1
                last_hidden_states = request_output.outputs.data
                sentence_embedding = last_hidden_states[target_index]
                batch_embeddings.append(sentence_embedding.cpu().numpy())
                valid_indices.append(i + idx)
            except (ValueError, IndexError):
                print(f"\nwarning: process {i+idx} failed, has skipped.")
                continue

        if batch_embeddings:
            batch_embeddings_np = np.array(batch_embeddings, dtype='float16')
            start_write_pos = saved_count
            end_write_pos = start_write_pos + len(batch_embeddings_np)
            final_embeddings_memmap[start_write_pos:end_write_pos] = batch_embeddings_np
            saved_count = end_write_pos

    final_embeddings_memmap.flush()
    print("\n--- memmap file has saved. ---")

    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*50)
    print("--- finished! ---")
    print(f"time consuming: {total_time:.2f}")
    print(f"total embedding: {saved_count} / {num_total_prompts}")
    print(f"embedding file path: {embedding_output_path}")
    print(f"shape of embedding: {final_embeddings_memmap.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="extract and save hidden states in position [EOS] of LLM")
    parser.add_argument("--merged_model_path", type=str, default="")
    parser.add_argument("--test_data_path", type=str, default='')
    parser.add_argument("--embedding_file_name", type=str, default='')
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_model_len", type=int, default=2047)
    args = parser.parse_args()
    main(args)