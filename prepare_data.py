from tokenizer import tokenizer
import datasets
from datasets import load_dataset

d = datasets = load_dataset("text", data_files={"train": 'data/train.json', 
                                                "test": 'data/test.json'})
def tokenize_function(examples):
    return tokenizer(examples["text"])
tokenized_d = d.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

block_size = 256
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

prepared_dataset = tokenized_d.map(
    group_texts,
    batched=True,
    num_proc=4
)

print(prepared_dataset)
prepared_dataset.save_to_disk("data/prepared_dataset")









