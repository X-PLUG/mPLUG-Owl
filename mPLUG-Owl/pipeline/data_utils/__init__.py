from .processors.builder import build_processors
from .xgpt3_dataset import MultiModalDataset

def train_valid_test_datasets_provider(data_path, config, tokenizer, seq_length=1024):
    """Build train and valid datasets."""
    print('> building train and validation datasets for mPLUG-Owl ...')
    train_ds, valid_ds = build_train_valid_test_datasets(
        input_file=data_path,  
        tokenizer=tokenizer,
        max_length=seq_length, 
        config=config)  
    print("> finished creating mPLUG-Owl datasets ...")

    return train_ds, valid_ds

def build_train_valid_test_datasets(input_file, tokenizer, max_length=80, config=None):
    train_processors = build_processors(config['train_processors'])
    valid_processors = build_processors(config['valid_processors'])

    assert len(input_file) == 2 # If you have files more than 2, modify code at here or merger them into train and dev
    train_ds = MultiModalDataset(input_file[0], tokenizer, train_processors, max_length)
    valid_ds = MultiModalDataset(input_file[1], tokenizer, valid_processors, max_length)
    test_ds = None
    return (train_ds, valid_ds)
