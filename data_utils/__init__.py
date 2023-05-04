from data_utils.processors.builder import build_processors
from .xgpt3_dataset import MultiModalDataset
from utils import get_tokenizer, print_rank_0, get_args

def train_valid_test_datasets_provider(data_path, iters_per_epoch, config):
    """Build train, valid, and test datasets."""
    args = get_args()
    print_rank_0('> building train, validation, and test datasets '
                 'for XGPT3 ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        input_file=data_path,
        tokenizer=get_tokenizer(),
        samples_per_epoch=iters_per_epoch * args.micro_batch_size if iters_per_epoch else None,
        eval_samples=args.micro_batch_size * args.eval_iters,
        max_completion_length=args.max_completion_length if args.max_completion_length is not None else args.seq_length,
        max_length=args.seq_length,
        config=config,
        start_from_scratch=args.start_from_scratch,
        splits=args.domain_splits)
    print_rank_0("> finished creating XGPT3 datasets ...")

    return train_ds, valid_ds, test_ds

def build_train_valid_test_datasets(input_file, tokenizer, max_completion_length=80, max_length=80, config=None, **kwargs):
    train_processors = build_processors(config['train_processors'])
    valid_processors = build_processors(config['valid_processors'])

    assert len(input_file) == 2 # If you have files more than 2, modify code at here or merger them into train and dev
    train_ds = MultiModalDataset(input_file[0], tokenizer, train_processors, max_completion_length, max_length)
    valid_ds = MultiModalDataset(input_file[1], tokenizer, valid_processors, max_completion_length, max_length)
    test_ds = None
    return (train_ds, valid_ds, test_ds)


