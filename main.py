import argparse
import os
import datasets, transformers
import torch

from accelerate import Accelerator

from dataloader import *
from executor import *
from active_learning import *

datasets.logging.set_verbosity_error()


processors = {
    "biolinkbert": BioLinkBERTDataProcessor,
    "pubmedbert": PubMedBERTDataProcessor
}

parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument(
    '-m',
    '--model_name',
    required=True,
    default="linkbert",
    choices=list(processors.keys()),
    help = "The name of the model to choose, by default BioLinkBERT",
)

parser.add_argument(
    '-d',
    '--data_dir',
    required=True,
    help = "The input data dir. Should contain the .json files for the task.",
)

parser.add_argument(
    '-o',
    '--output_dir',
    required=True,
    help = "The output directory where the results and checkpoints will be written.",
)

parser.add_argument(
    '--max_seq_length',
    default=256,
    type=int,
    help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded."
)

## Script parameters
parser.add_argument(
    '--do_train',
    action='store_true',
    help="Whether to run training"
)

parser.add_argument(
    '--do_eval',
    action='store_true',
    help="Whether to run evaluation on the dev set"
)

parser.add_argument(
    '--do_predict',
    action='store_true',
    help="Whether to run the model in inference mode on the test set."
)

parser.add_argument(
    '--train_batch_size',
    default=2,
    type=int,
    help="Total batch size for training."
)

parser.add_argument(
    '--eval_batch_size',
    default=8,
    type=int,
    help="Total batch size for eval."
)

parser.add_argument(
    '--predict_batch_size',
    default=8,
    type=int,
    help="Total batch size for predict."
)

parser.add_argument(
    '--learning_rate',
    default=2e-05,
    type=float,
    help="The initial learning rate for Adam."
)

parser.add_argument(
    '--num_epochs',
    default=3,
    type=int,
    help="Total number of training epochs to perform"
)

parser.add_argument(
    '--warmup_proportion',
    default=0.5,
    type=float,
    help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training."
)

parser.add_argument(
    '--binary',
    action='store_true',
    help="Whether to transform the task to a binary classification"
)

## Active learning parameters
parser.add_argument(
    '--do_active',
    action='store_true',
    help="Whether to run the model in active learning mode."
)

parser.add_argument(
    '--size_seed',
    default=20,
    type=int,
    help="Number of examples used as the initial training set."
)

parser.add_argument(
    '--cross_val',
    default=1,
    type=int,
    help="Number of folds for the cross-validation."
)

parser.add_argument(
    '--num_repeats',
    default=1,
    type=int,
    help="Number of times we use a different seed for the same fold."
)

parser.add_argument(
    '--batch_size_active',
    default=1,
    type=int,
    help="Number of queries to select each loop of active learning."
)

parser.add_argument(
    '--random_seed',
    default=1234,
    type=int,
    help="Random seed to fix the experiments."
)

parser.add_argument(
    '--strategy',
    default=None,
    choices=list(STRATEGIES.keys()),
    help="Active Learning Strategy to use"
)

parser.add_argument(
    '--restart_model',
    action='store_true',
    help='Use this option if you want to train your model from scratch after each query selection.'
)

FLAGS = parser.parse_args()

def checks():
    if (FLAGS.do_train or FLAGS.do_eval or FLAGS.do_predict) and FLAGS.do_active:
        raise ValueError(
            "Your can only do active learning alone, without full training or prediction/validation.")

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict and not FLAGS.do_active:
        raise ValueError(
            "At least one of `do_train`, `do_eval`, `do_predict', `do_active` must be True.")

    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)



def get_processor(args):
    return processors[args.model_name](args.max_seq_length, args.data_dir, args.do_active, args.binary)

def main():

    processor = get_processor(FLAGS)
    executor = Executor(FLAGS,processor)
    accelerator = Accelerator(split_batches=True) #the batch is split across GPUs
    model = None

    if accelerator.is_main_process:
        transformers.logging.set_verbosity_warning()
    else:
        transformers.logging.set_verbosity_error()


    if accelerator.is_main_process:
        checks()

    if FLAGS.do_train:
        model = executor.get_new_model(model)
        dataloader = executor.get_dataloader_split('train', True)
        model = executor.train(model,dataloader, accelerator)
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(unwrapped_model.state_dict(),f"{FLAGS.output_dir}/trained_model.bin")

    if FLAGS.do_eval:
        if model is None:
            model = executor.get_new_model(model)
            model.load_state_dict(torch.load(f"{FLAGS.output_dir}/trained_model.bin", map_location="cpu"))
            model = accelerator.prepare(model)
        dataloader = executor.get_dataloader_split('eval')
        results = executor.eval(model, dataloader, accelerator)
        accelerator.print(results)

    if FLAGS.do_predict:
        if model is None:
            model = executor.get_new_model(model)
            model.load_state_dict(torch.load(f"{FLAGS.output_dir}/trained_model.bin", map_location="cpu"))
            model = accelerator.prepare(model)
        dataloader = executor.get_dataloader_split('test')
        results = executor.predict(model, dataloader, accelerator)
        if accelerator.is_main_process:
            executor.write_results(results)

    if FLAGS.do_active:
        wrapper_active_learning(executor, accelerator)


if __name__ == "__main__":
    main()