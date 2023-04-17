import random
import gc
import torch
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from transformers import set_seed

from distil.active_learning_strategies import *


from os import mkdir
from os.path import exists


STRATEGIES = {
    'random': RandomSampling,
    'least_confident': LeastConfidenceSampling,
    'entropy': EntropySampling,
    'margin': MarginSampling,
    'coreset':CoreSet,
    'batchBALD':BatchBALDDropout,
}


def write_file(examples, data_dir, name):
    with open(f'{data_dir}{name}', 'w') as outfile:
        for example in examples:
            example = "\t".join(example)
            outfile.write(f'{example}\n')


def write_results(data_dir, repeat, strategy, performances, labeled, first=False):
    if first:
        with open(f'{data_dir}repeat{repeat}_{strategy}.tsv', 'w') as outfile:
            outfile.write(
            f'accuracy\tf1_score\tprecision\trecall\tsize_labeled\tlabeled\n')
    else:

        with open(f'{data_dir}repeat{repeat}_{strategy}.tsv', 'a') as outfile:
            outfile.write(
            f'{performances["accuracy"]}\t{performances["f1"]}\t{performances["precision"]}\t{performances["recall"]}\t{len(labeled)}\t{labeled.tolist()}\n')


def active_loop(strategy, batch_size, indices_labeled, indices_unlabeled, test_labels, executor, fold, repeat, data_dir, from_scratch, accelerator):
    """
    Main loop of the active learning with a specific strategy
    """


    args = {
        "batch_size" : executor.FLAGS.predict_batch_size,
        "executor" : executor,
        "accelerator" : accelerator,
        "device" : accelerator.device
    }

    if strategy == 'batchBALD':
        args.update({
            "mod_inject" : 'classifier', #linear layer called classifier in BERTForSequenceClassification
            "n_drop":10
        })

    parameters = {
        'labeled_dataset': indices_labeled,
        'unlabeled_dataset': indices_unlabeled,
        'net': None,
        'nclasses': executor.get_num_classes(),
        'args': args
    }

    selector = STRATEGIES[strategy](**parameters)

    accelerator.print(f"**** Fold {fold} - Repeat {repeat} - With strategy {strategy} ****")

    model = executor.get_new_model(None, fold)
    model = accelerator.prepare(model)
    selector.update_model(model)

    while len(indices_unlabeled) > 0:
        accelerator.print("*** SELECTING THE QUERIES ***")

        # selection part and update of the indices labeled and unlabeled
        if len(indices_unlabeled) > batch_size:
            selected_indices = selector.select(budget=batch_size)
        else:
            selected_indices = indices_unlabeled
        indices_labeled = np.append(indices_labeled, selected_indices)
        indices_unlabeled = np.setdiff1d(indices_unlabeled, selected_indices, assume_unique=True)

        selector.update_data(indices_labeled, indices_unlabeled)

        accelerator.print("*** TRAINING ***")
        # erase reference to old model
        selector.update_model(None)

        # no shuffle for training because we want to show that it is the indices added by the strategy at the origin
        # of the performance and not the order of the indices
        dataloader = executor.get_dataloader_indices(indices_labeled, executor.FLAGS.train_batch_size, True)
        model = executor.train(model, dataloader, accelerator, fold, from_scratch=from_scratch)
        selector.update_model(model)

        # save results performance
        accelerator.print("*** SAVING THE PERFORMANCE ***")
        dataloader = executor.get_dataloader_indices(test_labels, batch_size=executor.FLAGS.eval_batch_size)
        results = executor.eval(model, dataloader, accelerator)
        accelerator.wait_for_everyone()
        accelerator.print(results)
        if accelerator.is_main_process:
            write_results(data_dir, repeat, strategy, results, indices_labeled)

        # salt and burn cleaning
        del selected_indices, results, dataloader
        gc.collect()
        torch.cuda.empty_cache()

    accelerator.clear()


def wrapper_active_learning(executor, accelerator):
    FLAGS = executor.FLAGS
    random.seed(FLAGS.random_seed)
    set_seed(FLAGS.random_seed)

    skf = StratifiedKFold(n_splits=FLAGS.cross_val, shuffle=True,random_state=FLAGS.random_seed)

    accelerator.print("**** PREPARING THE DATA AND THE INITIAL MODEL ****")

    Y = executor.processor.get_data().with_format('numpy')['labels']

    # create initial model
    if accelerator.is_main_process:
        _ = executor.get_new_model(None)
        del _
        gc.collect()

    accelerator.wait_for_everyone()

    fold = 1
    for train, test in skf.split(np.arange(0,len(Y)), Y):
        output_dir = f'{FLAGS.output_dir}/{fold}/'
        if accelerator.is_main_process:
            if not exists(output_dir):
                mkdir(output_dir)

        for repeat in range(1, FLAGS.num_repeats + 1):
            indices_labeled = resample(
                np.arange(0,len(train)),
                replace = False,
                n_samples = FLAGS.size_seed,
                random_state = random.randint(0,1000)
            )
            indices_unlabeled = np.delete(train, indices_labeled)
            indices_labeled = train[indices_labeled]

            accelerator.print("**** INITIAL TRAINING ****")
            #load the initial model and train with the initial training seed
            dataloader = executor.get_dataloader_indices(indices_labeled, FLAGS.train_batch_size, True)
            model = executor.train(None, dataloader, accelerator, cv = None)
            if accelerator.is_main_process:
                executor.save_model(model, cv = fold)
            accelerator.print("*** SAVING THE PERFORMANCE ***")
            dataloader = executor.get_dataloader_indices(test, batch_size=FLAGS.eval_batch_size)
            first_results = executor.eval(model, dataloader, accelerator)

            del model, dataloader
            accelerator.clear()


            if FLAGS.strategy is not None:
                if accelerator.is_main_process:
                    write_results(output_dir, repeat, FLAGS.strategy, None, indices_labeled, first=True)
                    write_results(output_dir, repeat, FLAGS.strategy, first_results, indices_labeled)

                active_loop(FLAGS.strategy, FLAGS.batch_size_active, indices_labeled, indices_unlabeled, test, executor, fold, repeat,
                            output_dir, FLAGS.restart_model, accelerator)
            else:
                for strategy in STRATEGIES:
                    if accelerator.is_main_process:
                        write_results(output_dir, repeat, strategy, None, indices_labeled, first=True)
                        write_results(output_dir, repeat, strategy, first_results, indices_labeled)

                    active_loop(strategy, FLAGS.batch_size_active, indices_labeled, indices_unlabeled, test, executor, fold, repeat,
                                output_dir, FLAGS.restart_model, accelerator)

        # update fold number
        fold += 1
