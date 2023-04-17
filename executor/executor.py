import torch
import gc

from torch.nn.functional import softmax
from tqdm.auto import tqdm
from transformers import get_scheduler

from models import *
from evaluation import *

class Executor:

    def __init__(self, args, processor):
        self.FLAGS = args
        self.processor = processor
        self.factory = ModelFactory(self.FLAGS.model_name, self.processor.get_labels(), self.FLAGS.output_dir)

    def get_new_model(self, model=None, cv = None):
        return self.factory.produce(model, cv)

    def save_model(self, model, cv):
        self.factory.save_model(model, cv)

    def get_num_classes(self):
        return self.factory.get_num_classes()

    def get_embed_dim(self):
        return self.factory.get_embed_dim()

    def get_dataloader_indices(self, indices=None, batch_size=8, shuffle=False):
        return self.processor.get_dataloader(self.processor.get_data(indices), batch_size, shuffle)

    def get_dataloader_split(self, split_name, shuffle=False):
        if split_name == 'train':
            data = self.processor.get_data_train()
            batch_size = self.FLAGS.train_batch_size
        elif split_name == 'test':
            data = self.processor.get_data_test()
            batch_size = self.FLAGS.predict_batch_size
        else:
            data = self.processor.get_data_eval()
            batch_size = self.FLAGS.eval_batch_size
        return self.processor.get_dataloader(data, batch_size, shuffle)

    def train(self, model, dataloader, accelerator, cv = None, from_scratch=False):

        with accelerator.main_process_first():
            if from_scratch or model is None:
                accelerator.clear()
                model = self.get_new_model(model, cv)
                gc.collect()
                torch.cuda.empty_cache()

        accelerator.wait_for_everyone()

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.FLAGS.learning_rate)
        num_training_steps = self.FLAGS.num_epochs * len(dataloader)
        num_warm_steps = int(num_training_steps * self.FLAGS.warmup_proportion)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=num_warm_steps,
            num_training_steps=num_training_steps,
        )
        progress_bar = tqdm(range(num_training_steps),disable=not accelerator.is_local_main_process)

        model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, dataloader, lr_scheduler
        )

        for epoch in range(self.FLAGS.num_epochs):
            model.train()
            for batch in dataloader:
                batch.pop('index')
                model.zero_grad(set_to_none=True)
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)

                #del outputs, loss
                #gc.collect()
                #torch.cuda.empty_cache()


        del optimizer, dataloader, lr_scheduler, progress_bar
        gc.collect()
        torch.cuda.empty_cache()

        return model

    def eval(self, model, dataloader, accelerator):
        model.eval()
        dataloader = accelerator.prepare(dataloader)
        progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
        metric = PRF1Metrics()
        ids_added = set()
        for batch in dataloader:
            with torch.no_grad():
                indices = batch.pop('index')
                outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)

                predictions, references, indices = accelerator.gather((predictions, batch["labels"], indices))
                predictions, references, indices = predictions.tolist(), references.tolist(), indices.tolist()
                # to avoid adding information about indices already present, due to the loop of indices in the dataloader
                new_predictions, new_references = [], []
                for i in range(len(predictions)):
                    if indices[i] not in ids_added:
                        new_predictions.append(predictions[i])
                        new_references.append(references[i])
                        ids_added.add(indices[i])

                metric.add_batch(predictions=new_predictions, references=new_references)
                progress_bar.update(1)

        accelerator.wait_for_everyone()

        return metric.compute(average="binary" if self.FLAGS.binary else "macro")

    def predict(self, model, dataloader, accelerator, predictions_only=False):
        model.eval()
        dataloader = accelerator.prepare(dataloader)
        progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
        results = []
        ids_added = set()
        for batch in dataloader:
            with torch.no_grad():
                indices = batch.pop('index')
                outputs = model(**batch)
                probabilities = softmax(outputs.logits, dim=1)
                predictions = torch.argmax(probabilities, dim=-1)
                if predictions_only:
                    predictions, indices = accelerator.gather((predictions, indices))
                    predictions, indices = predictions.tolist(), indices.tolist()
                    # to avoid adding information about indices already present, due to the loop of indices in the dataloader
                    new_predictions = []
                    for i in range(len(predictions)):
                        if indices[i] not in ids_added:
                            new_predictions.append(predictions[i])
                            ids_added.add(indices[i])
                    results.extend(new_predictions)
                else:
                    probabilities, predictions, indices = accelerator.gather((probabilities, predictions, indices))
                    probabilities, predictions, indices = probabilities.tolist(), predictions.tolist(), indices.tolist()
                    # to avoid adding information about indices already present, due to the loop of indices in the dataloader
                    new_predictions, new_probabilities = [], []
                    for i in range(len(predictions)):
                        if indices[i] not in ids_added:
                            new_predictions.append(predictions[i])
                            new_probabilities.append(probabilities[i])
                            ids_added.add(indices[i])
                    results.extend(zip(new_probabilities, new_predictions))

                    del new_probabilities
                progress_bar.update(1)

                del probabilities, predictions, outputs, new_predictions
                gc.collect()

        return results

    def predict_prob(self, model, dataloader, accelerator):
        model.eval()
        dataloader = accelerator.prepare(dataloader)
        progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
        results = []
        ids_added = set()
        for batch in dataloader:
            with torch.no_grad():
                indices = batch.pop('index')
                outputs = model(**batch)
                probabilities = softmax(outputs.logits, dim=1)
                probabilities, indices = accelerator.gather((probabilities, indices))
                probabilities, indices = probabilities.tolist(), indices.tolist()

                # to avoid duplicates due to dataloader looping
                new_probabilities, new_indices = [],[]
                for i in range(len(probabilities)):
                    index = indices[i]
                    if index not in ids_added:
                        new_probabilities.append(probabilities[i])
                        ids_added.add(index)
                        new_indices.append(index)
                results.extend(zip(new_probabilities,new_indices))
                progress_bar.update(1)

                del probabilities, outputs, indices, new_indices, new_probabilities
                gc.collect()


        return results

    def get_embeddings(self, model, dataloader, accelerator):
        model.eval()
        dataloader = accelerator.prepare(dataloader)
        progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
        embeddings = []
        ids_added = set()
        for batch in dataloader:
            with torch.no_grad():
                indices = batch.pop('index')
                outputs = model(**batch)
                embedding = torch.mean(outputs.hidden_states[-1], dim=1).squeeze()
                embedding, indices = accelerator.gather((embedding, indices))
                embedding, indices = embedding.tolist(), indices.tolist()

                new_embeddings, new_indices = [], []
                for i in range(len(embedding)):
                    index = indices[i]
                    if index not in ids_added:
                        new_embeddings.append(embedding[i])
                        ids_added.add(index)
                        new_indices.append(index)
                embeddings.extend(zip(new_embeddings, new_indices))

                del embedding, outputs, indices, new_embeddings, new_indices
                gc.collect()

                progress_bar.update(1)

        return embeddings

    def write_results(self, results):
        output_predict_file = f"{self.FLAGS.output_dir}/predictions.tsv"
        with open(output_predict_file, "w") as writer:
            for proba, prediction in results:
                output_line = f'{",".join(str(class_probability) for class_probability in proba)}\t{prediction}\t{self.factory.id2label[prediction]}\n'
                writer.write(output_line)