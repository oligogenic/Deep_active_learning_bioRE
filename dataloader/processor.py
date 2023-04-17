from transformers import AutoTokenizer, DefaultDataCollator
from datasets import load_dataset, Features, ClassLabel, Value

from torch.utils.data import DataLoader

class DataProcessor:

    def __init__(self, origin, max_seq_length, data_dir, all_in_one=False, binary=False):
        self.tokenizer = self.get_tokenizer(origin)
        self.binary = binary
        self.max_seq_length = max_seq_length
        self.data, self.labels = self.import_data(data_dir, all_in_one)
        self.convert_data_to_torch()
        if binary:
            self.labels = ['No relation', 'Relation']

    def convert_data_to_torch(self):
        self.data = self.data.map(
            self.process_data,
            batched=True,
            remove_columns=['sentence', 'id', 'label']
        )
        self.data = self.data.add_column('index',[i for i in range(len(self.data))])
        self.data.set_format("torch")

    def get_data(self, indices = None):
        if indices is None:
            return self.data
        return self.data.select(indices)

    def get_data_train(self):
        return self.data['train']

    def get_data_test(self):
        return self.data['test']

    def get_data_eval(self):
        return self.data['validation']

    def get_dataloader(self, data, batch_size, shuffle=False):
        #collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        collator = DefaultDataCollator()
        return DataLoader(
            data,
            batch_size = batch_size,
            shuffle = shuffle,
            collate_fn=collator,
            pin_memory=True
        )

    def get_tokenizer(self, origin):
        return AutoTokenizer.from_pretrained(origin)

    def get_labels(self):
        return self.labels

    def get_num_classes(self):
        return len(self.labels)

    def process_data(self, example):
        # we have to do it padding full here and not dynamic because the predict function in
        # strategy does not do the collate_fn
        tokenized = self.tokenizer(
            example["sentence"],
            truncation = True,
            max_length = self.max_seq_length,
            padding="max_length"
        )
        if self.binary:
            label = [1 if l > 0 else 0 for l in example['label']]
        else:
            label = example['label']
        tokenized['labels'] = label
        return tokenized

    @staticmethod
    def import_data(data_dir, all_in_one):
        data_files = {
            'train': f"{data_dir}/train.json",
            'validation': f"{data_dir}/dev.json",
            'test': f"{data_dir}/test.json"
        }

        if all_in_one:
            split = 'train+test+validation'
        else:
            split = None

        if 'DDI' in data_dir:
            class_names = ["0", "DDI-mechanism", "DDI-effect", "DDI-advise", "DDI-int"]
        elif 'chemprot' in data_dir:
            class_names = ["0", "CPR:4", "CPR:6", "CPR:5", "CPR:9", "CPR:3"]
        elif 'nary' in data_dir:
            class_names = ["None", "resistance", "response", "resistance or non-response", "sensitivity"]
        elif 'drugprot' in data_dir:
            class_names = ["None", 'AGONIST', 'ANTAGONIST', 'SUBSTRATE', 'AGONIST-INHIBITOR', 'DIRECT-REGULATOR', 
              'INDIRECT-UPREGULATOR', 'SUBSTRATE_PRODUCT-OF', 'NOT', 'ACTIVATOR', 
              'INDIRECT-DOWNREGULATOR', 'INHIBITOR', 'PRODUCT-OF', 'AGONIST-ACTIVATOR', 'PART-OF']
        elif 'cdr' in data_dir or 'aimed' in data_dir:
            class_names = ['0', '1']
        elif 'BioRED' in data_dir:
            class_names = ["None", "Association", "Positive_Correlation", "Negative_Correlation", "Cotreatment",
            "Drug_Interaction", "Bind", "Comparison", "Conversion"]
        else:
            raise ValueError("Missing class names for the current dataset")

        features = Features({'sentence': Value('string'), 'id': Value('string'), 'label': ClassLabel(names=class_names)})
        data = load_dataset('json', data_files=data_files, features=features, split=split)

        return data, class_names


class BioLinkBERTDataProcessor(DataProcessor):

    def __init__(self, max_seq_length, data_dir, all_in_one=False, binary=False):
        super(BioLinkBERTDataProcessor, self).__init__('michiyasunaga/BioLinkBERT-large', max_seq_length, data_dir, all_in_one, binary)


class PubMedBERTDataProcessor(DataProcessor):

    def __init__(self, max_seq_length, data_dir, all_in_one = False, binary=False):
        super(PubMedBERTDataProcessor, self).__init__("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", max_seq_length, data_dir, all_in_one, binary)