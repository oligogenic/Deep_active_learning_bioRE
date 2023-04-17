"""
@article{10.1145/3458754,
author = {Gu, Yu and Tinn, Robert and Cheng, Hao and Lucas, Michael and Usuyama, Naoto and Liu, Xiaodong and Naumann, Tristan and Gao, Jianfeng and Poon, Hoifung},
title = {Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing},
year = {2021},
issue_date = {January 2022},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {3},
number = {1},
issn = {2691-1957},
url = {https://doi.org/10.1145/3458754},
doi = {10.1145/3458754},
abstract = {Pretraining large neural language models, such as BERT, has led to impressive gains on many natural language processing (NLP) tasks. However, most pretraining efforts focus on general domain corpora, such as newswire and Web. A prevailing assumption is that even domain-specific pretraining can benefit by starting from general-domain language models. In this article, we challenge this assumption by showing that for domains with abundant unlabeled text, such as biomedicine, pretraining language models from scratch results in substantial gains over continual pretraining of general-domain language models. To facilitate this investigation, we compile a comprehensive biomedical NLP benchmark from publicly available datasets. Our experiments show that domain-specific pretraining serves as a solid foundation for a wide range of biomedical NLP tasks, leading to new state-of-the-art results across the board. Further, in conducting a thorough evaluation of modeling choices, both for pretraining and task-specific fine-tuning, we discover that some common practices are unnecessary with BERT models, such as using complex tagging schemes in named entity recognition. To help accelerate research in biomedical NLP, we have released our state-of-the-art pretrained and task-specific models for the community, and created a leaderboard featuring our BLURB benchmark (short for Biomedical Language Understanding & Reasoning Benchmark) at .},
journal = {ACM Trans. Comput. Healthcare},
month = {oct},
articleno = {2},
numpages = {23},
keywords = {domain-specific pretraining, NLP, Biomedical}
}


@inproceedings{yasunaga-etal-2022-linkbert,
    title = "{L}ink{BERT}: Pretraining Language Models with Document Links",
    author = "Yasunaga, Michihiro  and
      Leskovec, Jure  and
      Liang, Percy",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.551",
    doi = "10.18653/v1/2022.acl-long.551",
    pages = "8003--8016"
}
"""
import os.path

import torch

from transformers import AutoModelForSequenceClassification

class ModelFactory:

    def __init__(self, model_name, labels, save_dir):
        if model_name == 'pubmedbert':
            self.origin = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        elif model_name == 'biolinkbert':
            self.origin = 'michiyasunaga/BioLinkBERT-large'
        else:
            raise ValueError("Wrong model name, you have to choose either PubmedBERT or BioLinkBERT")

        self.save_dir = save_dir
        self.embed_dim = 0

        self.label2id, self.id2label = self.get_labels_config(labels)

    def get_labels_config(self,labels):
        label2id = {v: i for i, v in enumerate(labels)}
        id2label = {id: label for label, id in label2id.items()}
        return label2id,id2label

    def get_num_classes(self):
        return len(self.label2id)

    def get_embed_dim(self):
        return self.embed_dim

    def produce(self, model=None, cv = None):
        if model is None:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.origin,
                num_labels=self.get_num_classes(),
                output_hidden_states = True
            )
            self.embed_dim = model.config.hidden_size
            model.gradient_checkpointing_enable()
        else:
            model = model.module

        if not os.path.exists(f"{self.save_dir}/pytorch_model.bin"):
            torch.save(model.state_dict(), f"{self.save_dir}/pytorch_model.bin")

            return model

        if cv is None:
            model.load_state_dict(torch.load(f"{self.save_dir}/pytorch_model.bin", map_location="cpu"))
        else :
            model.load_state_dict(torch.load(f"{self.save_dir}/pytorch_model_{cv}.bin", map_location="cpu"))
        return model

    def save_model(self, model, cv):
        distributed = getattr(model, 'module', None)
        model_to_save = model
        if distributed:
            model_to_save = model.module
        torch.save(model_to_save.state_dict(),f"{self.save_dir}/pytorch_model_{cv}.bin")