import json

from conversion import DataConverter, DataInstance, Document, Entity


class CDRInstance(DataInstance):

    def __init__(self, index, text, label, chemical, disease):
        super(CDRInstance, self).__init__(index, text, label)
        self.disease = disease
        self.chemical = chemical

    def __str__(self):
        return f"{self.index}\t{self.chemical}\t{self.disease}\t{self.text}\t{self.label}"


class CDRConverter(DataConverter):

    def __init__(self, tag_dict):
        super().__init__(tag_dict)

    def load_data(self, file):
        instances = []

        with open(file, 'r', encoding='utf8') as pub_reader:

            pmid = ''

            document = None
            chemicals = []
            diseases = []
            text = ''
            true_relation_pairs = []

            for line in pub_reader:
                line = line.rstrip()

                if line == '':
                    document = Document(pmid)

                    all_pairs = self.enumerate_all_cdr_pairs(chemicals, diseases)

                    document.text = text
                    document.relation_pairs = [(pair, '1' if pair in true_relation_pairs else '0') for pair in all_pairs]

                    instances.extend(self.extract_instances(document, chemicals, diseases))

                    chemicals = []
                    diseases = []
                    text = ''
                    true_relation_pairs = []
                    continue

                tks = line.split('|')

                if len(tks) > 1 and (tks[1] == 't' or tks[1] == 'a'):
                    # 2234245	250	270	audiovisual toxicity	Disease	D014786|D006311
                    pmid = tks[0]
                    # add space to separate title from abstract
                    if tks[1] == 'a':
                        text += ' '
                    text += tks[2]
                else:
                    _tks = line.split('\t')
                    if _tks[1] != 'CID':
                        start = int(_tks[1])
                        end = int(_tks[2])
                        mention = _tks[3]
                        ne_type = _tks[4]
                        # 2234245	250	270	audiovisual toxicity	Disease	D014786|D006311
                        ids = _tks[5]
                        entity = Entity(mention, set(ids.split('|')), start, end)

                        # merge the mentions of the same chemicals together
                        if ne_type == "Chemical":
                            try:
                                index = chemicals.index(entity)
                                chemicals[index].merge(entity)
                            except ValueError:
                                chemicals.append(entity)
                        else:
                            try:
                                index = diseases.index(entity)
                                diseases[index].merge(entity)
                            except ValueError:
                                diseases.append(entity)
                    else:
                        id1 = _tks[2]
                        id2 = _tks[3]
                        true_relation_pairs.append((id1, id2))

        return instances



    def extract_instances(self, document, chemicals, diseases):
        instances = []

        for pair, label in document.relation_pairs:
            chemicals_to_check = [chemical for chemical in chemicals if pair[0] in chemical.ids]
            diseases_to_check = [disease for disease in diseases if pair[1] in disease.ids]
            for chemical in chemicals_to_check:
                for disease in diseases_to_check:
                    new_text = ''
                    old_end = 0
                    all_starts = chemical.start + disease.start
                    all_starts = sorted(all_starts)
                    for start in all_starts:
                        if start in chemical.start:
                            tag = 'chemical'
                            end = chemical.end[chemical.start.index(start)]
                        else:
                            tag = 'disease'
                            end = disease.end[disease.start.index(start)]


                        new_text += document.text[old_end:start] + self.annotation_tag_dict[tag]
                        old_end = end

                    new_text += document.text[old_end:]

                    instance = CDRInstance(document.id, new_text, label, chemical.mention, disease.mention)
                    if instance not in instances:
                        instances.append(instance)
        
        self.clean_text(instances)

        return instances


    @staticmethod
    def enumerate_all_cdr_pairs(chemicals, diseases):
        all_cdr_pairs = set()

        all_chemical_ids = set()
        all_disease_ids = set()

        for chemical in chemicals:
            all_chemical_ids = (all_chemical_ids | chemical.ids)

        for disease in diseases:
            all_disease_ids = (all_disease_ids | disease.ids)

        for id1 in all_chemical_ids:
            for id2 in all_disease_ids:
                all_cdr_pairs.add((id1, id2))

        return all_cdr_pairs

    def write_instances(self, data_instances, file):
        lens = []
        with open(file, 'w', encoding='utf8') as outf:
            for instance in data_instances:
                out = {"id":instance.index, "sentence":instance.text, "label": instance.label}
                print(json.dumps(out, ensure_ascii=False), file=outf)
                lens.append(len(instance.text.split()))
        return lens