from itertools import product
import json

from conversion import DataConverter, Document, Entity, DataInstance


class BioREDEntity(Entity):

    def __init__(self, mention, ids, start, end):
        super().__init__(mention, ids, start, end)

    def __eq__(self, other):
        if self.ids == {'-'} or other.ids == {'-'}:
            return False
        return len(self.ids & other.ids) != 0
    
    def merge(self, other):
        self.start.extend(other.start)
        self.end.extend(other.end)


class BioREDInstance(DataInstance):

    def __init__(self, index, text, label, entity_one, entity_two):
        super().__init__(index, text, label)
        self.entity_one = entity_one
        self.entity_two = entity_two


class BioREDDocument(Document):

    def __init__(self, id):
        super().__init__(id)
        self.genes = []
        self.chemicals = []
        self.diseases = []
        self.variants = []
    
    def to_json(self):

        return {
            'text' : self.text,
            'relations' : {'_'.join(pair) : label for pair, label in self.relation_pairs},
            'entities' : {
                'gene' : {entity_id: entity.to_json() for entity in self.genes for entity_id in entity.ids},
                'chemical' : {entity_id: entity.to_json() for entity in self.chemicals for entity_id in entity.ids},
                'disease' : {entity_id: entity.to_json() for entity in self.diseases for entity_id in entity.ids},
                'variant' : {entity_id: entity.to_json() for entity in self.variants for entity_id in entity.ids}
            }
        }


class BioREDConverter(DataConverter):

    def __init__(self, tag_dict):
        super().__init__(tag_dict)
        
    @staticmethod
    def produce_documents(file):
        """
        Convert the pubtator into a json with an entry per document
        """
        document_dict = dict()
        with open(file, 'r', encoding='utf-8') as infile:
            pmid = ''

            document = None
            chemicals = []
            diseases = []
            genes = []
            variants = []
            text = ''
            relation_pairs = []

            for line in infile:
                line = line.rstrip()

                if line == '':
                    document = BioREDDocument(pmid)

                    document.text = text
                    document.relation_pairs = relation_pairs
                    document.genes = genes
                    document.variants = variants
                    document.diseases = diseases
                    document.chemicals = chemicals

                    document_dict[document.id] = document.to_json()

                    chemicals = []
                    diseases = []
                    genes = []
                    variants = []
                    text = ''
                    relation_pairs = []
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
                    if _tks[1].isdigit():
                        start = int(_tks[1])
                        end = int(_tks[2])
                        mention = _tks[3]
                        ne_type = _tks[4]
                        # 2234245	250	270	audiovisual toxicity	Disease	D014786|D006311
                        ids = _tks[5]
                        entity = BioREDEntity(mention, set(ids.split(',')), start, end)

                        # merge the mentions of the same chemicals together
                        if "Chemical" in ne_type:
                            try:
                                index = chemicals.index(entity)
                                chemicals[index].merge(entity)
                            except ValueError:
                                chemicals.append(entity)
                        elif "Disease" in ne_type:
                            try:
                                index = diseases.index(entity)
                                diseases[index].merge(entity)
                            except ValueError:
                                diseases.append(entity)
                        elif "Gene" in ne_type:
                            try:
                                index = genes.index(entity)
                                genes[index].merge(entity)
                            except ValueError:
                                genes.append(entity)
                        elif "Variant" in ne_type:
                            try:
                                index = variants.index(entity)
                                variants[index].merge(entity)
                            except ValueError:
                                variants.append(entity)
                        elif 'Organism' in ne_type or 'Cell' in ne_type:
                            continue
                        else:
                            print(ne_type)
                            raise TypeError("Missing the entity type")      
                    else:
                        relation_type = _tks[1]
                        id1 = _tks[2]
                        id2 = _tks[3]
                        relation_pairs.append(([id1, id2], relation_type))
        return document_dict

    def load_data(self, file):
        pass

    def extract_instances(self, pmid, relations, text, entities_one, entities_two, annotation_tags):
        instances = []

        all_possible_pairs = list(product(list(entities_one.keys()), list(entities_two.keys())))

        for pair in all_possible_pairs:
            pair = list(pair)

            # ignore the same entities in a pair
            if pair[0] == pair[1]:
                continue

            if '_'.join(pair) in relations:
                label = relations['_'.join(pair)]
            elif '_'.join([pair[1], pair[0]]) in relations:
                label = relations['_'.join([pair[1], pair[0]])]
            else:
                label = 'None'
            

            if pair[0] in entities_one:
                entity_one = entities_one[pair[0]]
                entity_two = entities_two[pair[1]]
            
            elif pair[0] in entities_two:
                entity_two = entities_two[pair[0]]
                entity_one = entities_one[pair[1]]

            new_text = ''
            old_end = 0
            all_starts = entity_one['start'] + entity_two['start']
            all_starts = sorted(all_starts)

            # we mask all the mention of the same entity
            for start in all_starts:
                if start in entity_one['start']:
                    tag = annotation_tags[0]
                    end = entity_one['end'][entity_one['start'].index(start)]
                else:
                    tag = annotation_tags[1]
                    end = entity_two['end'][entity_two['start'].index(start)]


                new_text += text[old_end:start] + tag
                old_end = end

            new_text += text[old_end:]

            instance = BioREDInstance(pmid, new_text, label, entity_one['mention'], entity_two['mention'])
            if instance not in instances:
                instances.append(instance)
        
        self.clean_text(instances)

        return instances


    def write_instances(self, data_instances, file):
        lens = []
        with open(file, 'w', encoding='utf8') as outf:
            for instance in data_instances:
                out = {"id":instance.index, "sentence":instance.text, "label": instance.label}
                print(json.dumps(out, ensure_ascii=False), file=outf)
                lens.append(len(instance.text.split()))
        return lens
