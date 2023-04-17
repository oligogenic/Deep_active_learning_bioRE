import json
from conversion import DataConverter, DataInstance, Document

class DrugProtEntity:

    def __init__(self, mention, id, start, end):
        self.mention = mention
        self.start = start
        self.end = end
        self.id = id


class DrugProtInstance(DataInstance):

    def __init__(self, index, text, label, chemical, gene):
        super(DrugProtInstance, self).__init__(index, text, label)
        self.gene = gene
        self.chemical = chemical

    def __str__(self):
        return f"{self.index}\t{self.chemical}\t{self.gene}\t{self.text}\t{self.label}"


class DrugProtConverter(DataConverter):

    def __init__(self, tag_dict):
        super().__init__(tag_dict)

    def load_data(self, file):
        """
        Inspired fom https://github.com/luckynozomi/ChemProt-BioCreative/blob/master/raw_data_processing/make_dataset.py
        """
        pmid_to_abstract_dict = {}
        with open(file[0], "r", encoding="utf-8") as abstract_file:
            for line in abstract_file:
                pmid, text_title, text_abstract = line.strip().split('\t')
                pmid_to_abstract_dict[pmid] = text_title + '\t' + text_abstract

        chemicals_entities_dict = {}
        genes_entities_dict = {}
        with open(file[1], "r", encoding="utf-8") as entity_file:
            for line in entity_file:
                pmid, label, entity_type, start_pos, end_pos, name = line.strip().split('\t')
                if pmid not in chemicals_entities_dict:
                    chemicals_entities_dict[pmid] = []
                if pmid not in genes_entities_dict:
                    genes_entities_dict[pmid] = []

                entity = DrugProtEntity(name, label, int(start_pos), int(end_pos))
                
                if entity_type == "CHEMICAL":
                    chemicals_entities_dict[pmid].append(entity)
                else:
                    genes_entities_dict[pmid].append(entity)

        pmid_to_relations_dict = {}
        relations_to_label_dict = {}
        with open(file[2], "r", encoding="utf-8") as relation_file:
            for line in relation_file:
                pmid, relation_type, arg1, arg2 = line.strip().split('\t')
                arg1 = arg1.replace('Arg1', '')
                arg2 = arg2.replace('Arg2','')
                if pmid in pmid_to_relations_dict:
                    pmid_to_relations_dict[pmid].append((arg1, arg2))
                else:
                    pmid_to_relations_dict[pmid] = [(arg1, arg2)]
                
                if pmid in relations_to_label_dict:
                    relations_to_label_dict[pmid][(arg1,arg2)] = relation_type
                else:
                    relations_to_label_dict[pmid] = {(arg1,arg2) : relation_type}


        instances = []                    
        for pmid in pmid_to_abstract_dict:
            chemicals = chemicals_entities_dict[pmid]
            genes = genes_entities_dict[pmid]

            # if no relations present in the text
            if pmid not in pmid_to_relations_dict:
                continue

            true_relation_pairs = pmid_to_relations_dict[pmid]
            label_true_relation = relations_to_label_dict[pmid]

            document = Document(pmid)
            all_pairs = self.enumerate_all_pairs(chemicals, genes)

            document.text = pmid_to_abstract_dict[pmid]
            relation_pairs = []

            for arg1,arg2 in all_pairs:
                if (arg1,arg2) in true_relation_pairs:
                    relation_pairs.append(((arg1,arg2), label_true_relation[(arg1,arg2)]))
                elif (arg2,arg1) in true_relation_pairs:
                    relation_pairs.append(((arg2,arg1), label_true_relation[(arg2,arg1)]))
                else:
                    relation_pairs.append(((arg1,arg2), "0"))
            
            document.relation_pairs = relation_pairs
            instances.extend(self.extract_instances(document,chemicals,genes))
        
        return instances

    @staticmethod
    def enumerate_all_pairs(chemicals, genes):
        all_pairs = set()

        all_chemical_ids = []
        all_genes_ids = []

        for chemical in chemicals:
            all_chemical_ids.append(chemical.id)

        for gene in genes:
            all_genes_ids.append(gene.id)

        for id1 in all_chemical_ids:
            for id2 in all_genes_ids:
                all_pairs.add((id1, id2))

        return all_pairs
    
    def extract_instances(self, document, chemicals, genes):
        instances = []

        for pair, label in document.relation_pairs:
            chemical_to_check = [chemical for chemical in chemicals if pair[0] == chemical.id or pair[1] == chemical.id][0]
            gene_to_check = [gene for gene in genes if pair[0] == gene.id or pair[1] == gene.id][0]
            
            new_text = ''
            old_end = 0
            all_starts = [chemical_to_check.start] + [gene_to_check.start]
            all_starts = sorted(all_starts)
            for start in all_starts:
                if start == chemical_to_check.start:
                    tag = 'chemical'
                    end = chemical_to_check.end
                else:
                    tag = 'gene'
                    end = gene_to_check.end
                
                new_text += document.text[old_end:start] + self.annotation_tag_dict[tag]
                old_end = end

            new_text += document.text[old_end:]

            instance = DrugProtInstance(document.id, new_text, label, chemical_to_check.mention, gene_to_check.mention)
            if instance not in instances and '@GENE$' in instance.text and '@CHEMICAL$' in instance.text:
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