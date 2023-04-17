import json

from conversion import DataConverter, DataInstance


class NaryInstance(DataInstance):

    def __init__(self, index, text, label):
        super(NaryInstance, self).__init__(index, text, label)
        self.drug = None
        self.variant = None
        self.gene = None

    def to_str(self, multi=False):
        if multi:
            return f"{self.index}\t{self.drug}\t{self.gene}\t{self.variant}\t{' '.join(self.tokens)}\t{self.label}"
        else:
            return f"{self.index}\t{self.drug}\t{self.variant}\t{' '.join(self.tokens)}\t{self.label}"

    def set_drug(self, instance):
        self.drug = instance

    def get_drug(self):
        return self.drug

    def set_variant(self, instance):
        self.variant = instance

    def get_variant(self):
        return self.variant

    def set_gene(self, instance):
        self.gene = instance

    def get_gene(self):
        return self.gene

    def belong_to_entity(self, mention):
        type_entity = None
        for entity in [self.gene,self.variant, self.drug]:
            if entity is not None:
                if mention in entity.text:
                    type_entity =  entity.type
        return type_entity

    def get_entity(self,type):
        if type == "gene":
            return self.gene
        elif type == "variant":
            return self.variant
        elif type == "drug":
            return self.drug


class NaryEntity:
    def __init__(self, mention, indices, type):
        self.text = mention.replace('\t','_')
        self.indices = indices
        self.type = type

    def get_indices(self):
        return self.indices

    def __str__(self):
        return self.text

class NaryConverter(DataConverter):

    def __init__(self, tag_dict):
        super(NaryConverter, self).__init__(tag_dict)


    def load_data(self, file):
        data_instances = []

        parsed_file_json_1 = json.loads(open(file[0], 'r', encoding='utf-8').read())
        parsed_file_json_2 = json.loads(open(file[1], 'r', encoding='utf-8').read())

        for i in range(len(parsed_file_json_1)):
            article = parsed_file_json_1[i]
            nary_instance = self.extract_instance(article)

            data_instances.append(nary_instance)

        for i in range(len(parsed_file_json_2)):
            article = parsed_file_json_2[i]
            nary_instance = self.extract_instance(article)
            data_instances.append(nary_instance)

        return data_instances

    def extract_instance(self, article):
        index = article['article']
        label = article['relationLabel'].strip().replace('  ', ' ')

        nary_instance = NaryInstance(index, '', label)
        for entity in article['entities']:
            nary_entity = NaryEntity(entity['mention'], entity['indices'], entity['type'])
            if entity['type'] == 'drug':
                nary_instance.set_drug(nary_entity)
            elif entity['type'] == 'variant':
                nary_instance.set_variant(nary_entity)
            elif entity['type'] == 'gene':
                nary_instance.set_gene(nary_entity)

        tokens = []

        nodes = []
        for sentence in article['sentences']:
            nodes.extend(sentence['nodes'])

        i = 0
        while i < len(nodes):
            token = nodes[i]
            mention = token['label'].replace('\t', '_')

            # this enables to avoid the indices which are actually not included in the mention
            entity_type = nary_instance.belong_to_entity(mention)

            if entity_type is not None:
                entity = nary_instance.get_entity(entity_type)

                # to avoid when sometimes the mention is present at different places in the text
                if i in entity.indices:
                    mention = self.annotation_tag_dict[entity_type]
                    if len(entity.text.split(' ')) > 1:
                        j = 0
                        # advance in the tokens while the token is part of the entity
                        while j < len(entity.indices) and nodes[entity.indices[j]]['label'].replace('\t', '_') in entity.text :
                            i += 1
                            j += 1

            tokens.append(mention)

            i += 1

        nary_instance.set_tokens(tokens)
        nary_instance.set_text(' '.join(nary_instance.get_tokens()))

        return nary_instance


if __name__ == "__main__":

    converter = NaryConverter()

    # test with error in indices for mention, more than should be
    article = {"article":"17877814","sentences":[{"paragraph":25,"paragraphSentence":4,"sentence":139,"root":3,"nodes":[{"index":0,"indexInsideSentence":0,"label":"The","lemma":"the","postag":"DT","arcs":[{"toIndex":1,"label":"adjtok:next"},{"toIndex":2,"label":"depinv:det"}]},{"index":1,"indexInsideSentence":1,"label":"T790M","lemma":"t790m","postag":"NN","arcs":[{"toIndex":0,"label":"adjtok:prev"},{"toIndex":2,"label":"adjtok:next"},{"toIndex":2,"label":"depinv:nn"}]},{"index":2,"indexInsideSentence":2,"label":"mutation","lemma":"mutation","postag":"NN","arcs":[{"toIndex":1,"label":"adjtok:prev"},{"toIndex":3,"label":"adjtok:next"},{"toIndex":0,"label":"deparc:det"},{"toIndex":1,"label":"deparc:nn"},{"toIndex":3,"label":"depinv:nsubj"}]},{"index":3,"indexInsideSentence":3,"label":"abrogated","lemma":"abrogate","postag":"VBD","arcs":[{"toIndex":2,"label":"adjtok:prev"},{"toIndex":4,"label":"adjtok:next"},{"toIndex":2,"label":"deparc:nsubj"},{"toIndex":5,"label":"deparc:dobj"},{"toIndex":20,"label":"deparc:advmod"},{"toIndex":23,"label":"deparc:prep_in"},{"toIndex":-1,"label":"root:root"}]},{"index":4,"indexInsideSentence":4,"label":"the","lemma":"the","postag":"DT","arcs":[{"toIndex":3,"label":"adjtok:prev"},{"toIndex":5,"label":"adjtok:next"},{"toIndex":5,"label":"depinv:det"}]},{"index":5,"indexInsideSentence":5,"label":"effect","lemma":"effect","postag":"NN","arcs":[{"toIndex":4,"label":"adjtok:prev"},{"toIndex":6,"label":"adjtok:next"},{"toIndex":4,"label":"deparc:det"},{"toIndex":3,"label":"depinv:dobj"},{"toIndex":7,"label":"deparc:prep_of"},{"toIndex":9,"label":"deparc:prep_on"},{"toIndex":19,"label":"deparc:conj_and"}]},{"index":6,"indexInsideSentence":6,"label":"of","lemma":"of","postag":"IN","arcs":[{"toIndex":5,"label":"adjtok:prev"},{"toIndex":7,"label":"adjtok:next"}]},{"index":7,"indexInsideSentence":7,"label":"erlotinib","lemma":"erlotinib","postag":"NN","arcs":[{"toIndex":6,"label":"adjtok:prev"},{"toIndex":8,"label":"adjtok:next"},{"toIndex":5,"label":"depinv:prep_of"}]},{"index":8,"indexInsideSentence":8,"label":"on","lemma":"on","postag":"IN","arcs":[{"toIndex":7,"label":"adjtok:prev"},{"toIndex":9,"label":"adjtok:next"}]},{"index":9,"indexInsideSentence":9,"label":"L858R","lemma":"l858r","postag":"NN","arcs":[{"toIndex":8,"label":"adjtok:prev"},{"toIndex":10,"label":"adjtok:next"},{"toIndex":5,"label":"depinv:prep_on"}]},{"index":10,"indexInsideSentence":10,"label":",","lemma":",","postag":",","arcs":[{"toIndex":9,"label":"adjtok:prev"},{"toIndex":11,"label":"adjtok:next"}]},{"index":11,"indexInsideSentence":11,"label":"and","lemma":"and","postag":"CC","arcs":[{"toIndex":10,"label":"adjtok:prev"},{"toIndex":12,"label":"adjtok:next"}]},{"index":12,"indexInsideSentence":12,"label":"the","lemma":"the","postag":"DT","arcs":[{"toIndex":11,"label":"adjtok:prev"},{"toIndex":13,"label":"adjtok:next"},{"toIndex":19,"label":"depinv:det"}]},{"index":13,"indexInsideSentence":13,"label":"L858R/T790M","lemma":"l858r/t790m","postag":"NN","arcs":[{"toIndex":12,"label":"adjtok:prev"},{"toIndex":14,"label":"adjtok:next"},{"toIndex":19,"label":"depinv:nn"}]},{"index":14,"indexInsideSentence":14,"label":"double","lemma":"double","postag":"JJ","arcs":[{"toIndex":13,"label":"adjtok:prev"},{"toIndex":15,"label":"adjtok:next"},{"toIndex":19,"label":"depinv:amod"}]},{"index":15,"indexInsideSentence":15,"label":"mutant","lemma":"mutant","postag":"JJ","arcs":[{"toIndex":14,"label":"adjtok:prev"},{"toIndex":16,"label":"adjtok:next"},{"toIndex":19,"label":"depinv:amod"}]},{"index":16,"indexInsideSentence":16,"label":"readily","lemma":"readily","postag":"RB","arcs":[{"toIndex":15,"label":"adjtok:prev"},{"toIndex":17,"label":"adjtok:next"},{"toIndex":17,"label":"depinv:advmod"}]},{"index":17,"indexInsideSentence":17,"label":"induced","lemma":"induced","postag":"JJ","arcs":[{"toIndex":16,"label":"adjtok:prev"},{"toIndex":18,"label":"adjtok:next"},{"toIndex":16,"label":"deparc:advmod"},{"toIndex":18,"label":"depinv:amod"}]},{"index":18,"indexInsideSentence":18,"label":"Akt","lemma":"akt","postag":"NN","arcs":[{"toIndex":17,"label":"adjtok:prev"},{"toIndex":19,"label":"adjtok:next"},{"toIndex":17,"label":"deparc:amod"},{"toIndex":19,"label":"depinv:nn"}]},{"index":19,"indexInsideSentence":19,"label":"phosphorylation","lemma":"phosphorylation","postag":"NN","arcs":[{"toIndex":18,"label":"adjtok:prev"},{"toIndex":20,"label":"adjtok:next"},{"toIndex":12,"label":"deparc:det"},{"toIndex":13,"label":"deparc:nn"},{"toIndex":14,"label":"deparc:amod"},{"toIndex":15,"label":"deparc:amod"},{"toIndex":18,"label":"deparc:nn"},{"toIndex":5,"label":"depinv:conj_and"}]},{"index":20,"indexInsideSentence":20,"label":"even","lemma":"even","postag":"RB","arcs":[{"toIndex":19,"label":"adjtok:prev"},{"toIndex":21,"label":"adjtok:next"},{"toIndex":3,"label":"depinv:advmod"}]},{"index":21,"indexInsideSentence":21,"label":"in","lemma":"in","postag":"IN","arcs":[{"toIndex":20,"label":"adjtok:prev"},{"toIndex":22,"label":"adjtok:next"}]},{"index":22,"indexInsideSentence":22,"label":"the","lemma":"the","postag":"DT","arcs":[{"toIndex":21,"label":"adjtok:prev"},{"toIndex":23,"label":"adjtok:next"},{"toIndex":23,"label":"depinv:det"}]},{"index":23,"indexInsideSentence":23,"label":"presence","lemma":"presence","postag":"NN","arcs":[{"toIndex":22,"label":"adjtok:prev"},{"toIndex":24,"label":"adjtok:next"},{"toIndex":22,"label":"deparc:det"},{"toIndex":3,"label":"depinv:prep_in"},{"toIndex":27,"label":"deparc:prep_of"}]},{"index":24,"indexInsideSentence":24,"label":"of","lemma":"of","postag":"IN","arcs":[{"toIndex":23,"label":"adjtok:prev"},{"toIndex":25,"label":"adjtok:next"}]},{"index":25,"indexInsideSentence":25,"label":"10","lemma":"10","postag":"CD","arcs":[{"toIndex":24,"label":"adjtok:prev"},{"toIndex":26,"label":"adjtok:next"},{"toIndex":26,"label":"depinv:number"}]},{"index":26,"indexInsideSentence":26,"label":"μM","lemma":"μm","postag":"NN","arcs":[{"toIndex":25,"label":"adjtok:prev"},{"toIndex":27,"label":"adjtok:next"},{"toIndex":25,"label":"deparc:number"},{"toIndex":27,"label":"depinv:amod"}]},{"index":27,"indexInsideSentence":27,"label":"erlotinib","lemma":"erlotinib","postag":"NN","arcs":[{"toIndex":26,"label":"adjtok:prev"},{"toIndex":28,"label":"adjtok:next"},{"toIndex":26,"label":"deparc:amod"},{"toIndex":23,"label":"depinv:prep_of"}]},{"index":28,"indexInsideSentence":28,"label":".","lemma":".","postag":".","arcs":[{"toIndex":27,"label":"adjtok:prev"}]}]}],"entities":[{"paragraph":25,"paragraphSentence":4,"sentence":139,"type":"drug","id":"erlotinib","mention":"erlotinib","indices":[26,27]},{"paragraph":25,"paragraphSentence":4,"sentence":139,"type":"variant","id":"T790M","mention":"T790M","indices":[1]}],"relationLabel":"resistance or non-response"}
    annotation = converter.extract_info(article)
    print(annotation.text)

    # test with mention comprising several indices
    article = {"article":"22233865","sentences":[{"paragraph":46,"paragraphSentence":9,"sentence":137,"root":17,"nodes":[{"index":0,"indexInsideSentence":0,"label":"ELISA","lemma":"elisa","postag":"NN","arcs":[{"toIndex":1,"label":"adjtok:next"},{"toIndex":17,"label":"depinv:dep"},{"toIndex":5,"label":"deparc:dep"}]},{"index":1,"indexInsideSentence":1,"label":",","lemma":",","postag":",","arcs":[{"toIndex":0,"label":"adjtok:prev"},{"toIndex":2,"label":"adjtok:next"}]},{"index":2,"indexInsideSentence":2,"label":"enzyme","lemma":"enzyme","postag":"JJ","arcs":[{"toIndex":1,"label":"adjtok:prev"},{"toIndex":3,"label":"adjtok:next"},{"toIndex":3,"label":"depinv:hyphen"}]},{"index":3,"indexInsideSentence":3,"label":"linked","lemma":"link","postag":"VBD","arcs":[{"toIndex":2,"label":"adjtok:prev"},{"toIndex":4,"label":"adjtok:next"},{"toIndex":5,"label":"depinv:amod"},{"toIndex":2,"label":"deparc:hyphen"}]},{"index":4,"indexInsideSentence":4,"label":"immunosorbent","lemma":"immunosorbent","postag":"JJ","arcs":[{"toIndex":3,"label":"adjtok:prev"},{"toIndex":5,"label":"adjtok:next"},{"toIndex":5,"label":"depinv:amod"}]},{"index":5,"indexInsideSentence":5,"label":"assay","lemma":"assay","postag":"NN","arcs":[{"toIndex":4,"label":"adjtok:prev"},{"toIndex":6,"label":"adjtok:next"},{"toIndex":3,"label":"deparc:amod"},{"toIndex":4,"label":"deparc:amod"},{"toIndex":0,"label":"depinv:dep"},{"toIndex":7,"label":"deparc:dep"}]},{"index":6,"indexInsideSentence":6,"label":";","lemma":";","postag":":","arcs":[{"toIndex":5,"label":"adjtok:prev"},{"toIndex":7,"label":"adjtok:next"}]},{"index":7,"indexInsideSentence":7,"label":"ICOS","lemma":"icos","postag":"NN","arcs":[{"toIndex":6,"label":"adjtok:prev"},{"toIndex":8,"label":"adjtok:next"},{"toIndex":5,"label":"depinv:dep"},{"toIndex":10,"label":"deparc:appos"},{"toIndex":12,"label":"deparc:dep"},{"toIndex":15,"label":"deparc:appos"}]},{"index":8,"indexInsideSentence":8,"label":",","lemma":",","postag":",","arcs":[{"toIndex":7,"label":"adjtok:prev"},{"toIndex":9,"label":"adjtok:next"}]},{"index":9,"indexInsideSentence":9,"label":"inducible","lemma":"inducible","postag":"JJ","arcs":[{"toIndex":8,"label":"adjtok:prev"},{"toIndex":10,"label":"adjtok:next"},{"toIndex":10,"label":"depinv:amod"}]},{"index":10,"indexInsideSentence":10,"label":"costimulator","lemma":"costimulator","postag":"NN","arcs":[{"toIndex":9,"label":"adjtok:prev"},{"toIndex":11,"label":"adjtok:next"},{"toIndex":9,"label":"deparc:amod"},{"toIndex":7,"label":"depinv:appos"}]},{"index":11,"indexInsideSentence":11,"label":";","lemma":";","postag":":","arcs":[{"toIndex":10,"label":"adjtok:prev"},{"toIndex":12,"label":"adjtok:next"}]},{"index":12,"indexInsideSentence":12,"label":"IgG","lemma":"igg","postag":"NN","arcs":[{"toIndex":11,"label":"adjtok:prev"},{"toIndex":13,"label":"adjtok:next"},{"toIndex":7,"label":"depinv:dep"}]},{"index":13,"indexInsideSentence":13,"label":",","lemma":",","postag":",","arcs":[{"toIndex":12,"label":"adjtok:prev"},{"toIndex":14,"label":"adjtok:next"}]},{"index":14,"indexInsideSentence":14,"label":"immunoglobulin","lemma":"immunoglobulin","postag":"NN","arcs":[{"toIndex":13,"label":"adjtok:prev"},{"toIndex":15,"label":"adjtok:next"},{"toIndex":15,"label":"depinv:nn"}]},{"index":15,"indexInsideSentence":15,"label":"G","lemma":"g","postag":"NN","arcs":[{"toIndex":14,"label":"adjtok:prev"},{"toIndex":16,"label":"adjtok:next"},{"toIndex":14,"label":"deparc:nn"},{"toIndex":7,"label":"depinv:appos"}]},{"index":16,"indexInsideSentence":16,"label":";","lemma":";","postag":":","arcs":[{"toIndex":15,"label":"adjtok:prev"},{"toIndex":17,"label":"adjtok:next"}]},{"index":17,"indexInsideSentence":17,"label":"IL10","lemma":"il10","postag":"NN","arcs":[{"toIndex":16,"label":"adjtok:prev"},{"toIndex":18,"label":"adjtok:next"},{"toIndex":0,"label":"deparc:dep"},{"toIndex":20,"label":"deparc:appos"},{"toIndex":22,"label":"deparc:dep"},{"toIndex":26,"label":"deparc:appos"},{"toIndex":32,"label":"deparc:dep"},{"toIndex":-1,"label":"root:root"}]},{"index":18,"indexInsideSentence":18,"label":",","lemma":",","postag":",","arcs":[{"toIndex":17,"label":"adjtok:prev"},{"toIndex":19,"label":"adjtok:next"}]},{"index":19,"indexInsideSentence":19,"label":"interleukin","lemma":"interleukin","postag":"NN","arcs":[{"toIndex":18,"label":"adjtok:prev"},{"toIndex":20,"label":"adjtok:next"},{"toIndex":20,"label":"depinv:nn"}]},{"index":20,"indexInsideSentence":20,"label":"10","lemma":"10","postag":"NN","arcs":[{"toIndex":19,"label":"adjtok:prev"},{"toIndex":21,"label":"adjtok:next"},{"toIndex":17,"label":"depinv:appos"},{"toIndex":19,"label":"deparc:nn"}]},{"index":21,"indexInsideSentence":21,"label":";","lemma":";","postag":":","arcs":[{"toIndex":20,"label":"adjtok:prev"},{"toIndex":22,"label":"adjtok:next"}]},{"index":22,"indexInsideSentence":22,"label":"T1D","lemma":"t1d","postag":"NN","arcs":[{"toIndex":21,"label":"adjtok:prev"},{"toIndex":23,"label":"adjtok:next"},{"toIndex":17,"label":"depinv:dep"}]},{"index":23,"indexInsideSentence":23,"label":",","lemma":",","postag":",","arcs":[{"toIndex":22,"label":"adjtok:prev"},{"toIndex":24,"label":"adjtok:next"}]},{"index":24,"indexInsideSentence":24,"label":"type","lemma":"type","postag":"NN","arcs":[{"toIndex":23,"label":"adjtok:prev"},{"toIndex":25,"label":"adjtok:next"},{"toIndex":26,"label":"depinv:nn"}]},{"index":25,"indexInsideSentence":25,"label":"1","lemma":"1","postag":"CD","arcs":[{"toIndex":24,"label":"adjtok:prev"},{"toIndex":26,"label":"adjtok:next"},{"toIndex":26,"label":"depinv:num"}]},{"index":26,"indexInsideSentence":26,"label":"diabetes","lemma":"diabetes","postag":"NNS","arcs":[{"toIndex":25,"label":"adjtok:prev"},{"toIndex":27,"label":"adjtok:next"},{"toIndex":24,"label":"deparc:nn"},{"toIndex":25,"label":"deparc:num"},{"toIndex":17,"label":"depinv:appos"}]},{"index":27,"indexInsideSentence":27,"label":";","lemma":";","postag":":","arcs":[{"toIndex":26,"label":"adjtok:prev"},{"toIndex":28,"label":"adjtok:next"}]},{"index":28,"indexInsideSentence":28,"label":"Tregs","lemma":"tregs","postag":"RB","arcs":[{"toIndex":27,"label":"adjtok:prev"},{"toIndex":29,"label":"adjtok:next"},{"toIndex":32,"label":"depinv:advmod"}]},{"index":29,"indexInsideSentence":29,"label":",","lemma":",","postag":",","arcs":[{"toIndex":28,"label":"adjtok:prev"},{"toIndex":30,"label":"adjtok:next"}]},{"index":30,"indexInsideSentence":30,"label":"regulatory","lemma":"regulatory","postag":"JJ","arcs":[{"toIndex":29,"label":"adjtok:prev"},{"toIndex":31,"label":"adjtok:next"},{"toIndex":32,"label":"depinv:amod"}]},{"index":31,"indexInsideSentence":31,"label":"T","lemma":"t","postag":"NN","arcs":[{"toIndex":30,"label":"adjtok:prev"},{"toIndex":32,"label":"adjtok:next"},{"toIndex":32,"label":"depinv:nn"}]},{"index":32,"indexInsideSentence":32,"label":"cells","lemma":"cell","postag":"NNS","arcs":[{"toIndex":31,"label":"adjtok:prev"},{"toIndex":33,"label":"adjtok:next"},{"toIndex":28,"label":"deparc:advmod"},{"toIndex":30,"label":"deparc:amod"},{"toIndex":31,"label":"deparc:nn"},{"toIndex":17,"label":"depinv:dep"}]},{"index":33,"indexInsideSentence":33,"label":".","lemma":".","postag":".","arcs":[{"toIndex":32,"label":"adjtok:prev"}]}]}],"entities":[{"paragraph":46,"paragraphSentence":9,"sentence":137,"type":"drug","id":"intravenous immunoglobulin","mention":"immunoglobulin G","indices":[14,15]},{"paragraph":46,"paragraphSentence":9,"sentence":137,"type":"variant","id":"T1D","mention":"T1D","indices":[22]}],"relationLabel":"None"}
    annotation = converter.extract_info(article)
    print(annotation.text)
