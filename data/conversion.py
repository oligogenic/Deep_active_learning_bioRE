import re

class DataInstance:

    def __init__(self, index, text, label):
        self.index = index
        self.text = text
        self.label = label
        self.tokens = []

    def __str__(self):
        return f"{self.index}\t{' '.join(self.tokens)}\t{' '.join(str(x) for x in self.in_neighbors)}\t{self.label}"

    def __eq__(self, other):
        return self.index == other.index and self.text == other.text and self.label == other.label
    
    def set_tokens(self, tokens):
        self.tokens = tokens

    def set_text(self, text):
        self.text = text

    def get_tokens(self):
        return self.tokens

class Entity:

    def __init__(self, mention, ids, start, end):
        self.mention = mention
        self.start = [start]
        self.end = [end]
        self.ids = ids

    def __eq__(self, other):
        if self.ids == {'-1'} or other.ids == {'-1'}:
            return False
        return self.ids == other.ids and self.mention == other.mention

    def to_json(self):
        return {
            'mention' : self.mention,
            'start' : self.start,
            'end' : self.end
        }

    def merge(self, other):
        self.start.extend(other.start)
        self.end.extend(other.end)


class DataConverter:

    def __init__(self, tag_dict):
        self.annotation_tag_dict = tag_dict

    def load_data(self, file):
        raise NotImplementedError('Will be implemented for each specific dataset')

    def clean_text(self, data_instances):
        
        for data_instance in data_instances:
            data_instance.text = data_instance.text.replace('$.', '$ .').replace('$,', '$ ,').replace('$,', '$ ,').replace('$:', '$ :').replace('$?', '$ ?').replace('$!', '$ !').replace('$;', '$ ;').replace('$/', '$ /').replace('$)','$ )').replace('$(','$ (').replace('$-','$ -').replace('$+','$ +').replace('$[','$ [').replace('$]', '$ ]')
            data_instance.text = data_instance.text.replace('-@', '- @').replace('+@', '+ @').replace('/@', '/ @').replace(':@', ': @').replace(';@', '; @').replace('(@', '( @').replace('.@', '. @').replace(')@', ') @').replace(',@', ', @').replace('!@', '! @').replace('?@', '? @').replace(']@', '] @').replace('[@', '[ @')

            # for chemicals with numbers for ions
            data_instance.text = re.sub('(\$)([0-9]+)','$ \\2',data_instance.text)
            data_instance.text = re.sub('([0-9]+)@', '\\1 @', data_instance.text)

    def write_instances(self, data_instances, file):
        raise NotImplementedError('Will be implemented for each specific dataset')

class Document:

    def __init__(self, id):
        self.id = id
        self.text = ''
        self.relation_pairs = []