import subprocess
import numpy as np
from cdr import CDRConverter

def download_cdr():
    subprocess.call(['curl', '-k', '-o', 'CDR_Data.zip',
                     'https://biocreative.bioinformatics.udel.edu/media/store/files/2016/CDR_Data.zip'])
    subprocess.call(['mkdir', '-p', 'cdr'])
    subprocess.call(['mkdir', '-p', 'cdr/processed'])
    subprocess.call(['unzip', 'CDR_Data.zip', '-d', 'cdr'])
    subprocess.call(['rm', 'CDR_Data.zip'])


def convert_cdr():
    IN_DIRECTORY_CDR = 'cdr/CDR_Data/CDR.Corpus.v010516/'
    OUT_DIRECTORY_CDR = 'cdr/processed/'
    files = {
        'dev': 'CDR_DevelopmentSet.PubTator.txt',
        'test': 'CDR_TestSet.PubTator.txt',
        'train': 'CDR_TrainingSet.PubTator.txt'
    }
    tag_dict = {
        'chemical': '@CHEMICAL$',
        'disease' : '@DISEASE$'
    }
    converter = CDRConverter(tag_dict)
    
    lens = []

    for type in files:
        instances = converter.load_data(f"{IN_DIRECTORY_CDR}{files[type]}")
        stats = converter.write_instances(instances, f"{OUT_DIRECTORY_CDR}{type}.json")
        lens.extend(stats)

    print("total", len(lens), "seqlen mean", int(np.mean(lens)), "median", int(np.median(lens)), "95th",
                  int(np.percentile(lens, 95)), "max", np.max(lens))


if __name__ == "__main__":
    download_cdr()
    convert_cdr()