import subprocess, random
import numpy as np

from drugprot import DrugProtConverter

def import_drugprot():
    subprocess.call(['mkdir', '-p', 'drugprot-gs-training-development/processed'])

def convert_drugprot():
    IN_DIRECTORY = 'drugprot-gs-training-development/'
    OUT_DIRECTORY = 'drugprot-gs-training-development/processed/'
    tag_dict = {
        'chemical': '@CHEMICAL$',
        'gene' : '@GENE$'
    }

    converter = DrugProtConverter(tag_dict)

    for folder in ['training', 'development']:
        files = [
            f"{IN_DIRECTORY}{folder}/drugprot_{folder}_abstracs.tsv",
            f"{IN_DIRECTORY}{folder}/drugprot_{folder}_entities.tsv",
            f"{IN_DIRECTORY}{folder}/drugprot_{folder}_relations.tsv"
        ]

        if folder == 'training':
            trainset = converter.load_data(files)
        else:
            devset = converter.load_data(files)
    
    random.seed(1234)
    random.shuffle(trainset)
    testset = trainset[:200]
    trainset = trainset[200:]

    lens = []

    lens.extend(converter.write_instances(trainset,f"{OUT_DIRECTORY}train.json"))
    lens.extend(converter.write_instances(devset,f"{OUT_DIRECTORY}dev.json"))
    lens.extend(converter.write_instances(testset,f"{OUT_DIRECTORY}test.json"))

    print("total", len(lens), "seqlen mean", int(np.mean(lens)), "median", int(np.median(lens)), "95th",
                  int(np.percentile(lens, 95)), "max", np.max(lens))


if __name__ == "__main__":
    import_drugprot()
    convert_drugprot()