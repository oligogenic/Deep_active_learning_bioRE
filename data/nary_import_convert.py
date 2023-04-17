import subprocess, tarfile, os, random, json
import numpy as np
from nary import NaryConverter

def download_nary():
    subprocess.call(['curl', '-L', '-o', 'data.tgz',
                     'https://github.com/freesunshine0316/nary-grn/blob/master/peng_data/data.tgz?raw=true'])
    subprocess.call(['mkdir', '-p', 'nary_hf/origin'])

    with tarfile.open('data.tgz','r:gz') as infile:
        infile.extractall('nary_hf/origin')

    subprocess.call(['rm', 'data.tgz'])

def convert_nary():
    in_directory = 'nary_hf/origin/'
    out_directory = 'nary_hf/processed/'
    tag_dict = {
            'variant': '@VARIANT$',
            'gene': '@GENE$',
            'drug': '@DRUG$'
        }
    if not os.path.exists(out_directory):
        os.mkdir(out_directory)
    converter = NaryConverter(tag_dict)

    for type in ['drug_gene_var/', 'drug_var/']:

        instances = []

        # put all data in each file
        for cv in range(5):
            in_directory_cv = f'{in_directory}{type}{cv}/'
            out_directory_cv = f'{out_directory}{type}'
            if not os.path.exists(out_directory_cv):
                os.mkdir(out_directory_cv)

            instances.append(
                converter.load_data(
                    [
                        f'{in_directory_cv}data_graph_1',
                        f'{in_directory_cv}data_graph_2'
                    ]
                )
            )

        all_train = []
        for i in range(4):
            all_train.extend(instances[i])
        testset = instances[4]

        random.seed(1234)
        random.shuffle(all_train)
        devset = all_train[:200]
        trainset = all_train[200:]

        lens = []
        with open(f"nary_hf/processed/{type}test.json", "w", encoding="utf-8") as outf:
            for instance in testset:
                out = {"id":instance.index, "sentence":instance.text, "label": instance.label}
                print(json.dumps(out, ensure_ascii=False), file=outf)
                lens.append(len(instance.tokens))
        print("total", len(lens), "seqlen mean", int(np.mean(lens)), "median", int(np.median(lens)), "95th",
                  int(np.percentile(lens, 95)), "max", np.max(lens))

        lens = []
        with open(f"nary_hf/processed/{type}dev.json", "w", encoding="utf-8") as outf:
            for instance in devset:
                out = {"id":instance.index, "sentence":instance.text, "label": instance.label}
                print(json.dumps(out, ensure_ascii=False), file=outf)
                lens.append(len(instance.tokens))
        print("total", len(lens), "seqlen mean", int(np.mean(lens)), "median", int(np.median(lens)), "95th",
                      int(np.percentile(lens, 95)), "max", np.max(lens))

        lens = []
        with open(f"nary_hf/processed/{type}train.json", "w", encoding="utf-8") as outf:
            for instance in trainset:
                out = {"id":instance.index, "sentence":instance.text, "label": instance.label}
                print(json.dumps(out, ensure_ascii=False), file=outf)
                lens.append(len(instance.tokens))
        print("total", len(lens), "seqlen mean", int(np.mean(lens)), "median", int(np.median(lens)), "95th",
                      int(np.percentile(lens, 95)), "max", np.max(lens))

if __name__ == "__main__":
    print("Dowloading...")
    download_nary()
    print("Converting...")
    convert_nary()