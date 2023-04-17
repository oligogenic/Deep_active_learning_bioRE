import json, subprocess
import numpy as np

def import_aimed():
    subprocess.call(['mkdir', '-p', 'aimed/processed'])

def convert_aimed():
    IN_DIRECTORY = 'aimed/1/'
    OUT_DIRECTORY = 'aimed/processed/'
    files = ['train.tsv', 'test.tsv', 'dev.tsv']

    lens = []
    index_add = 1
    for file in files:
        with open(f"{IN_DIRECTORY}{file}", 'r', encoding="utf-8") as infile:
            if 'test' in file:
                infile.readline() # pass title
            with open(f"{OUT_DIRECTORY}{file.replace('tsv','json')}", 'w', encoding='utf8') as outf:
                for line in infile:
                    if 'test' in file:
                        index, text, label = line.strip().split('\t')
                        index =str( int(index) + index_add)
                    else:
                        index = str(index_add)
                        text, label = line.strip().split('\t')
                        index_add +=1
                    out = {"id":index, "sentence":text, "label": label}
                    print(json.dumps(out, ensure_ascii=False), file=outf)
                    lens.append(len(text.split()))
    print("total", len(lens), "seqlen mean", int(np.mean(lens)), "median", int(np.median(lens)), "95th",
                      int(np.percentile(lens, 95)), "max", np.max(lens))

if __name__ == "__main__":
    import_aimed()
    convert_aimed()