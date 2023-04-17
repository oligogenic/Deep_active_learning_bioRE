# Overview
You can find below the links for download and publications for the different data sets used, as well as the commands to optionally convert the scripts to the format used in the experiments.

------

## Data sets download

### AIMED

- Files at https://drive.google.com/file/d/1dn2yDKj7-3SsyKQ5Zm_5sTlLxTCfqQpy/view 
- Published by https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-022-04642-w

>(total 5834 seqlen mean 30 median 27 95th 70 max 124)

### BioRED

- Files at https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/
- Published in https://academic.oup.com/bib/article/23/5/bbac282/6645993

>test : total 6395 seqlen mean 279 median 273 95th 363 max 369

>dev : total 6086 seqlen mean 274 median 269 95th 404 max 414

>train : total 19105 seqlen mean 251 median 255 95th 335 max 496

### BC5CDR

- Files downloaded with `curl -k -o CDR_Data.zip https://biocreative.bioinformatics.udel.edu/media/store/files/2016/CDR_Data.zip`
- Published in https://pubmed.ncbi.nlm.nih.gov/27161011/

>(total 30056 seqlen mean 212 median 216 95th 352 max 541)

### DDI13 & ChemProt

- Files at https://microsoft.github.io/BLURB/
- Published in https://pubmed.ncbi.nlm.nih.gov/23906817/

### DrugProt

- Files at https://zenodo.org/record/5042151#.YzK4UlJBzao

>(total 599362 seqlen mean 260 median 257 95th 386 max 847)

### Nary

- Files at https://github.com/freesunshine0316/nary-grn 
- Published in https://aclanthology.org/D18-1246/

#### DV

>test : total 1211 seqlen mean 59 median 55 95th 117 max 226

>dev : total 200 seqlen mean 58 median 53 95th 112 max 343

>train : total 4676 seqlen mean 61 median 54 95th 118 max 1300

#### DGV

>test : total 1298 seqlen mean 74 median 69 95th 134 max 291

>dev : total 200 seqlen mean 71 median 62 95th 140 max 343

>train : total 5489 seqlen mean 73 median 66 95th 134 max 446

------

## Data sets Conversion
For each instance, the entities are masked with their corresponding mask accoridng to their entity type, i.e. @VARIANT$, @GENE$, @DRUG$, @DISEASE$ and @CHEMICAL$ for respectively variants, genes, drugs, diseases and chemicals.

Processed data sets are in a json format commonly used with huggingface library. To convert a desired data set, you can just used the following command

```bash
python DATASET_import_convert
```

with DATASET replaced with the data set you wish to convert. Be warned that the AIMED, BioRED and DrugProt data sets need to be downloaded prior to the conversion.