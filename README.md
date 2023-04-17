# Deep active learning for biomedical relation extraction

Code and data to study deep active learning for biomedical relation extraction (bioRE).

Data sets can be found in the [data](/data/) folder.

Scripts to reproduce the results of the article can be found in the [launchers](/launchers/) folder.

The experiments were conducted on a server with Ubuntu Desktop 20.04.5 LTS (GNU/Linux 5.15.0-56-generic x86_64) operating system, Nvidia driver 470.161.03, CUDA version 11.4, with 32GB RAM on 2 Asus GTX 1080 TI GPUs. You may want to modify the `config_accelerate.yaml` file to fit your hardware.

An environment to run the experiments can be built with the `environment_pytorch.yaml` file using conda :
````
conda env create -f environment_pytorch.yaml
````
and then activated with :
````
conda activate pytorch
````

## Acknowledgements

This project was realised at the Interuniversity Institute of Bioinformatics in Brussels (IB2), a collaborative bioinformatics research initiative between Université Libre de Bruxelles (ULB) and Vrije Universiteit Brussel (VUB). This work was supported by the Service Public de Wallonie Recherche by DIGITALWALLONIA4.AI [2010235—ARIAC]; the European Regional Development Fund (ERDF) and the Brussels-Capital Region-Innoviris within the framework of the Operational Programme 2014-2020 through the ERDF-2020 project ICITY-RDI.BRU [27.002.53.01.4524]; an F.N.R.S-F.R.S PDR project [35276964]; Innoviris Joint R\&D project Genome4Brussels [2020 RDIR 55b]; and the Research Foundation-Flanders (F.W.O.) Infrastructure project associated with ELIXIR Belgium [I002819N].

## License

This work is under a MIT license.


## Cite us

See above by clicking on "Cite this repository"
