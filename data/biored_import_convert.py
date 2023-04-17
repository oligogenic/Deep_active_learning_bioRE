import subprocess, json
import numpy as np

from biored import BioREDConverter

relation_types = [
        ('gene', 'disease'),
        ('gene', 'gene'),
        ('gene', 'chemical'),
        ('disease', 'variant'),
        ('chemical', 'disease'),
        ('chemical', 'variant'),
        ('chemical', 'chemical'),
        ('variant', 'variant')
]


def import_biored():
    subprocess.call(['mkdir', '-p', 'BioRED/processed'])
    subprocess.call(['mkdir', '-p', f'BioRED/processed/all'])
    for entity_type_1, entity_type_2 in relation_types:
        subprocess.call(['mkdir', '-p', f'BioRED/processed/{entity_type_1}_{entity_type_2}'])



def convert_biored():
    # first transform to json with all the entities and relations attached to a text
    # with that file, we can create the different files for all subtypes of relation (all, D-G, D-C, G-C, G-G, D-V, C-V,
    # C-C, V-V)
    IN_DIRECTORY = 'BioRED/'
    OUT_DIRECTORY = 'BioRED/processed/'
    files = [
        "Dev.PubTator",
        "Train.PubTator",
        "Test.PubTator"
    ]

    tag_dict = {
        'chemical': '@CHEMICAL$',
        'disease' : '@DISEASE$',
        'gene' : '@GENE$',
        'variant' : '@VARIANT$'
    }
    converter = BioREDConverter(tag_dict)

    for file in files:
        all_documents = converter.produce_documents(f"{IN_DIRECTORY}{file}")

        with open(f"{OUT_DIRECTORY}{file.lower().replace('pubtator','json')}", 'w', encoding='utf-8') as outf:
            json.dump(all_documents, outf, indent=2)

        all_instances = []
        for entity_type_1, entity_type_2 in relation_types:
            relation_instances = []

            for pmid in all_documents:
                relation_instances.extend(converter.extract_instances(
                    pmid, 
                    all_documents[pmid]['relations'],
                    all_documents[pmid]['text'],
                    all_documents[pmid]['entities'][entity_type_1],
                    all_documents[pmid]['entities'][entity_type_2],
                    [tag_dict[entity_type_1]] + [tag_dict[entity_type_2]]
                    )
                )
            
            converter.write_instances(relation_instances, f"{OUT_DIRECTORY}{entity_type_1}_{entity_type_2}/{file.lower().replace('pubtator','json')}")
            
            all_instances.extend(relation_instances)
        
        lens = converter.write_instances(all_instances, f"{OUT_DIRECTORY}all/{file.lower().replace('pubtator','json')}")
        print("total", len(lens), "seqlen mean", int(np.mean(lens)), "median", int(np.median(lens)), "95th",
                  int(np.percentile(lens, 95)), "max", np.max(lens))


if __name__ == "__main__":
    import_biored()
    convert_biored()