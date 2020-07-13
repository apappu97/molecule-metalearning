from chembl_webresource_client.new_client import new_client
import pickle
from collections import defaultdict

assay = new_client.assay

with open('../chemprop/data/chembl_assay_names.pickle', 'rb') as handle:
    assay_names = pickle.load(handle)

num_skipped = 0
assays_skipped = []
assay_name_to_type = {}
assay_type_to_names = defaultdict(list)
for assay_name in assay_names:
    try:
        res = assay.get(assay_name)
        assay_type = res['assay_type']
        assay_name_to_type[assay_name] = assay_type
        assay_type_to_names[assay_type].append(assay_name)
    except Exception as e:
        print('Skipping assay type: {}'.format(assay_name))
        print('Incurred error: {}'.format(e))
        num_skipped+=1
        assays_skipped.append(assay_name)

print('Skipped {} assays total because 404 error incurred'.format(num_skipped))

with open('../chemprop/data/chembl_assay_name_to_type.pickle', 'wb') as handle:
    pickle.dump(assay_name_to_type, handle)

with open('../chemprop/data/chembl_assay_type_to_names.pickle', 'wb') as handle:
    pickle.dump(assay_type_to_names, handle)

with open('../chemprop/data/chembl_assay_type_legend.pickle', 'wb') as handle:
    assay_legend = {'B': 'Binding', 'F': 'Functional', 'A': 'ADME', 'T':
            'Toxicity', 'P': 'Physicochemical'}
    pickle.dump(assay_legend, handle)

with open('../chemprop/data/chembl_assays_skipped_404.pickle', 'wb') as handle:
    pickle.dump(assays_skipped, handle)

