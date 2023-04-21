import os
import yaml

class ShapeNetDataset():

    def __init__(self, dataset_folder, categories=None):
        self.dataset_folder = dataset_folder

        if categories is None:
            categories = os.listdir(dataset_folder)
            categories = [c for c in categories
                          if os.path.isdir(os.path.join(dataset_folder, c))]
        categories.sort()
        print(categories)

        metadata_file = os.path.join(dataset_folder, "metadata.yaml")

        if(os.path.exists(metadata_file)):
            print("path: true")
            with open(metadata_file, 'r') as f:
                print("open: true")
                self.metadata = yaml.safe_load(f)
        else:
            self.metadata = {
                c: {'id': c, 'name': 'n/a'} for c in categories
            }

        for c_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = c_idx
            # print(c_idx, c)
            print(self.metadata[c])

        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                print('Category %s does not exist in dataset.' % c)

            split_file = os.path.join(subpath, 'train' + '.lst')
            if not os.path.exists(split_file):
                models_c = [f for f in os.listdir(
                        subpath) if os.path.isdir(os.path.join(subpath, f))]
                print(models_c, len(models_c))
            else:
                with open(split_file, 'r') as f:
                    models_c = f.read().split('\n')
            
            models_c = list(filter(lambda x: len(x) > 0, models_c))
            self.models += [
                    {'category': c, 'model': m, 'category_id': c_idx}
                    for m in models_c
                ]

    def __len__(self):
        return len(self.models)


dataset = ShapeNetDataset('D:/Projects/Main Project/test/data/ShapeNet')