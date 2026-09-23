"""Dataset registry: CUB200-2011, Cars196 and Stanford Online Products."""

import datasets.cars196
import datasets.cub200
import datasets.stanford_online_products


def select(opt, **kwargs):
    dataset = opt.dataset
    normalized = dataset.lower()
    data_path = opt.source_path

    if 'cub200' in normalized:
        return datasets.cub200.give_dataloaders(opt, data_path)
    if 'cars196' in normalized:
        return datasets.cars196.give_dataloaders(opt, data_path)
    if 'online_products' in normalized or 'sop' in normalized:
        return datasets.stanford_online_products.give_dataloaders(opt, data_path)
    raise NotImplementedError(
        'Unsupported dataset: {}'.format(dataset))
