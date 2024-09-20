from data.data_utils import MergedDataset

from data.cub import get_cub_datasets

from copy import deepcopy
import pickle
import os

from config import osr_split_dir


get_dataset_funcs = {
    'CUB_200_2011': get_cub_datasets,
}


def get_datasets(dataset_name, train_transform, test_transform, args):
# 用于加载和处理指定的数据集，并为训练和测试准备合适的数据对象，同时处理类别标签的转换
    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """

    #
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(train_transform=train_transform, test_transform=test_transform,
                            train_classes=args.train_classes,
                            prop_train_labels=args.prop_train_labels,
                            split_train_val=False)
    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            dataset.target_transform = target_transform

    # Train split (labelled and unlabelled classes) for training
    train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                  unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

    test_dataset = datasets['test']
    unlabelled_train_examples_test = deepcopy(datasets['train_unlabelled'])
    unlabelled_train_examples_test.transform = test_transform

    return train_dataset, test_dataset, unlabelled_train_examples_test, datasets

# 根据 CUB 数据集的需求和配置来设置训练和未标记类别，并处理自定义的类别分割。
def get_class_splits(args):
    # 根据所使用的数据集（在这里是 CUB_200_2011 数据集），返回适合的类别划分（已标记类和未标记类）
    # For FGVC datasets, optionally return bespoke splits
    if args.dataset_name == 'CUB_200_2011':
        if hasattr(args, 'use_ssb_splits'):
            use_ssb_splits = args.use_ssb_splits
        else:
            use_ssb_splits = False

    # -------------
    # GET CLASS SPLITS
    # -------------

    if args.dataset_name == 'CUB_200_2011':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'cub_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(100)
            args.unlabeled_classes = range(100, 200)

    else:

        raise NotImplementedError

    return args
