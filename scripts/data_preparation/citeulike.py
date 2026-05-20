from datarec.datasets import CiteULike
# download dataset
from datarec.processing.kcore import UserItemIterativeKCore
from datarec.splitters.user_stratified.hold_out import UserStratifiedHoldOut
from datarec.io.writers import write_transactions_tabular as write_tabular
from recdistill.paths import CITEULIKE, dataset_directory, dataset_filepath

dataset_name = CITEULIKE

dataset = CiteULike(version="a")

dataset = dataset.prepare_and_load()

kcore = UserItemIterativeKCore(cores=[5, 5])
dataset = kcore.run(dataset)

dataset.data['rating'] = 1
dataset.rating_col = 'rating'
print(len(dataset))

print('Train validation test splitting...')
spl = UserStratifiedHoldOut(test_ratio=0.2, val_ratio=0.1)
split = spl.run(dataset)

print('Train validation test splitting done')
print('Writing dataset...')

dataset_dir = dataset_directory(dataset_name=dataset_name, create_if_not_exists=True)
print(f'Dataset directory: {dataset_dir}')

write_tabular(dataset,
              filepath=dataset_filepath(dataset_name, 'processed', exists=False),
              sep='\t',
              header=False,
              decimal='.',
              include_user=True,
              include_item=True,
              include_rating=True,
              include_timestamp=False)

print(len(split['train']))

write_tabular(split['train'].to_rawdata(),
              filepath=dataset_filepath(dataset_name, 'train', exists=False),
              sep='\t',
              header=False,
              decimal='.',
              include_user=True,
              include_item=True,
              include_rating=True,
              include_timestamp=False)

print(len(split['test']))

write_tabular(split['test'].to_rawdata(),
              filepath=dataset_filepath(dataset_name, 'test', exists=False),
              sep='\t',
              header=False,
              decimal='.',
              include_user=True,
              include_item=True,
              include_rating=True,
              include_timestamp=False)

print(len(split['val']))

write_tabular(split['val'].to_rawdata(),
              filepath=dataset_filepath(dataset_name, 'val', exists=False),
              sep='\t',
              header=False,
              decimal='.',
              include_user=True,
              include_item=True,
              include_rating=True,
              include_timestamp=False)
