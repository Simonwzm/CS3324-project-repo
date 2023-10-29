# AGIQA-3K dataset support for PyIQA

We prepared AGIQA dataset support for the `pyiqa` module. 
To use the dataset in `pyiqa`, first make sure you configure the environment for `pyiqa` and can run the following code to train an IQA metric model (as long as it starts training, you can exit halfway, for it is just to make sure all the requirements and paths are correctly created).

```shell
# Run in shell in the root path of pyiqa.
# train for single experiment
python pyiqa/train.py -opt options/train/DBCNN/train_DBCNN.yml

# train N splits for small datasets
python pyiqa/train_nsplits.py -opt options/train/DBCNN/train_DBCNN.yml
```

PyIQA supports Windows, so you are able to run it on either Linux or Windows platform

To add the dataset to pyiqa, download the file `datasets\meta_info\meta_info_AGIQA-3K.csv` and `datasets\meta_info\AGIQA-3K.pkl` and place them locally to your cloned `pyiqa` repo while ensuring files are on the same path as shown in this repo. 

Then extract the AGIQA-3K dataset to the `datasets/` directory. Now the structure should be like

```
your_cloned_pyiqa_repo/
|
|___ datasets/
|    |
|    |___ AGIQA-3K/
|    |    |
|    |    |___ image1.jpg ...
|    |
|    |___ other_datasets/
|    |
|    |___ meta_info/
|         |
|         |___ AGIQA-3K.pkl
|         |
|         |___ meta_info_AGIQA-3K.csv
|
|___ other_dirs/

```

After preparing the dataset, you can try to train HyperIQA on this dataset.

If you have a GPU and have plenty of time, you can use the prepared training options in `options\train\train_AGICQ-3K.yml` which use 10-fold cross validation and set batch_size to 40, num_epoches to 100.

To use the prepared training options, first download  `options\train\train_AGICQ-3K.yml` and place it on the same path as shown in this repo like we previouly did in preparing for datasets. Then run the following command in shell in the root path of your pyiqa repo to start training.

```shell
python pyiqa/train_nsplits.py -opt options/train/train_AGICQ-3k.yml
```

You can modify the `.yml` file to change details in training.
