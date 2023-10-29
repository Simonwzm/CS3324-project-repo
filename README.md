# AGIQA-3K dataset support for PyIQA

We build dataset support for AGIQA-3K to train using PyIQA. 
To use the dataset, first make sure to configure the environment for `pyiqa` and can run the following code to train an IQA metric model (as long as it starts training, you can exit halfway, for it is just to make sure all the requirements and paths are correctly created).

PyIQA supports Windows, so you are able to run it on either Linux or Windows platform

```shell
# Run in shell in the root path of pyiqa.
# train for single experiment
python pyiqa/train.py -opt options/train/DBCNN/train_DBCNN.yml

# train N splits for small datasets
python pyiqa/train_nsplits.py -opt options/train/DBCNN/train_DBCNN.yml
```

To add the dataset to pyiqa, download the file `datasets\meta_info\meta_info_AGIQA-3K.csv` and `datasets\meta_info\AGIQA-3K.pkl` and copy them to the same path of your pyiqa repo. 

Then extract AGIQA-3K dataset to the `datasets/` directory. Now the structure should be like

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

After preparing the dataset, you can can try to train HyperIQA on this dataset.

If you have a GPU and have plenty of time, you can use the prepared training options in `options\train\train_AGICQ-3K.yml`, which use 10-fold cross validation and set batch_size to 40, num_epoches to 100.

To use the prepared training options, first download  `options\train\train_AGICQ-3K.yml` and place it to the same place in your pyiqa project as we previouly did for preparing datasets. Then run the following command in shell in the root path of your pyiqa repo 

```shell
python pyiqa/train_nsplits.py -opt options/train/train_AGICQ-3k.yml
```

You can modify the `.yml` file to change details in training.



