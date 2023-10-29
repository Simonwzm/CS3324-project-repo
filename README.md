# AGIQA-3K dataset for PyIQA

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


