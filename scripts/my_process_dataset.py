import os
import scipy.io as sio
import random
import numpy
import pickle
import csv
import pandas as pd

def get_random_splits(seed=123):
    random.seed(seed)
    total_num = 300
    all_prompts_index = list(range(total_num))
    num_splits = 10
    save_path = f'./datasets/meta_info/my_dataset_seed{seed}.pkl'

    ratio = [0.8, 0.2]  # train/val/test
    sep_index = int(round(0.8 * total_num))

    split_info = {}

    for j in range(num_splits):
        random.shuffle(all_prompts_index)
        all_train_index = all_prompts_index[:sep_index]
        all_val_index = all_prompts_index[sep_index:]
        train_img = []
        val_img = []
        # for each index in all_train/val_index list, call get_img_in_prompt(index) and += the returning list to train_img/val_img
        for i in all_train_index:
            train_img += get_img_in_prompt(i)
        for i in all_val_index:
            val_img += get_img_in_prompt(i)

        split_info[j + 1] = {'train': train_img, 'val': val_img}
    print(split_info[3]['train'], len(split_info[1]['train']))
    print(split_info[3]['val'], len(split_info[1]['val']))
    with open(save_path, 'wb') as sf:
        pickle.dump(split_info, sf)


def get_meta_info():
    info_file = './3kdata.csv'
    save_meta_path = './datasets/meta_info/meta_info_my_dataset.csv'
    # split_info = {'train': [], 'val': [], 'test': []}
    df_info = pd.read_csv(info_file)
    # prompts = df_info['prompt'].unique().tolist()
    df = df_info[['name', 'mos_quality', 'std_quality']]
    df.to_csv(save_meta_path, index=False)
    # print(len(prompts))

def get_img_in_prompt(prompt_index):
    info_file = './3kdata.csv'
    df = pd.read_csv(info_file)
    prompts = df['prompt'].unique().tolist()
    prompt = prompts[prompt_index]
    # get all rows with column "prompt" value == prompt
    df_prompt = df.loc[df['prompt'] == prompt]
    # get the value of the "name" column of these rows, convert to list
    img_list = df_prompt['name'].tolist()
    return img_list





if __name__ == '__main__':
    # get_meta_info()
    get_random_splits()


