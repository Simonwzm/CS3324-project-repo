import pickle
with open('./datasets/meta_info/.pkl', 'rb') as f:
    meta_info = pickle.load(f)
    print(meta_info.keys())
    print([[len(meta_info[y][x]) for x in meta_info[y].keys()] for y in meta_info.keys()])