import h5py

# 打开 HDF5 文件（只读模式）
file_path = '/share/DavidHong/data/leju/data_scp/output_directory_new/hdf5_train/episode1.hdf5'
with h5py.File(file_path, 'r') as f:
    # 查看文件中的顶层组和数据集
    print("Top level groups:", list(f.keys()))
    
    # 如果知道具体的数据集名称，例如 'dataset1'
    if 'dataset1' in f:
        dataset = f['dataset1'][:]  # 加载整个数据集到内存
        print("Dataset contents:", dataset)

    # 遍历文件结构以查看全部层次结构
    def print_structure(name, obj):
        print(name, obj)
    f.visititems(print_structure)  # 打印文件结构
