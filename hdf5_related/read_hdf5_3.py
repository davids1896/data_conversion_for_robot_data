import h5py

file_path = '/share/DavidHong/data/leju/data_scp/output_directory_new/episode3.hdf5'  # 修改为实际文件路径
with h5py.File(file_path, 'r') as f:
    # 打印 action 数据集的前 5 行
    action_data = f['action'][:5]
    print("Action data (first 5 rows):", action_data)

    # 打印 observations/images 数据集中的前 5 张图片
    images_data = f['observations/images'][:5]
    #print("Observations images data (first 5 entries):", images_data)

    # 打印 observations/qpos 数据集的前 5 行
    qpos_data = f['observations/qpos'][:5]
    print("Qpos data (first 5 rows):", qpos_data)
