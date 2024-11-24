import h5py

file_path = '/share/DavidHong/data/leju/home/kuavo/rosbag_record/pick-lemon/output_directory/episode2.hdf5'
with h5py.File(file_path, 'r') as f:
    # 打印 action 数据集的前 5 行  目标
    action_data = f['action'][:5]
    print("Action data (first 5 rows):", action_data)

    # 打印 base_action 数据集的前 5 行
    base_action_data = f['base_action'][:5]
    print("Base action data (first 5 rows):", base_action_data)

    # 打印 observations/effort 数据集的前 5 行
    effort_data = f['observations/effort'][:5]
    print("Effort data (first 5 rows):", effort_data)

    # 打印 observations/images 中每个成员的前 5 行
    for image_key in f['observations/images']:
        image_data = f['observations/images'][image_key][:5]
        print(f"Image data for {image_key} (first 5 rows):", image_data)

    # 打印 observations/qpos 数据集的前 5 行  传感器
    qpos_data = f['observations/qpos'][:5]
    print("Qpos data (first 5 rows):", qpos_data)

    # 打印 observations/qvel 数据集的前 5 行
    qvel_data = f['observations/qvel'][:5]
    print("Qvel data (first 5 rows):", qvel_data)
