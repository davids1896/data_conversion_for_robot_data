import h5py
import json
import os
from PIL import Image
import numpy as np

# 定义根目录路径
root_dir = '/share/DavidHong/data/leju/data_scp/output_directory_new'

# 遍历每个 episode 文件夹
for episode_folder in os.listdir(root_dir):
    episode_path = os.path.join(root_dir, episode_folder)
    
    # 检查是否为目录，且包含 data.json 和 images 文件夹
    json_path = os.path.join(episode_path, 'data.json')
    images_dir = os.path.join(episode_path, 'images')
    if os.path.isdir(episode_path) and os.path.isfile(json_path) and os.path.isdir(images_dir):
        # 定义 HDF5 文件路径
        hdf5_path = os.path.join(root_dir, f"{episode_folder}.hdf5")

        # 打开 JSON 文件并加载数据
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 创建 HDF5 文件
        with h5py.File(hdf5_path, 'w') as hdf5_file:
            # 创建 datasets
            action_dataset = hdf5_file.create_dataset('action', (len(data), 8), dtype='f')
            qpos_dataset = hdf5_file.create_dataset('observations/qpos', (len(data), 8), dtype='f')

            # 假设所有图像大小相同，这里使用第一张图片的大小来初始化 dataset
            first_image_path = os.path.join(images_dir, '1.jpg')
            if os.path.exists(first_image_path):
                first_image = Image.open(first_image_path)
                img_shape = (len(data),) + np.array(first_image).shape
                images_dataset = hdf5_file.create_dataset('observations/images/camera_high', img_shape, dtype='uint8')

                # 将 JSON 数据填充到 HDF5 文件中
                for i, entry in enumerate(data):
                    # 保存 action 数据
                    action_values = entry["cmd_joint_angles"][:7] + [1.0 if entry["gripper"] else 0.0]
                    action_dataset[i] = action_values

                    # 保存 qpos 数据
                    qpos_values = entry["joint_angles"][:7] + [1.0 if entry["gripper"] else 0.0]
                    qpos_dataset[i] = qpos_values

                    # 保存 images 数据到 observations/images/camera_high
                    img_path = os.path.join(images_dir, f"{entry['index']}.jpg")
                    if os.path.exists(img_path):
                        image = Image.open(img_path)
                        images_dataset[i] = np.array(image)
                    else:
                        print(f"Image not found: {img_path}")

            print(f"Data successfully saved to {hdf5_path}")
    else:
        print(f"Skipping {episode_folder}: missing data.json or images directory.")
