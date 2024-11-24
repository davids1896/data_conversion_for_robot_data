import h5py
import json
import os
from PIL import Image
import numpy as np

# 定义文件路径
json_path = '/share/DavidHong/data/leju/home/kuavo/rosbag_record/pick-lemon/output_directory/episode2/data.json'
images_dir = '/share/DavidHong/data/leju/home/kuavo/rosbag_record/pick-lemon/output_directory/episode2/images'
hdf5_path = '/share/DavidHong/data/leju/home/kuavo/rosbag_record/pick-lemon/output_directory/episode2.hdf5'

# 打开 JSON 文件并加载数据
with open(json_path, 'r') as f:
    data = json.load(f)

# 创建 HDF5 文件
with h5py.File(hdf5_path, 'w') as hdf5_file:
    # 创建 datasets
    action_dataset = hdf5_file.create_dataset('action', (len(data), len(data[0]["cmd_joint_angles"])), dtype='f')
    qpos_dataset = hdf5_file.create_dataset('observations/qpos', (len(data), len(data[0]["joint_angles"])), dtype='f')
    
    # 假设所有图像大小相同，这里使用第一张图片的大小来初始化 dataset
    first_image_path = os.path.join(images_dir, '1.jpg')
    first_image = Image.open(first_image_path)
    img_shape = (len(data),) + np.array(first_image).shape
    images_dataset = hdf5_file.create_dataset('observations/images', img_shape, dtype='uint8')

    # 将 JSON 数据填充到 HDF5 文件中
    for i, entry in enumerate(data):
        # 保存 action 数据
        action_dataset[i] = entry["cmd_joint_angles"]
        
        # 保存 qpos 数据
        qpos_dataset[i] = entry["joint_angles"]
        
        # 保存 images 数据
        img_path = os.path.join(images_dir, f"{entry['index']}.jpg")
        if os.path.exists(img_path):
            image = Image.open(img_path)
            images_dataset[i] = np.array(image)
        else:
            print(f"Image not found: {img_path}")

print("Data successfully saved to HDF5 format.")
