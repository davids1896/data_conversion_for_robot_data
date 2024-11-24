import rosbag
import json
import os
import cv2
from cv_bridge import CvBridge
import numpy as np
from scipy.spatial.transform import Rotation as R  # 用于四元数到欧拉角的转换

def extract_and_save_all_bags(bag_directory, output_folder):
    # 获取目录下所有.bag文件，并按文件名排序
    bag_files = sorted([f for f in os.listdir(bag_directory) if f.endswith('.bag')])
    
    # 依次处理每个bag文件
    for episode_index, bag_file in enumerate(bag_files):
        bag_path = os.path.join(bag_directory, bag_file)
        episode_folder = os.path.join(output_folder, f"episode{episode_index + 1}")
        extract_and_save(bag_path, episode_folder)

def extract_and_save(bag_file, episode_folder):
    bag = rosbag.Bag(bag_file)
    bridge = CvBridge()
    
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(episode_folder):
        os.makedirs(episode_folder)
    
    # 创建存储图像的文件夹
    os.makedirs(os.path.join(episode_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(episode_folder, 'depth_images'), exist_ok=True)

    images = []
    depth_images=[]
    poses = []
    cmd_poses = []
    hand_positions = []
    number=0

    # 从rosbag读取数据
    for topic, msg, t in bag.read_messages(topics=['/camera/color/image_raw',
                                                   '/camera/depth/image_rect_raw',
                                                   '/drake_ik/real_arm_hand_pose',
                                                   '/drake_ik/cmd_arm_hand_pose',
                                                   '/robot_hand_position']):
        if topic == '/camera/color/image_raw':
            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            images.append((msg.header.stamp.to_nsec(), cv_image))
            
        elif topic == '/camera/depth/image_rect_raw':
            number=number+1
            print(number)
            cv_image = bridge.imgmsg_to_cv2(msg, "passthrough")
            depth_images.append((msg.header.stamp.to_nsec(), cv_image))
            
        elif topic == '/drake_ik/real_arm_hand_pose':
            poses.append((msg.header.stamp.to_nsec(), msg))
        elif topic == '/drake_ik/cmd_arm_hand_pose':
            cmd_poses.append((msg.header.stamp.to_nsec(), msg))
        elif topic == '/robot_hand_position':
            hand_positions.append((msg.header.stamp.to_nsec(), msg.left_hand_position[0] != 0))
    
    # 按时间戳排序
    images.sort(key=lambda x: x[0])
    depth_images.sort(key=lambda x: x[0])
    poses.sort(key=lambda x: x[0])
    cmd_poses.sort(key=lambda x: x[0])
    hand_positions.sort(key=lambda x: x[0])

    data_list = []  # 用于存储所有图像和对应信息的列表

    # 处理每一帧图像，并找到最接近的姿态和手部位置
    for idx, (img_stamp, image) in enumerate(images):
        depth_image=min(depth_images, key=lambda x: abs(x[0] - img_stamp))
        pose = min(poses, key=lambda x: abs(x[0] - img_stamp))
        cmd_pose = min(cmd_poses, key=lambda x: abs(x[0] - img_stamp))
        hand_pos = min(hand_positions, key=lambda x: abs(x[0] - img_stamp))
        
        # 保存图像
        img_path = os.path.join(episode_folder, 'images', f"{idx+1}.jpg")
        depth_image_data=depth_image[1]
        depth_img_path = os.path.join(episode_folder, 'depth_images', f"{idx+1}.jpg")
        cv2.imwrite(img_path, image)
        cv2.imwrite(depth_img_path, depth_image[1])

        # 四元数转换为欧拉角 (RPY)
        quat_real = pose[1].left_pose.quat_xyzw
        quat_cmd = cmd_pose[1].left_pose.quat_xyzw
        
        # 使用scipy的Rotation模块进行转换
        r_real = R.from_quat([quat_real[0], quat_real[1], quat_real[2], quat_real[3]])
        r_cmd = R.from_quat([quat_cmd[0], quat_cmd[1], quat_cmd[2], quat_cmd[3]])
        
        euler_real = r_real.as_euler('xyz', degrees=True)  # 欧拉角，以度为单位
        euler_cmd = r_cmd.as_euler('xyz', degrees=True)

        # 创建JSON条目
        data = {
            "index": idx + 1,
            "image": img_path,
            "depth_image": depth_img_path,
            "pos_xyz": list(pose[1].left_pose.pos_xyz),
            "quat_xyzw": list(pose[1].left_pose.quat_xyzw),
            "cmd_pos_xyz": list(cmd_pose[1].left_pose.pos_xyz),
            "cmd_quat_xyzw": list(cmd_pose[1].left_pose.quat_xyzw),
            "gripper": hand_pos[1],
            "language_instruction": "make juice",
            "joint_angles": list(pose[1].left_pose.joint_angles),
            "cmd_joint_angles": list(cmd_pose[1].left_pose.joint_angles),
            "euler_rpy": euler_real.tolist(),  # 实际姿态的欧拉角
            "cmd_euler_rpy": euler_cmd.tolist()  # 命令姿态的欧拉角
        }
        
        data_list.append(data)  # 将数据添加到列表中

    # 将所有数据保存到一个data.json文件中
    with open(os.path.join(episode_folder, 'data.json'), 'w') as f:
        json.dump(data_list, f, indent=4)

    bag.close()

# 示例用法
extract_and_save_all_bags('/home/lab/rosbag_record/test', '/home/lab/rosbag_record/output_directory')