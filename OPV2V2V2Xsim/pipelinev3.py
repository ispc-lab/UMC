import os
import open3d as o3d
import numpy as np
import uuid
import json
import shutil
import yaml
import argparse


def generate_uuid(name):
    """
    :param name:
        1. dataset-stype-log-(log_id)
        2. dataset-stype-scene-(number)
        3. dataset-stype-sample-(scene_id.frame_id)
        4. dataset-stype-sampled-(scene_id.frame_id.agent_id.sensor_type)  # 和 ego_pose_token 共享
        5. dataset-stype-calib-(scene_id.agent_id.sensor_type)
        6. dataset-stype-attr-(type[car, pedestrian])
        7. dataset-stype-inst-(car_id)
        8. dataset-stype-samplea-(scene_id.frame_id.car_id)
        9. dataset-stype-category-(type[car, pedestrian])
        10. dataset-stype-sensor-(agent_id)
        11. dataset-stype-map-(map_id)
    :return: 32 bit token
    """
    return uuid.uuid5(uuid.NAMESPACE_DNS, name).hex


def check_folder(folder_path):
    """
    :param folder_path: check whether there are folder
    :return: None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


def listdir(path):
    return next(os.walk(path))[1]


def clear_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def pcd_yaml_account(root):
    """
    :param root:
    :return:
    """
    pcd_dot = ".pcd"
    yaml_dot = ".yaml"
    walk_generator = os.walk(root)
    account = 0
    pcd_file = list()
    for root_path, dirs, files in walk_generator:
        if len(files) < 1:
            continue
        files.sort()
        for file in files:
            file_name, suffix_name = os.path.splitext(file)
            if suffix_name == pcd_dot:
                account += 1
                pcd_file.append(file_name)
            elif suffix_name == yaml_dot:
                account += 1
    return account, pcd_file


def scene_json(record_id, scene_id, seq_len, stype, save_dir):

    scene_template = {
        "token": generate_uuid('opv2v-'+stype+'-scene-'+record_id+'-'+scene_id),
        "log_token": generate_uuid('opv2v-'+stype+'-log-'+record_id),
        "nbr_samples": seq_len,
        "first_sample_token": generate_uuid('opv2v-'+stype+'-sample-'+record_id+'-'+scene_id+'.'+str(0)),
        "last_sample_token":  generate_uuid('opv2v-'+stype+'-sample-'+record_id+'-'+scene_id+'.'+str(seq_len-1)),
        "name": "scene_" + record_id + "_" + scene_id,
        "description": "The scene of " + record_id + "-" + scene_id
    }
    json_save = json.dumps([scene_template], indent=0)
    with open(os.path.join(save_dir, 'scene.json'), 'a') as jsonfile:
        jsonfile.write(json_save)


def sample_json_tmp():
    return {
        "token": "",
        "timestamp": 0,
        "prev": "",
        "next": "",
        "scene_token": ""}


def sample_json(record_id, scene_id, seq_len, stype, save_dir):
    with open(os.path.join(save_dir, 'sample.json'), 'a') as jsonfile:
        save_list = list()
        for time in range(seq_len):
            tmp_sample = sample_json_tmp()
            tmp_sample["token"] = generate_uuid('opv2v-'+stype+'-sample-'+record_id+'-'+scene_id+'.'+str(time))
            tmp_sample["timestamp"] = time
            if time != 0:
                tmp_sample["prev"] = generate_uuid('opv2v-'+stype+'-sample-'+record_id+'-'+scene_id+'.'+str(time-1))
            if time != (seq_len-1):
                tmp_sample['next'] = generate_uuid('opv2v-'+stype+'-sample-'+record_id+'-'+scene_id+'.'+str(time+1))
            tmp_sample["scene_token"] = generate_uuid('opv2v-'+stype+'-scene-'+record_id+'-'+scene_id)

            save_list.append(tmp_sample)

        jsonfile.write(json.dumps(save_list, indent=0))


def pcd_convert(record_id, scene_id, target_file, source_dir, save_dir):
    for frame_id in range(len(target_file)):
        lidar = list()
        pcd = o3d.io.read_point_cloud(os.path.join(source_dir, target_file[frame_id]+".pcd"), format='pcd')
        points = np.array(pcd.points)
        for line in points:
            line_convert = list(map(float, line))
            if len(line) == 3:
                [line_convert.append(0) for i in range(2)]
            if len(line) == 4:
                line_convert.append(0)
            lidar.append(line_convert)
        pl = np.array(lidar, dtype=np.float32).reshape(-1)
        attach = 'scene_' + record_id + '-' + scene_id + '-' + target_file[frame_id] + '.pcd'
        pl.tofile(os.path.join(save_dir, attach + '.bin'))
        # print("The {} of car {} 's {} scene of {} has been done!".format(record_id, source_dir.split('/')[-1], scene_id, target_file[frame_id]))


def sampled_json_tmp():
    return {
        "token": "",
        "sample_token": "",
        "ego_pose_token": "",
        "calibrated_sensor_token": "",
        "timestamp": 0,
        "fileformat": "",
        "is_key_frame": True,
        "height": 0,
        "width": 0,
        "filename": "",
        "prev": "",
        "next": ""}


def sample_data_json(record_id, scene_id, seq_len,
                     stype, sensor_type, activate_car, target_file, save_dir):
    with open(os.path.join(save_dir, 'sample_data.json'), 'a') as jsonfile:
        save_list = list()
        if sensor_type == 'lidar':
            fileformat = 'pcd'
        for idx in range(activate_car):
            for time in range(seq_len):
                tmp_sampled = sampled_json_tmp()
                tmp_sampled["token"] = generate_uuid('opv2v-'+stype+'-sampled-'+record_id+'-'+scene_id+'.'+str(time)+'.'+str(idx)+'.'+sensor_type)
                tmp_sampled["sample_token"] = generate_uuid('opv2v-'+stype+'-sample-'+record_id+'-'+scene_id+'.'+str(time))
                tmp_sampled["ego_pose_token"] = tmp_sampled["token"]
                tmp_sampled["calibrated_sensor_token"] = generate_uuid('opv2v-'+stype+'-calib-'+record_id+'-'+scene_id+'.'+str(idx)+'.'+sensor_type)
                tmp_sampled["timestamp"] = time
                tmp_sampled["fileformat"] = fileformat
                tmp_sampled["filename"] = "sweeps/" + "LIDAR_TOP_id_" + str(idx) + "/" + 'scene_' + record_id + '-' + scene_id + '-' + target_file[time] + '.pcd.bin'
                if time != 0:
                    tmp_sampled["prev"] = generate_uuid('opv2v-'+stype+'-sampled-'+record_id+'-'+scene_id+'.'+str(time-1)+'.'+str(idx)+'.'+sensor_type)
                if time != (seq_len-1):
                    tmp_sampled["next"] = generate_uuid('opv2v-'+stype+'-sampled-'+record_id+'-'+scene_id+'.'+str(time+1)+'.'+str(idx)+'.'+sensor_type)
                save_list.append(tmp_sampled)

        jsonfile.write(json.dumps(save_list, indent=0))


def ego_json_tmp():
    return {
        "token": "",
        "timestamp": 0,
        "rotation": list(),
        "translation": list()}


def convert_quaternion(roll, yaw, pitch):
    """
    :param roll: roll -> x
    :param yaw:  yaw -> z
    :param pitch: pitch - > y
    :return:
    """
    cy = np.cos(yaw * 0.5 * np.pi/180)
    sy = np.sin(yaw * 0.5 * np.pi/180)
    cp = np.cos(pitch * 0.5 * np.pi/180)
    sp = np.sin(pitch * 0.5 * np.pi/180)
    cr = np.cos(roll * 0.5 * np.pi/180)
    sr = np.sin(roll * 0.5 * np.pi/180)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    return [qw, qx, qy, qz]


def read_yaml_ego(source_dir, yaml_file):
    with open(os.path.join(source_dir, yaml_file), 'r') as file:
        data = file.read()
        content = yaml.load(data, Loader=yaml.Loader)['true_ego_pos']
        return content[:3], convert_quaternion(content[-3], content[-2], content[-1])


def ego_pose_json(record_id, scene_id, seq_len, stype, sensor_type, activated_car, source_dir, target_file, save_dir):
    with open(os.path.join(save_dir, 'ego_pose.json'), 'a') as jsonfile:
        save_list = list()
        for idx in range(activated_car):
            tmp_root = os.path.join(source_dir, listdir(source_dir)[idx])
            for time in range(seq_len):
                tmp_ego = ego_json_tmp()
                tmp_ego["token"] = generate_uuid('opv2v-'+stype+'-sampled-'+record_id+'-'+scene_id+'.'+str(time)+'.'+str(idx)+'.'+sensor_type)
                tmp_ego["timestamp"] = time
                tmp_ego["translation"], tmp_ego["rotation"] = read_yaml_ego(tmp_root, target_file[time]+'.yaml')

                save_list.append(tmp_ego)

        jsonfile.write(json.dumps(save_list, indent=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--savepath",
        default="/media/th/T7-1/dataset/opv2v",
        type=str,
        help="Directory for saving the generated data",
    )
    parser.add_argument(
        "-r",
        "--root",
        default="/media/th/T7-1/dataset/OPV2V/",
        type=str,
        help="Root path to OPV2V dataset"
    )
    parser.add_argument(
        "-t",
        "--type",
        default="train",
        type=str,
        help="train/validate/test/test_culver_city"
    )
    parser.add_argument("--frames", default=60, type=int, help="each video contains how many frames")

    args = parser.parse_args()
    clear_dir(args.savepath + "-" + args.type)
    save_dir = check_folder(args.savepath + "-" + args.type)
    source_dir = os.path.join(args.root, args.type)
    max_agents = 5

    # build the basic file struct
    for i in range(max_agents):
        check_folder(os.path.join(save_dir, 'sweeps', 'LIDAR_TOP_id_' + str(i)))

    # 这里加速
    import multiprocessing
    procs = []
    for record_id in listdir(source_dir):
        record_dir = os.path.join(source_dir, record_id)
        if len(listdir(record_dir)) <= max_agents:
            activated_car = len(listdir(record_dir))
        else:
            activated_car = max_agents
        # check frame amount
        pcd_amount = list()
        file_struct = list()
        for idx in range(activated_car):
            if idx == 0:
                account, file_struct = pcd_yaml_account(os.path.join(record_dir, listdir(record_dir)[idx]))
            else:
                account, _ = pcd_yaml_account(os.path.join(record_dir, listdir(record_dir)[idx]))
            pcd_amount.append(account)

        pcd_unique = set(pcd_amount)

        if len(pcd_unique) > 1 or (list(pcd_unique)[0] // 2 == 0):
            print("The file number is not correct, please check it!")
            break

        total_frames = int(list(pcd_unique)[0] / 2)

        if total_frames < args.frames:
            print("There are limited frames for generation of", record_id)
        else:
            scene_number = round(total_frames / args.frames)
            divider = total_frames // args.frames
            for scene_id in range(scene_number):
                if scene_id < divider:
                    begin_id = scene_id * args.frames
                    end_id = scene_id * args.frames + args.frames
                    target_file = file_struct[begin_id:end_id]
                else:
                    target_file = file_struct[-args.frames:]

                procs.append(multiprocessing.Process(target=sample_data_json,
                            args=(record_id, str(scene_id), args.frames, args.type,
                                  'lidar', activated_car, target_file, save_dir)))
                procs.append(multiprocessing.Process(target=ego_pose_json,
                            args=(record_id, str(scene_id), args.frames, args.type,
                                  'lidar', activated_car, record_dir, target_file, save_dir)))
                procs.append()

    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()


if __name__ == "__main__":
    main()







