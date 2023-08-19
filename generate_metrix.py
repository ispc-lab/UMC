import os
import numpy as np
from shapely.geometry import Polygon, MultiPoint


def main(agent_id, source_root):
    iou_threhold = 0.7 # need 0.5 and 0.7
    model_det_root = source_root + "/det" + str(agent_id) + "/"
    label_root = "./gt_cali_4_class/label" + str(agent_id) + "/"
    scene_list = os.listdir(model_det_root)

    for scene in scene_list:
        path_list = os.listdir(os.path.join(model_det_root, scene))
        path_list.sort(key=lambda x: int(x.split(".")[0]))
        # recall
        tp_0 = list()
        tp_1 = list()
        tp_2 = list()
        tp_3 = list()
        
        tp_fn_0 = list()
        tp_fn_1 = list()
        tp_fn_2 = list()
        tp_fn_3 = list()
        # frame precision, recall
        tp_fp_frame = list()
        tp_frame = list()
        tp_fn_frame = list()

        for frame_id in path_list:
            det_results = np.unique(np.load(os.path.join(model_det_root, scene, frame_id), allow_pickle=True), axis=0)
            label_results = np.load(os.path.join(label_root, scene, frame_id), allow_pickle=True).item()
            tmp_0 = 0
            tmp_1 = 0
            tmp_2 = 0
            tmp_3 = 0
            t_tp_fn_0, t_tp_fn_1, t_tp_fn_2, t_tp_fn_3 = sort_label(label_results)

            for idx in range(det_results.shape[0]):
                tmp_type = iou_check(det_results[idx], label_results, iou_threhold)

                if tmp_type == "tp_0":
                    tmp_0 += 1
                elif tmp_type == "tp_1":
                    tmp_1 += 1
                elif tmp_type == "tp_2":
                    tmp_2 += 1
                elif tmp_type == "tp_3":
                    tmp_3 += 1
                elif tmp_type == "tp_none":
                    pass

            tp_0.append(tmp_0)
            tp_1.append(tmp_1)
            tp_2.append(tmp_2)
            tp_3.append(tmp_3)
            tp_frame.append(tmp_0 + tmp_1 + tmp_2 + tmp_3)
            tp_fp_frame.append(det_results.shape[0])
            tp_fn_frame.append(len(label_results.keys()))
            tp_fn_0.append(t_tp_fn_0)
            tp_fn_1.append(t_tp_fn_1)
            tp_fn_2.append(t_tp_fn_2)
            tp_fn_3.append(t_tp_fn_3)

        save_dict = {}
        save_idx = ['tp_0', 'tp_fn_0', 'tp_1', 'tp_fn_1', 'tp_2', 'tp_fn_2', \
                    'tp_3', 'tp_fn_3', 'tp_frame', 'tp_fp_frame', 'tp_fn_frame']
        save_content = [tp_0, tp_fn_0, tp_1, tp_fn_1, tp_2, tp_fn_2, tp_3, tp_fn_3, tp_frame, tp_fp_frame, tp_fn_frame]

        for idx, item in enumerate(save_idx):
            save_dict[item] = save_content[idx]

        save_type = "./visualize_npy/" + source_root.split('/')[-2][5:] + "_" + source_root.split('/')[-1] + "_" + str(iou_threhold)
        save_dir = check_folder(os.path.join(save_type, "agent" + str(agent_id)))
        np.save(os.path.join(save_dir, scene + ".npy"), save_dict)
        print("The agent id {} scene id {}".format(agent_id, scene))


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


def sort_label(label_infor):
    type_0, type_1, type_2, type_3 = 0, 0, 0, 0

    for item in label_infor.keys():
        tmp = label_infor[item]['det_type']
        if tmp == 0:
            type_0 += 1
        elif tmp == 1:
            type_1 += 1
        elif tmp == 2:
            type_2 += 1
        elif tmp == 3:
            type_3 += 1

    return type_0, type_1, type_2, type_3


def iou_check(det_tmp, label_list, iou_threthold):
    #
    poly1 = Polygon(det_tmp[:4]).convex_hull
    iou_list = list()
    correspond_type = list()
    for tmp in label_list.keys():
        poly2 = Polygon(label_list[tmp]['location'][:4])
        union_poly = np.concatenate((det_tmp[:4], label_list[tmp]['location'][:4]))
        if not poly1.intersection(poly2):
            iou_list.append(0)
        else:
            inter_area = poly1.intersection(poly2).area
            union_area = MultiPoint(union_poly).convex_hull.area
            if inter_area == union_area:
                iou_list.append(1)
            else:
                iou_list.append(float(inter_area / union_area))

    iou_max = max(iou_list)
    if iou_max < iou_threthold:
        return "tp_none"
    else:
        index = np.argmax(iou_list)
        return "tp_" + str(label_list[str(index)]['det_type'])


def trans_file(target_dir):
    agent_results = {}
    for agent_id in os.listdir(target_dir):
        total_results = {}
        for scene_id in os.listdir(os.path.join(target_dir, agent_id)):
            tmp_file = np.load(os.path.join(target_dir, agent_id, scene_id), allow_pickle=True).item()
            save_results = {}
            for matrix in ['tp_0', 'tp_1', 'tp_2', 'tp_3', 'tp_frame']:
                if matrix == 'tp_frame':
                    tp_fp_np = np.array(tmp_file['tp_fp_frame'])
                    tp_fp_idx = np.nonzero(tp_fp_np)[0]
                    if tp_fp_idx.size != 0:
                        save_results['p_frame'] = np.mean(np.array(tmp_file[matrix])[tp_fp_idx] / tp_fp_np[tp_fp_idx])
                    else:
                        save_results['p_frame'] = 0
                    tp_fn_np = np.array(tmp_file['tp_fn_frame'])
                    tp_fn_idx = np.nonzero(tp_fn_np)[0]
                    if tp_fn_idx.size != 0:
                        save_results['r_frame'] = np.mean(np.array(tmp_file[matrix])[tp_fn_idx] / tp_fn_np[tp_fn_idx])
                    else:
                        save_results['r_frame'] = 0
                else:
                    tmp_name = matrix[:3] + "fn" + matrix[2:]
                    tp_fn_x = np.array(tmp_file[tmp_name])
                    tp_fn_x_idx = np.nonzero(tp_fn_x)[0]
                    if tp_fn_x_idx.size != 0:
                        save_results['r_' + matrix[-1]] = np.mean(
                            np.array(tmp_file[matrix])[tp_fn_x_idx] / tp_fn_x[tp_fn_x_idx])
                    else:
                        save_results['r_' + matrix[-1]] = 0

            total_results[scene_id] = save_results

        agent_results[agent_id] = total_results

    save_dir = check_folder('./trans_file/')
    np.save(os.path.join(save_dir, target_dir.split('/')[-2] + ".npy"), agent_results)


if __name__ == "__main__":

    eval_list = ['./0923_kd_off/Entropy_Multi_V2_layer2_GraphGRU_V1',
                 './0923_kd_off/Entropy_Multi_V3_layer2_GraphGRU_V1',
                 './0923_kd_off/Multi_V2_Layer2_GraphGRU_V1',
                 './0923_kd_off/Multi_V3_Layer2_GraphGRU_V1',
                 './0923_datapipeline/disco',
                 './0914_when2com/when2com',
                 './0914_v2v/v2v',
                 './0923_kd_off/NoFusion',
                 './0923_kd_off/GraphGRU_v1']

    for results in eval_list:
        import multiprocessing

        procs = []
        for i in [0, 1, 2, 3, 4]:
            procs.append(multiprocessing.Process(target=main, args=(i, results)))
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()

    generate_wether = True

    if generate_wether:
        
        tag_list = ['kd_off_Entropy_Multi_V2_layer2_GraphGRU_V1_0.7',
                    'kd_off_Entropy_Multi_V3_layer2_GraphGRU_V1_0.7',
                    'kd_off_GraphGRU_v1_0.7',
                    'kd_off_NoFusion_0.7',
                    'kd_off_Multi_V2_Layer2_GraphGRU_V1_0.7',
                    'kd_off_Multi_V3_Layer2_GraphGRU_V1_0.7',
                    'datapipeline_disco_0.7',
                    'when2com_when2com_0.7',
                    'v2v_v2v_0.7',
                    'kd_off_Entropy_Multi_V2_layer2_GraphGRU_V1_0.5',
                    'kd_off_Entropy_Multi_V3_layer2_GraphGRU_V1_0.5',
                    'kd_off_GraphGRU_v1_0.5',
                    'kd_off_NoFusion_0.5',
                    'kd_off_Multi_V2_Layer2_GraphGRU_V1_0.5',
                    'kd_off_Multi_V3_Layer2_GraphGRU_V1_0.5',
                    'datapipeline_disco_0.5',
                    'when2com_when2com_0.5',
                    'v2v_v2v_0.5']

        for item in tag_list:
            trans_file(target_dir="./visualize_npy/" + item + "/")

