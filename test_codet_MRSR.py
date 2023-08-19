import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import torch
import torch.optim as optim
import argparse
from tqdm import tqdm
from utils.CoDetModel import *
from utils.CoDetModule import *
from utils.loss import *
from data.Dataset import V2XSIMDataset, SceneDataset, custom_collate_fn, get_frame_by_idx_from_scene_list
from data.config import Config, ConfigGlobal
from utils.mean_ap import eval_map

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def stack_list(tmp_list, get_frame):

    for idx, item in enumerate(zip(*get_frame)):
        if idx in [7, 8]:
            tmp_list[idx].append(item)
        else:
            tmp_list[idx].append(torch.stack(tuple(item), 0))

    return tmp_list

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

def main(args):

    seed = 622
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    config = Config('train', binary=True, only_det=True)
    config_global = ConfigGlobal('train', binary=True, only_det=True)

    need_log = args.log  
    num_workers = args.nworker
    # start_epoch = 1
    batch_size = 1

    gpu_list = [args.gpu]
    gpu_list_str = ','.join(map(str, gpu_list))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.bound == 'upperbound':
        flag = 'upperbound'
    elif args.bound == 'lowerbound':
        if args.com == 'GCGRU':
            flag = 'GCGRU'
        elif args.com == 'UMC_GrainSelection_1_3':
            flag = 'UMC_GrainSelection_1_3'
        elif args.com == 'UMC_GrainSelection_2_3':
            flag = 'UMC_GrainSelection_2_3'
        elif args.com == 'UMC':
            flag = 'UMC'
        elif args.com == 'MGFE_GCGRU':
            flag = 'MGFE_GCGRU'
        elif args.com == 'EntropyCS_GCGRU':
            flag = 'EntropyCS_GCGRU'
        else:
            flag = 'lowerbound'
    else:
        raise ValueError('not implement')

    config.flag = flag
    vallset = SceneDataset(data_root=args.data, val=True, config=config, config_global=config_global)
    valloader = torch.utils.data.DataLoader(vallset, shuffle=False, batch_size=batch_size, collate_fn=custom_collate_fn, num_workers=num_workers, drop_last=True)

    print("Testing dataset size:", len(vallset))

    logger_root = args.logpath if args.logpath != '' else 'logs'

    if args.com == '':
        model = FaFNet(config)
    elif args.com == 'UMC':
        model = UMC(config, layer=args.layer, warp_flag=args.warp_flag)
    elif args.com == 'UMC_GrainSelection_1_3':
        model = UMC_GrainSelection_1_3(config, gnn_iter_times=args.gnn_iter_times, layer=args.layer, layer_channel=256)
    elif args.com == 'UMC_GrainSelection_2_3':
        model = UMC_GrainSelection_2_3(config, layer=args.layer, kd_flag=args.kd_flag)
    elif args.com == 'MGFE_GCGRU':
        model = MGFE_GCGRU(config, layer=args.layer, kd_flag=args.kd_flag)
    elif args.com == 'GCGRU':
        model = GCGRU(config, layer=args.layer, kd_flag=args.kd_flag)
    elif args.com == 'EntropyCS_GCGRU':
        model = EntropyCS_GCGRU(config, layer=args.layer, kd_flag=args.kd_flag)
    
    # model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = {'cls':   SoftmaxFocalClassificationLoss(),
                 'loc': WeightedSmoothL1LocalizationLoss()}


    fafmodule = FaFModule(model, model, config, optimizer, criterion, args.kd_flag)

    model_save_path = args.resume[:args.resume.rfind('/')]

    log_file_name = os.path.join(model_save_path, 'log.txt')
    saver = open(log_file_name, "a")
    saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
    # Logging the details for this experiment
    saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
    saver.write(args.__repr__() + "\n\n")
    saver.flush()
    # 加载预训练模型
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch'] + 1
    
    #import pdb; pdb.set_trace()
    weight_dict = {}
    for name in checkpoint['model_state_dict'].keys():
        new_name = name.replace('module.', '') if 'module' in name else name
        weight_dict[new_name] = checkpoint['model_state_dict'][name]
    # import pdb; pdb.set_trace()
    #fafmodule.model.load_state_dict(checkpoint['model_state_dict'])
    fafmodule.model.load_state_dict(weight_dict)
    # fafmodule.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # fafmodule.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))

    #  ===== eval =====
    fafmodule.model.eval()
    save_fig_path = [check_folder(os.path.join(model_save_path, f'vis{i}')) for i in range(5)]
    tracking_path = [check_folder(os.path.join(model_save_path, f'tracking{i}')) for i in range(5)]
    label_path = [check_folder(os.path.join(model_save_path, f'det{i}')) for i in range(5)]

    # for local and global mAP evaluation
    det_results_local = [[] for i in range(5)]
    annotations_local = [[] for i in range(5)]
    tracking_file = [set()] * 5

    t2 = tqdm(valloader)
    for test_data in t2:
        
        if args.com == 'GCGRU':
            fafmodule.init(batch_size=batch_size)
        elif args.com == 'EntropyCS_GCGRU':
            fafmodule.init(batch_size=batch_size)
        elif args.com == 'UMC':
            fafmodule.init(batch_size=batch_size)
        elif args.com == 'MGFE_GCGRU':
            fafmodule.init(batch_size=batch_size)
        elif args.com == 'UMC_GrainSelection_1_3':
            fafmodule.init(batch_size=batch_size)
        elif args.com == 'UMC_GrainSelection_2_3':
            fafmodule.init(batch_size=batch_size)
            
        t = tqdm(range(100)) # 100
        for frame_id in t:
            
            tmp_list = list([] for i in range(15))
            for agent in range(5):
                tmp_list = stack_list(tmp_list, get_frame_by_idx_from_scene_list(test_data, agent_id=agent, idx=frame_id))

            start_time = time.time()
            filename0 = tmp_list[8][0]
            trans_matrices = torch.stack(tuple(tmp_list[-4]), 1)
            target_agent_id = torch.stack(tuple(tmp_list[-6]), 1)
            num_agent = torch.stack(tuple(tmp_list[-5]), 1)

            roi_infor = torch.cat(tuple(tmp_list[-1]), 0)  # shape [batch*agent_num, 6, 4]
            gnss_infor = torch.cat(tuple(tmp_list[-3]), 0) # shape [batch*agent_num, 2, 3]
            imu_infor = torch.cat(tuple(tmp_list[-2]), 0)  # shape [batch*agent_num, 2, 3]

            if flag == 'upperbound':
                padded_voxel_point = torch.cat(tuple(tmp_list[1]), 0)
            else:
                padded_voxel_point = torch.cat(tuple(tmp_list[0]), 0)

            label_one_hot = torch.cat(tuple(tmp_list[2]), 0)
            reg_target = torch.cat(tuple(tmp_list[3]), 0)
            reg_loss_mask = torch.cat(tuple(tmp_list[4]), 0)
            anchors_map = torch.cat(tuple(tmp_list[5]), 0)
            vis_maps = torch.cat(tuple(tmp_list[6]), 0)

            data = {}
            data['bev_seq'] = padded_voxel_point.to(device) # shape [agent*batch, 1, 256, 256, 13]
            data['label'] = label_one_hot.to(device) # shape [agent*batch, 256, 256, 6, 2]
            data['reg_targets'] = reg_target.to(device)
            data['anchors'] = anchors_map.to(device) # shape [agent*batch, 256, 256, 6, 6]
            data['reg_loss_mask'] = reg_loss_mask.to(device).type(dtype=torch.bool) # shape [agent*batch, 256, 256, 6, 1]

            data['vis_maps'] = vis_maps.to(device)  # shape [agent*batch, 0]
            data['target_agent_ids'] = target_agent_id.to(device) # shape [batch, agent]
            data['num_agent'] = num_agent.to(device)  # shape [batch, agent]

            data['trans_matrices'] = trans_matrices # shape [batch, agent, 5, 4, 4]
            data['gnss'] = gnss_infor
            data['imu'] = imu_infor
            data['roi'] = roi_infor

            loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, frame_id, 1)

            num_sensor = tmp_list[-5][0][0].numpy()
            for k in range(num_sensor):
                data_agents = {}
                data_agents['bev_seq'] = torch.unsqueeze(padded_voxel_point[k, :, :, :, :], 1)
                data_agents['reg_targets'] = torch.unsqueeze(reg_target[k, :, :, :, :, :], 0)
                data_agents['anchors'] = torch.unsqueeze(anchors_map[k, :, :, :, :], 0)
                data_agents['gt_max_iou'] = torch.tensor(tmp_list[7][k][0][0]['gt_box'])
                result_temp = result[k]
                # import pdb; pdb.set_trace()
                if len(result_temp) == 0:
                    continue
                # import pdb; pdb.set_trace()
                temp = {'bev_seq': data_agents['bev_seq'][0, -1].cpu().numpy(),
                        'result': result_temp[0][0],
                        'reg_targets': data_agents['reg_targets'].cpu().numpy()[0],
                        'anchors_map': data_agents['anchors'].cpu().numpy()[0],
                        'gt_max_iou': data_agents['gt_max_iou']}

                det_results_local[k], annotations_local[k] = cal_local_mAP(config, temp, det_results_local[k],
                                                                           annotations_local[k])
                filename = str(filename0[0][0])
                cut = filename[filename.rfind('agent') + 7:]
                seq_name = cut[:cut.rfind('_')]
                idx = cut[cut.rfind('_') + 1:cut.rfind('/')]
                seq_save = os.path.join(save_fig_path[k], seq_name)
                check_folder(seq_save)
                idx_save = str(idx) + '.png'
                # feat_idx_save = str(idx) + '_f' + '.png'
                from copy import deepcopy
                temp_ = deepcopy(temp)
                if args.visualization:
                    scene, frame = filename.split('/')[-2].split('_')
                    visualization(config, temp, label_path[k], scene, frame_id, os.path.join(seq_save, idx_save))
                    # == tracking == #
                    det_file = os.path.join(tracking_path[k], f'det_{scene}.txt')
                    if scene not in tracking_file[k]:
                        det_file = open(det_file, 'w')
                        tracking_file[k].add(scene)
                    else:
                        det_file = open(det_file, 'a')
                    det_corners = get_det_corners(config, temp_)
                    for ic, c in enumerate(det_corners):
                        det_file.write(','.join([
                            str(frame),
                            '-1',
                            '{:.2f}'.format(c[0]),
                            '{:.2f}'.format(c[1]),
                            '{:.2f}'.format(c[2]),
                            '{:.2f}'.format(c[3]),
                            str(result_temp[0][0][0]['score'][ic]),
                            '-1', '-1', '-1'
                        ]) + '\n')
                        det_file.flush()

                    det_file.close()

            print("Validation scene {}, at frame {}".format(seq_name, idx))
            print("Takes {} s\n".format(str(time.time() - start_time)))

    if need_log:
        saver.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default=None, type=str, help='The path to the preprocessed sparse BEV training data')
    parser.add_argument('--nworker', default=12, type=int, help='Number of workers')
    parser.add_argument('--log', action='store_true', help='Whether to log')
    parser.add_argument('--logpath', default='./results', help='The path to the output log file')
    parser.add_argument('--resume', default='', type=str, help='The path to the saved model that is loaded to resume training')
    parser.add_argument('--visualization', default=True, help='Visualize validation result')
    parser.add_argument('--com', default='', type=str, help='wisecom')
    parser.add_argument('--layer', default=3, type=int, help='Communicate which layer in the single layer com mode')
    parser.add_argument('--warp_flag', action='store_true', help='Whether to use pose info for When2com')
    parser.add_argument('--gnn_iter_times', default=3, type=int, help='Number of message passing for V2VNet')
    parser.add_argument('--kd_flag', default=0, type=int, help='Whether to enable distillation (only DiscNet is 1 )')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id')
    parser.add_argument('--inference', type=str)
    parser.add_argument('--tracking', action='store_true')
    parser.add_argument('--box_com', action='store_true')
    parser.add_argument('--bound', type=str, help='The input setting: lowerbound -> single-view or upperbound -> multi-view')
    # parser.add_argument('--gru_type', default='', type=str, help='gpu type')
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parser.parse_args()
    print(args)
    main(args)
