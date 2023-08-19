import numpy as np
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "0" 
import torch
import torch.optim as optim
import argparse
from tqdm import tqdm
from utils.CoDetModel import *
from utils.CoDetModule import *
from utils.loss import *
from data.Dataset import V2XSIMDataset, SceneDataset, custom_collate_fn, get_frame_by_idx_from_scene_list
from data.config import Config, ConfigGlobal

# import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = "3" 

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

    num_epochs = args.nepoch
    need_log = args.log
    num_workers = args.nworker
    start_epoch = 1
    batch_size = args.batch

    # Specify gpu device
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

    trainset = SceneDataset(data_root=args.data, val=False, config=config, config_global=config_global)
    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=batch_size, collate_fn=custom_collate_fn, num_workers=num_workers, drop_last=True)
    
    print("Training dataset size:", len(trainset))
    
    logger_root = args.logpath if args.logpath != '' else 'logs'

    # import pdb; pdb.set_trace()
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
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = {'cls': SoftmaxFocalClassificationLoss(), 'loc': WeightedSmoothL1LocalizationLoss()}

    if args.kd_flag == 1:
        teacher = TeacherNet(config)
        # teacher = nn.DataParallel(teacher)
        teacher = teacher.to(device)
        fafmodule = FaFModule(model, teacher, config, optimizer, criterion, args.kd_flag)
        #checkpoint_teacher = torch.load(args.resume_teacher)
        #start_epoch_teacher = checkpoint_teacher['epoch']
        #fafmodule.teacher.load_state_dict(checkpoint_teacher['model_state_dict'])
        #print("Load teacher model from {}, at epoch {}".format(args.resume_teacher, start_epoch_teacher))
        #fafmodule.teacher.eval()
    else:
        fafmodule = FaFModule(model, model, config, optimizer, criterion, args.kd_flag)

    if args.resume == '':
        model_save_path = check_folder(logger_root)
        model_save_path = check_folder(os.path.join(model_save_path, flag))

        log_file_name = os.path.join(model_save_path, 'log.txt')
        saver = open(log_file_name, "w")
        saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
        saver.flush()

        # Logging the details for this experiment
        saver.write("command line: {}\n".format(" ".join(sys.argv[0:])))
        saver.write(args.__repr__() + "\n\n")
        saver.flush()
    else:
        model_save_path = args.resume[:args.resume.rfind('/')]

        log_file_name = os.path.join(model_save_path, 'log.txt')
        saver = open(log_file_name, "a")
        saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
        saver.flush()

        # Logging the details for this experiment
        saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
        saver.write(args.__repr__() + "\n\n")
        saver.flush()

        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        fafmodule.model.load_state_dict(checkpoint['model_state_dict'])
        fafmodule.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        fafmodule.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))

    for epoch in range(start_epoch, num_epochs + 1):
        lr = fafmodule.optimizer.param_groups[0]['lr']
        print("Epoch {}, learning rate {}".format(epoch, lr))

        if need_log:
            saver.write("epoch: {}, lr: {}\t".format(epoch, lr))
            saver.flush()

        running_loss_disp = AverageMeter('Total loss', ':.6f')
        running_loss_class = AverageMeter('classification Loss', ':.6f')  # for cell classification error
        running_loss_loc = AverageMeter('Localization Loss', ':.6f')  # for state estimation error

        fafmodule.model.train()

        t2 = tqdm(trainloader)
        for train_data in t2:
            
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

            t = tqdm(range(100))
            for frame_id in t:
                tmp_list = list([] for i in range(13))
                for agent in range(5):
                    tmp_list = stack_list(tmp_list, get_frame_by_idx_from_scene_list(train_data, agent_id=agent, idx=frame_id))
                
                trans_matrices = torch.stack(tuple(tmp_list[-4]), 1)
                target_agent_id = torch.stack(tuple(tmp_list[-6]), 1)
                num_agent = torch.stack(tuple(tmp_list[-5]), 1)
                
                roi_infor = torch.stack(tuple(tmp_list[-1]), 1)  # shape [batch*agent_num, 6, 4]
                gnss_infor = torch.stack(tuple(tmp_list[-3]), 1) # shape [batch*agent_num, 2, 3]
                imu_infor = torch.stack(tuple(tmp_list[-2]), 1)  # shape [batch*agent_num, 2, 3]
                
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
                data['labels'] = label_one_hot.to(device) # shape [agent*batch, 256, 256, 6, 2]
                data['reg_targets'] = reg_target.to(device)
                data['anchors'] = anchors_map.to(device) # shape [agent*batch, 256, 256, 6, 6]
                data['reg_loss_mask'] = reg_loss_mask.to(device).type(dtype=torch.bool) # shape [agent*batch, 256, 256, 6, 1]

                data['vis_maps'] = vis_maps.to(device)  # shape [agent*batch, 0]
                data['target_agent_ids'] = target_agent_id.to(device) # shape [batch, agent]
                data['num_agent'] = num_agent.to(device)  # shape [batch, agent]

                data['trans_matrices'] = trans_matrices # shape [batch, agent, 5, 4, 4]
                data['gnss'] = gnss_infor               
                data['imu']  = imu_infor
                data['roi']  = roi_infor
                
                if args.kd_flag == 1:
                    padded_voxel_points_teacher = torch.cat(tuple(tmp_list[1]), 0)
                    data['bev_seq_teacher'] = padded_voxel_points_teacher.to(device)
                    data['kd_weight'] = args.kd_weight
                
                loss, cls_loss, loc_loss = fafmodule.step(data, batch_size)
                # import pdb; pdb.set_trace()
                running_loss_disp.update(loss)
                running_loss_class.update(cls_loss)
                running_loss_loc.update(loc_loss)
                
                if np.isnan(loss) or np.isnan(cls_loss) or np.isnan(loc_loss):
                    print(f'Epoch {epoch}, loss is nan: {loss}, {cls_loss} {loc_loss}')
                    # import pdb; pdb.set_trace()
                    sys.exit();
                    
                t.set_description("Epoch {},     lr {}".format(epoch, lr))
                t.set_postfix(cls_loss=running_loss_class.avg, loc_loss=running_loss_loc.avg)
                
        fafmodule.scheduler.step()
            
        if need_log:
            saver.write("{}\t{}\t{}\n".format(running_loss_disp, running_loss_class, running_loss_loc))
            saver.flush()
            save_dict = {'epoch': epoch,
                        'model_state_dict': fafmodule.model.state_dict(),
                        'optimizer_state_dict': fafmodule.optimizer.state_dict(),
                        'scheduler_state_dict': fafmodule.scheduler.state_dict(),
                        'loss': running_loss_disp.avg}

            torch.save(save_dict, os.path.join(model_save_path, 'epoch_' + str(epoch) + '.pth'))
            
    if need_log:
        saver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default=None, type=str, help='The path to the preprocessed sparse BEV training data')
    parser.add_argument('--batch', default=4, type=int, help='Batch size')
    parser.add_argument('--nepoch', default=100, type=int, help='Number of epochs')
    parser.add_argument('--nworker', default=2, type=int, help='Number of workers')
    parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--log', action='store_true', help='Whether to log')
    parser.add_argument('--logpath', default='./final-2nd', help='The path to the output log file')
    parser.add_argument('--resume', default='', type=str, help='The path to the saved model that is loaded to resume training')
    parser.add_argument('--resume_teacher', default='', type=str, help='The path to the saved teacher model that is loaded to resume training')
    parser.add_argument('--layer', default=3, type=int, help='Communicate which layer in the single layer com mode')
    parser.add_argument('--warp_flag', action='store_true', help='Whether to use pose info for When2com')
    parser.add_argument('--kd_flag', default=0, type=int, help='Whether to enable distillation (only DiscNet is 1 )')
    parser.add_argument('--kd_weight', default=100000, type=int, help='KD loss weight')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id')
    parser.add_argument('--gnn_iter_times', default=3, type=int, help='Number of message passing for V2VNet')
    parser.add_argument('--visualization', default=True, help='Visualize validation result')
    parser.add_argument('--com', default='', type=str, help='UMC/MGFE_GCGRU/GCGRU/EntropyCS_GCGRU/...')
    parser.add_argument('--bound', type=str, help='The input setting: lowerbound -> single-view or upperbound -> multi-view')
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parser.parse_args()
    print(args)
    main(args)
