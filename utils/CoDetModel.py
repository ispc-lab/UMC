import torch.nn.functional as F
import torch.nn as nn
import os
import torch
from utils.model import *
import numpy as np
import copy
import torchgeometry as tgm
import random
import convolutional_rnn as convrnn


class UMC(nn.Module):
    def __init__(self, config, layer=3, in_channels=13, kd_flag=False):
        super(UMC, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)
        self.u_encoder = lidar_encoder(height_feat_size=in_channels)
        self.agent_num = 5
        self.kd_flag = kd_flag
        self.layer = layer
        self.ModulationLayer3 = ModulationLayer3(config)
        if self.layer ==3:
            self.PixelWeightedFusion = PixelWeightedFusionSoftmax(256*3)
            self.PixelWeightedFusion_v1 = PixelWeightedFusionSoftmax(128*3)
        elif self.layer ==2:
            self.PixelWeightedFusion = PixelWeightedFusionSoftmax(128)

        # Detection decoder
        self.decoder = lidar_decoder_v2(32)
        self.hidden_f_last = self.init_hidden_f(4)
        self.cnn_w_hid = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_w_hid = nn.BatchNorm2d(256)
        self.cnn_w_hid_v1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_w_hid_v1 = nn.BatchNorm2d(128)
        self.gru_r_z = GRU_R_Z(256)
        self.gru_r_z_v1 = GRU_R_Z(128)
   
        self.compress = CompressNet(256)
        self.compress_v1 = CompressNet(128)
  
        self.masker = conv_mask_uniform(256, 256, kernel_size=3, padding=1)
        self.masker_v1= conv_mask_uniform(128, 128, kernel_size=3, padding=1)

        self.stack = stack_channel(1, 9, kernel_size=3, padding=1)
        
    def init_hidden_f(self, batch_size):
        hidden_f = list()
        hidden_f_v1 = list()
        for i in range(batch_size):
            tmp_hidden = list()
            for j in range(5):
                tmp_hidden.append(torch.nn.Parameter(torch.nn.init.zeros_(torch.empty(256, 32, 32))))
            hidden_f.append(tmp_hidden)

        for k in range(batch_size):
            tmp_hidden = list()
            for m in range(5):
                tmp_hidden.append(torch.nn.Parameter(torch.nn.init.zeros_(torch.empty(128, 64, 64))))
            hidden_f_v1.append(tmp_hidden)
        
        return [hidden_f, hidden_f_v1]

    def agents2batch(self, feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat
    
    def acc_entropy_selection(self, tg_agent, nb_agent, delta1, delta2, M=3, N=3):
        
        w = nb_agent.shape[-2]
        h = nb_agent.shape[-1]
        batch_nb = nb_agent.reshape(-1, 1, 1, 1) # 32*32 x 1 x 1 x 1
        stack = self.stack(nb_agent).permute(2,3,1,0).contiguous().reshape(-1, 9, 1, 1)
        p = F.sigmoid((stack - batch_nb)).mean(dim=1).reshape(w, h)
        entropy_tmp = p * torch.log(p)
        
        with torch.no_grad():
            top_delta = torch.sort(entropy_tmp.reshape(-1), descending=True)
            self_holder = top_delta[0][int(w*h*delta1)]
        
        masker = torch.where(entropy_tmp>=self_holder)
        
        stack_tg = self.stack(tg_agent).permute(2,3,1,0).contiguous().reshape(-1, 9, 1, 1)
        p_t = F.sigmoid((stack_tg - batch_nb)).mean(dim=1).reshape(w, h)
        entropy_t = p_t * torch.log(p_t)
            
        # masker
        tmp_masker = - torch.ones_like(entropy_t)
        tmp_masker[masker] = entropy_t[masker]

        with torch.no_grad():
            top_delta2 = torch.sort(tmp_masker[tmp_masker!=-1].reshape(-1), descending=True)
            thresholds = top_delta2[0][int(w*h*delta2)]
              
        return torch.where(tmp_masker>=thresholds)
        
    def interplot_f(self, feature, masker):
               
        masker_t = torch.zeros_like(feature)
        masker_t[:, masker[0], masker[1]] = 1
        masker_f = masker_t[None, :, :, :].float()
        
        if feature.shape[0] == 256:
            inter = self.masker(feature.unsqueeze(0), masker_f)
        elif feature.shape[0] == 128:
            inter = self.masker_v1(feature.unsqueeze(0), masker_f)
        
        return torch.squeeze(inter)
        
    def forward(self, bevs, trans_matrices, num_agent_tensor, gnss, imu, batch_size=1):
        
        bevs = bevs.permute(0, 1, 4, 2, 3) # (Batch * agents, seq, z, h, w)
        x_0,x_1,x_2,x_3,x_4 = self.u_encoder(bevs)
        device = bevs.device

        if batch_size == 1:
            imu = torch.unsqueeze(imu, 0)
            gnss = torch.unsqueeze(gnss, 0)

        if self.layer ==4:
            feat_maps = x_4
            size = (1, 512, 16, 16)
        elif self.layer ==3:
            feat_maps = x_3
            size = (1, 256, 32, 32)
        elif self.layer == 2:
            feat_maps = x_2
            size = (1, 128, 64, 64)
        elif self.layer == 1:
            feat_maps = x_1
            size = (1, 64, 128, 128)
        elif self.layer == 0:
            feat_maps = x_0
            size = (1, 32, 256, 256)

        feat_map = {}
        feat_map_v1 = {}
        feat_list = []
        feat_list_v1 = []
        feat_maps_v1 = x_2

        for i in range(self.agent_num):
            feat_map[i] = torch.unsqueeze(feat_maps[batch_size * i:batch_size * (i + 1)], 1)
            feat_map_v1[i] = torch.unsqueeze(feat_maps_v1[batch_size * i:batch_size * (i + 1)], 1)
            # shape of feat_maps [batch * agents, 256c, 32w, 32h] 
            feat_list.append(feat_map[i])
            feat_list_v1.append(feat_map_v1[i])
            # import pdb; pdb.set_trace()

        local_com_mat = torch.cat(tuple(feat_list), 1) # [2 5 512 16 16] [batch, agent, channel, height, width]
        local_com_mat_update = torch.cat(tuple(feat_list), 1) # to avoid the inplace operation

        local_com_mat_v1 = torch.cat(tuple(feat_list_v1), 1)
        local_com_mat_update_v1 = torch.cat(tuple(feat_list_v1), 1)

        p = np.array([1.0, 0.0])
        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]
            for i in range(num_agent):
                tg_agent = local_com_mat[b, i] # 256x32x32
                tg_agent_v1 = local_com_mat_v1[b, i] # 128x64x64

                rot_infor = torch.unsqueeze(imu[b, i], 0).to(device)
                trans_infor = torch.unsqueeze(gnss[b, i], 0).to(device)
                hd_grid_rot = F.affine_grid(rot_infor, size=torch.Size(size))
                hd_grid_rot_v1 = F.affine_grid(rot_infor, size=torch.Size((1, 128, 64, 64)))
                hd_grid_trans = F.affine_grid(trans_infor, size=torch.Size(size))
                hd_grid_trans_v1 = F.affine_grid(trans_infor, size=torch.Size((1, 128, 64, 64)))

                hid_infor = torch.unsqueeze(self.hidden_f_last[0][b][i], 0).to(device)
                hid_infor_v1 = torch.unsqueeze(self.hidden_f_last[1][b][i], 0).to(device)

                warp_hd_rot = F.grid_sample(hid_infor, hd_grid_rot, mode='bilinear')
                warp_hd_rot_v1 = F.grid_sample(hid_infor_v1, hd_grid_rot_v1, mode='bilinear')
                warp_hd_trans = F.grid_sample(warp_hd_rot, hd_grid_trans, mode='bilinear')
                warp_hd_trans_v1 = F.grid_sample(warp_hd_rot_v1, hd_grid_trans_v1, mode='bilinear')

                z, hd_agent = self.gru_r_z(warp_hd_trans, tg_agent)
                z_v1, hd_agent_v1 = self.gru_r_z_v1(warp_hd_trans_v1, tg_agent_v1)
                # import pdb; pdb.set_trace()
                all_warp = trans_matrices[b, i]
                
                neighbor_feat_list = list()
                neighbor_feat_list.append(tg_agent)
                neighbor_feat_list_v1 = list()
                neighbor_feat_list_v1.append(tg_agent_v1)
                
                p_com_outage = np.random.choice([0, 1], p=p.ravel())
                if p_com_outage == 1:
                    # 这里还差fusion, 到时候加，不碍事
                    agent_wise_weight_feat = neighbor_feat_list[0]
                else:
                    for j in range(num_agent):
                        if j != i:
                            nb_agent = torch.unsqueeze(local_com_mat[b, j], 0)
                            nb_agent_v1 = torch.unsqueeze(local_com_mat_v1[b, j], 0)
                            
                            nb_warp = all_warp[j]
                            x_trans = (4*nb_warp[0, 3])/128
                            y_trans = -(4*nb_warp[1, 3])/128
                            theta_rot = torch.tensor([[nb_warp[0,0], nb_warp[0,1], 0.0], [nb_warp[1,0], nb_warp[1,1], 0.0]]).type(dtype=torch.float).to(device)
                            theta_rot = torch.unsqueeze(theta_rot, 0)
                            grid_rot = F.affine_grid(theta_rot, size=torch.Size(size))  # for grid sample
                            grid_rot_v1 = F.affine_grid(theta_rot, size=torch.Size((1, 128, 64, 64)))
                            theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(device)
                            theta_trans = torch.unsqueeze(theta_trans, 0)
                            grid_trans = F.affine_grid(theta_trans, size=torch.Size(size))  # for grid sample
                            grid_trans_v1 = F.affine_grid(theta_trans, size=torch.Size((1, 128, 64, 64)))

                            warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
                            warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
                            warp_feat = torch.squeeze(warp_feat_trans)
                            
                            warp_feat_rot_v1 = F.grid_sample(nb_agent_v1, grid_rot_v1, mode='bilinear')
                            warp_feat_trans_v1 = F.grid_sample(warp_feat_rot_v1, grid_trans_v1, mode='bilinear')
                            
                            warp_feat_v1 = torch.squeeze(warp_feat_trans_v1)
                            
                            if warp_feat.min() + warp_feat.max() == 0:
                                neighbor_feat_list.append(warp_feat)
                                neighbor_feat_list_v1.append(warp_feat_v1)
                            else:
                                tg_agent_com = self.compress(torch.unsqueeze(tg_agent, 0))
                                tg_agent_com_v1 = self.compress_v1(torch.unsqueeze(tg_agent_v1, 0))
                                warp_feat_trans_com = self.compress(warp_feat_trans)
                                warp_feat_trans_com_v1 = self.compress_v1(warp_feat_trans_v1)
                                selection = self.acc_entropy_selection(tg_agent_com, warp_feat_trans_com, delta1=0.1, delta2=0.1, M=3, N=3)
                                selection_v1 = self.acc_entropy_selection(tg_agent_com_v1, warp_feat_trans_com_v1, delta1=0.1, delta2=0.1, M=3, N=3)
                                warp_feat_interplot = self.interplot_f(warp_feat, selection)
                                warp_feat_interplot_v1 = self.interplot_f(warp_feat_v1, selection_v1)
                                neighbor_feat_list.append(warp_feat_interplot)
                                neighbor_feat_list_v1.append(warp_feat_interplot_v1)


                    tmp_agent_weight_list = list()
                    tmp_agent_weight_list_v1 = list()
                    sum_weight = 0
                    sum_weight_v1 = 0
                    for k in range(num_agent):
                        cat_feat = torch.cat([hd_agent, tg_agent, neighbor_feat_list[k]], dim=0)
                        cat_feat_v1 = torch.cat([hd_agent_v1, tg_agent_v1, neighbor_feat_list_v1[k]], dim=0)
                        AgentWeight = torch.squeeze(self.PixelWeightedFusion(cat_feat))
                        AgentWeight_v1 = torch.squeeze(self.PixelWeightedFusion_v1(cat_feat_v1))
                        tmp_agent_weight_list.append(torch.exp(AgentWeight))
                        tmp_agent_weight_list_v1.append(torch.exp(AgentWeight_v1))
                        sum_weight = sum_weight + torch.exp(AgentWeight)
                        sum_weight_v1 = sum_weight_v1 + torch.exp(AgentWeight_v1)

                    agent_weight_list = list()
                    agent_weight_list_v1 = list()
                    for k in range(num_agent):
                        AgentWeight = torch.div(tmp_agent_weight_list[k], sum_weight)
                        AgentWeight_v1 = torch.div(tmp_agent_weight_list_v1[k], sum_weight_v1)
                        AgentWeight.expand([256, -1, -1])
                        AgentWeight_v1.expand([128, -1, -1])
                        agent_weight_list.append(AgentWeight)
                        agent_weight_list_v1.append(AgentWeight_v1)

                    agent_wise_weight_feat = 0
                    agent_wise_weight_feat_v1 = 0
                    
                    for k in range(num_agent):
                        agent_wise_weight_feat = agent_wise_weight_feat + agent_weight_list[k]*neighbor_feat_list[k]
                        agent_wise_weight_feat_v1 = agent_wise_weight_feat_v1 + agent_weight_list_v1[k]*neighbor_feat_list_v1[k]

                # feature update and hid renew
                out_stage = torch.mul(1-z, agent_wise_weight_feat.unsqueeze(0)) + torch.mul(z, hd_agent.unsqueeze(0))
                out_stage_v1 = torch.mul(1-z_v1, agent_wise_weight_feat_v1.unsqueeze(0)) + torch.mul(z_v1, hd_agent_v1.unsqueeze(0))
                local_com_mat_update[b, i] = out_stage.squeeze(0)
                local_com_mat_update_v1[b, i] = out_stage_v1.squeeze(0)
                    
                self.hidden_f_last[0][b][i] = (F.relu(self.bn_w_hid(self.cnn_w_hid(out_stage))).squeeze(0)).detach()
                self.hidden_f_last[1][b][i] = (F.relu(self.bn_w_hid_v1(self.cnn_w_hid_v1(out_stage_v1))).squeeze(0)).detach()
        
        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents2batch(local_com_mat_update)
        feat_fuse_mat_v1 = self.agents2batch(local_com_mat_update_v1)
        # import pdb; pdb.set_trace()
        if self.kd_flag == 1:
            if self.layer ==4:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,x_2,x_3,feat_fuse_mat,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 3:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,feat_fuse_mat_v1,feat_fuse_mat,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 2:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,feat_fuse_mat,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 1:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,feat_fuse_mat,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 0:
                x_8, x_7, x_6, x_5 = self.decoder(feat_fuse_mat,x_1,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)

            x = x_8
        else:
            if self.layer ==4:
                x = self.decoder(x_0,x_1,x_2,x_3,feat_fuse_mat,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 3:
                x = self.decoder(x_0, x_1, x_2, x_3, x_4, feat_fuse_mat_v1, feat_fuse_mat, batch_size, kd_flag = self.kd_flag)
            elif self.layer == 2:
                x = self.decoder(x_0,x_1,feat_fuse_mat,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 1:
                x = self.decoder(x_0,feat_fuse_mat,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 0:
                x = self.decoder(feat_fuse_mat,x_1,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
        
        # Cell Classification head
        cls_preds = self.classification(x)
        # import pdb; pdb.set_trace()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0],-1,self.category_num)
        
        # Detection head
        loc_preds =self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1,loc_preds.size(1),loc_preds.size(2),self.anchor_num_per_loc,self.out_seq_len,self.box_code_size)
        # loc_pred (N * T * W * H * loc)
        result = {'loc': loc_preds,
                  'cls': cls_preds}

        #MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0],-1,motion_cat)
            result['state'] = motion_cls_preds

        if self.kd_flag == 1:
            return result, x_8, x_7, x_6, x_5, feat_fuse_mat
        else:
            return result

#################################### grains experiments ####################################

class UMC_GrainSelection_1_3(nn.Module):
    def __init__(self, config, layer=3, in_channels=13, kd_flag=False):
        super(UMC_GrainSelection_1_3, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)
        self.u_encoder = lidar_encoder(height_feat_size=in_channels)
        self.agent_num = 5
        self.kd_flag = kd_flag
        self.layer = layer
        self.ModulationLayer3 = ModulationLayer3(config)
        
        if self.layer ==3:
            self.PixelWeightedFusion = PixelWeightedFusionSoftmax(256*3)
            self.PixelWeightedFusion_v1 = PixelWeightedFusionSoftmax(64*3)
        elif self.layer ==2:
            self.PixelWeightedFusion = PixelWeightedFusionSoftmax(128)
            
        self.decoder = lidar_decoder_v2_1_3(32) 
        self.hidden_f_last = self.init_hidden_f(4)
        
        self.cnn_w_hid = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_w_hid = nn.BatchNorm2d(256)
        self.cnn_w_hid_v1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_w_hid_v1 = nn.BatchNorm2d(64)
        
        self.gru_r_z = GRU_R_Z(256)
        self.gru_r_z_v1 = GRU_R_Z(64)
        
        self.compress = CompressNet(256)
        self.compress_v1 = CompressNet(64)
        
        self.masker = conv_mask_uniform(256, 256, kernel_size=3, padding=1)
        self.masker_v1= conv_mask_uniform(64, 64, kernel_size=7, padding=3)
        
        self.stack = stack_channel(1, 9, kernel_size=3, padding=1)
    
    def init_hidden_f(self, batch_size):
        hidden_f = list()
        hidden_f_v1 = list()
        for i in range(batch_size):
            tmp_hidden = list()
            for j in range(5):
                tmp_hidden.append(torch.nn.Parameter(torch.nn.init.normal_(torch.empty(256, 32, 32))))
            hidden_f.append(tmp_hidden)

        for k in range(batch_size):
            tmp_hidden = list()
            for m in range(5):
                tmp_hidden.append(torch.nn.Parameter(torch.nn.init.normal_(torch.empty(64, 128, 128))))
            hidden_f_v1.append(tmp_hidden)
        
        return [hidden_f, hidden_f_v1]
    
    def agents2batch(self, feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat
    
    def acc_entropy_selection(self, tg_agent, nb_agent, M=3, N=3):

        w = nb_agent.shape[-2]
        h = nb_agent.shape[-1]
        batch_nb = nb_agent.reshape(-1, 1, 1, 1) # 32*32 x 1 x 1 x 1
        stack = self.stack(nb_agent).permute(2,3,1,0).contiguous().reshape(-1, 9, 1, 1)
        p = F.sigmoid((stack - batch_nb)).mean(dim=1).reshape(w, h)
        entropy_tmp = p * torch.log(p)
         
        self_holder = entropy_tmp.mean()
        masker = torch.where(entropy_tmp>=self_holder)
        
        stack_tg = self.stack(tg_agent).permute(2,3,1,0).contiguous().reshape(-1, 9, 1, 1)
        p_t = F.sigmoid((stack_tg - batch_nb)).mean(dim=1).reshape(w, h)
        entropy_t = p_t * torch.log(p_t)
        # masker
        tmp_masker = - torch.ones_like(entropy_t)
        tmp_masker[masker] = entropy_t[masker]
     
        thresholds = tmp_masker[tmp_masker!=-1].mean()
        
        return torch.where(tmp_masker>=thresholds)
    
    def interplot_f(self, feature, masker):
             
        masker_t = torch.zeros_like(feature)
        masker_t[:, masker[0], masker[1]] = 1
        masker_f = masker_t[None, :, :, :].float()
        
        if feature.shape[0] == 256:
            inter = self.masker(feature.unsqueeze(0), masker_f)
        elif feature.shape[0] == 64:
            inter = self.masker_v1(feature.unsqueeze(0), masker_f)
        
        return torch.squeeze(inter)
    
    def forward(self, bevs, trans_matrices, num_agent_tensor, gnss, imu, batch_size=1):
        bevs = bevs.permute(0, 1, 4, 2, 3) # (Batch * agents, seq, z, h, w)
        x_0,x_1,x_2,x_3,x_4 = self.u_encoder(bevs)
        device = bevs.device
        
        if batch_size == 1:
            imu = torch.unsqueeze(imu, 0)
            gnss = torch.unsqueeze(gnss, 0)
            
        if self.layer ==4:
            feat_maps = x_4
            size = (1, 512, 16, 16)
        elif self.layer ==3:
            feat_maps = x_3
            size = (1, 256, 32, 32)
        elif self.layer == 2:
            feat_maps = x_2
            size = (1, 128, 64, 64)
        elif self.layer == 1:
            feat_maps = x_1
            size = (1, 64, 128, 128)
        elif self.layer == 0:
            feat_maps = x_0
            size = (1, 32, 256, 256)
            
        feat_map = {}
        feat_map_v1 = {}
        feat_list = []
        feat_list_v1 = []
        feat_maps_v1 = x_1
        
        for i in range(self.agent_num):
            feat_map[i] = torch.unsqueeze(feat_maps[batch_size * i:batch_size * (i + 1)], 1)
            feat_map_v1[i] = torch.unsqueeze(feat_maps_v1[batch_size * i:batch_size * (i + 1)], 1)
            # shape of feat_maps [batch * agents, 256c, 32w, 32h] 
            feat_list.append(feat_map[i])
            feat_list_v1.append(feat_map_v1[i])
            # import pdb; pdb.set_trace()
            
        local_com_mat = torch.cat(tuple(feat_list), 1) # [2 5 512 16 16] [batch, agent, channel, height, width]
        local_com_mat_update = torch.cat(tuple(feat_list), 1) # to avoid the inplace operation
        # 处理 (128, 64, 64)
        local_com_mat_v1 = torch.cat(tuple(feat_list_v1), 1)
        local_com_mat_update_v1 = torch.cat(tuple(feat_list_v1), 1)
        
        p = np.array([1.0, 0.0])
        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]
            for i in range(num_agent):
                tg_agent = local_com_mat[b, i] # 256x32x32
                tg_agent_v1 = local_com_mat_v1[b, i] # 128x64x64

                rot_infor = torch.unsqueeze(imu[b, i], 0).to(device)
                trans_infor = torch.unsqueeze(gnss[b, i], 0).to(device)
                hd_grid_rot = F.affine_grid(rot_infor, size=torch.Size(size))
                hd_grid_rot_v1 = F.affine_grid(rot_infor, size=torch.Size((1, 64, 128, 128)))
                hd_grid_trans = F.affine_grid(trans_infor, size=torch.Size(size))
                hd_grid_trans_v1 = F.affine_grid(trans_infor, size=torch.Size((1, 64, 128, 128))) 

                hid_infor = torch.unsqueeze(self.hidden_f_last[0][b][i], 0).to(device)
                hid_infor_v1 = torch.unsqueeze(self.hidden_f_last[1][b][i], 0).to(device)

                warp_hd_rot = F.grid_sample(hid_infor, hd_grid_rot, mode='bilinear')
                warp_hd_rot_v1 = F.grid_sample(hid_infor_v1, hd_grid_rot_v1, mode='bilinear')
                warp_hd_trans = F.grid_sample(warp_hd_rot, hd_grid_trans, mode='bilinear')
                warp_hd_trans_v1 = F.grid_sample(warp_hd_rot_v1, hd_grid_trans_v1, mode='bilinear')
                
                z, hd_agent = self.gru_r_z(warp_hd_trans, tg_agent)
                z_v1, hd_agent_v1 = self.gru_r_z_v1(warp_hd_trans_v1, tg_agent_v1)
                
                all_warp = trans_matrices[b, i]
                
                neighbor_feat_list = list()
                neighbor_feat_list.append(tg_agent)
                neighbor_feat_list_v1 = list()
                neighbor_feat_list_v1.append(tg_agent_v1)

                p_com_outage = np.random.choice([0, 1], p=p.ravel())
                if p_com_outage == 1:
                    agent_wise_weight_feat = neighbor_feat_list[0]
                else:
                    for j in range(num_agent):
                        if j != i:
                            nb_agent = torch.unsqueeze(local_com_mat[b, j], 0)
                            nb_agent_v1 = torch.unsqueeze(local_com_mat_v1[b, j], 0)
                            
                            nb_warp = all_warp[j]
                            x_trans = (4*nb_warp[0, 3])/128
                            y_trans = -(4*nb_warp[1, 3])/128
                            theta_rot = torch.tensor([[nb_warp[0,0], nb_warp[0,1], 0.0], [nb_warp[1,0], nb_warp[1,1], 0.0]]).type(dtype=torch.float).to(device)
                            theta_rot = torch.unsqueeze(theta_rot, 0)
                            grid_rot = F.affine_grid(theta_rot, size=torch.Size(size))  # for grid sample
                            grid_rot_v1 = F.affine_grid(theta_rot, size=torch.Size((1, 64, 128, 128)))
                            theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(device)
                            theta_trans = torch.unsqueeze(theta_trans, 0)
                            grid_trans = F.affine_grid(theta_trans, size=torch.Size(size))  # for grid sample
                            grid_trans_v1 = F.affine_grid(theta_trans, size=torch.Size((1, 64, 128, 128)))

                            warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
                            warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
                            warp_feat = torch.squeeze(warp_feat_trans)
                            
                            warp_feat_rot_v1 = F.grid_sample(nb_agent_v1, grid_rot_v1, mode='bilinear')
                            warp_feat_trans_v1 = F.grid_sample(warp_feat_rot_v1, grid_trans_v1, mode='bilinear')
                            
                            warp_feat_v1 = torch.squeeze(warp_feat_trans_v1) 
                            
                            if warp_feat.min() + warp_feat.max() == 0:
                                neighbor_feat_list.append(warp_feat)
                                neighbor_feat_list_v1.append(warp_feat_v1)
                            else:
                                tg_agent_com = self.compress(torch.unsqueeze(tg_agent, 0))
                                tg_agent_com_v1 = self.compress_v1(torch.unsqueeze(tg_agent_v1, 0))
                                warp_feat_trans_com = self.compress(warp_feat_trans)
                                warp_feat_trans_com_v1 = self.compress_v1(warp_feat_trans_v1)
                                selection = self.acc_entropy_selection(tg_agent_com, warp_feat_trans_com, M=3, N=3)
                                selection_v1 = self.acc_entropy_selection(tg_agent_com_v1, warp_feat_trans_com_v1, M=3, N=3)
                                
                                warp_feat_interplot = self.interplot_f(warp_feat, selection)

                                warp_feat_interplot_v1 = self.interplot_f(warp_feat_v1, selection_v1)
                                neighbor_feat_list.append(warp_feat_interplot)
                                neighbor_feat_list_v1.append(warp_feat_interplot_v1)
                                
                    tmp_agent_weight_list = list()
                    tmp_agent_weight_list_v1 = list()
                    sum_weight = 0
                    sum_weight_v1 = 0
                    for k in range(num_agent):
                        cat_feat = torch.cat([hd_agent, tg_agent, neighbor_feat_list[k]], dim=0)
                        cat_feat_v1 = torch.cat([hd_agent_v1, tg_agent_v1, neighbor_feat_list_v1[k]], dim=0)
                        AgentWeight = torch.squeeze(self.PixelWeightedFusion(cat_feat))
                        AgentWeight_v1 = torch.squeeze(self.PixelWeightedFusion_v1(cat_feat_v1))
                        tmp_agent_weight_list.append(torch.exp(AgentWeight))
                        tmp_agent_weight_list_v1.append(torch.exp(AgentWeight_v1))
                        sum_weight = sum_weight + torch.exp(AgentWeight)
                        sum_weight_v1 = sum_weight_v1 + torch.exp(AgentWeight_v1)
                    
                    agent_weight_list = list()
                    agent_weight_list_v1 = list()
                    for k in range(num_agent):
                        AgentWeight = torch.div(tmp_agent_weight_list[k], sum_weight)
                        AgentWeight_v1 = torch.div(tmp_agent_weight_list_v1[k], sum_weight_v1)
                        AgentWeight.expand([256, -1, -1])
                        AgentWeight_v1.expand([64, -1, -1])
                        agent_weight_list.append(AgentWeight)
                        agent_weight_list_v1.append(AgentWeight_v1)
                        
                    agent_wise_weight_feat = 0
                    agent_wise_weight_feat_v1 = 0
                    
                    for k in range(num_agent):
                        agent_wise_weight_feat = agent_wise_weight_feat + agent_weight_list[k]*neighbor_feat_list[k]
                        agent_wise_weight_feat_v1 = agent_wise_weight_feat_v1 + agent_weight_list_v1[k]*neighbor_feat_list_v1[k]

                out_stage = torch.mul(1-z, agent_wise_weight_feat.unsqueeze(0)) + torch.mul(z, hd_agent.unsqueeze(0))
                out_stage_v1 = torch.mul(1-z_v1, agent_wise_weight_feat_v1.unsqueeze(0)) + torch.mul(z_v1, hd_agent_v1.unsqueeze(0))
                local_com_mat_update[b, i] = out_stage.squeeze(0)
                local_com_mat_update_v1[b, i] = out_stage_v1.squeeze(0)
                self.hidden_f_last[0][b][i] = (F.relu(self.bn_w_hid(self.cnn_w_hid(out_stage))).squeeze(0)).detach()
                self.hidden_f_last[1][b][i] = (F.relu(self.bn_w_hid_v1(self.cnn_w_hid_v1(out_stage_v1))).squeeze(0)).detach()
        
        feat_fuse_mat = self.agents2batch(local_com_mat_update)
        feat_fuse_mat_v1 = self.agents2batch(local_com_mat_update_v1)                              

        if self.kd_flag == 1:
            if self.layer ==4:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,x_2,x_3,feat_fuse_mat,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 3:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,feat_fuse_mat_v1,feat_fuse_mat,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 2:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,feat_fuse_mat,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 1:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,feat_fuse_mat,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 0:
                x_8, x_7, x_6, x_5 = self.decoder(feat_fuse_mat,x_1,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)

            x = x_8  
        else:
            if self.layer ==4:
                x = self.decoder(x_0,x_1,x_2,x_3,feat_fuse_mat,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 3:
                x = self.decoder(x_0, x_1, x_2, x_3, x_4, feat_fuse_mat_v1, feat_fuse_mat, batch_size, kd_flag = self.kd_flag)
            elif self.layer == 2:
                x = self.decoder(x_0,x_1,feat_fuse_mat,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 1:
                x = self.decoder(x_0,feat_fuse_mat,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 0:
                x = self.decoder(feat_fuse_mat,x_1,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)

        # Cell Classification head
        cls_preds = self.classification(x)
        # import pdb; pdb.set_trace()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0],-1,self.category_num)

        # Detection head
        loc_preds =self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1,loc_preds.size(1),loc_preds.size(2),self.anchor_num_per_loc,self.out_seq_len,self.box_code_size)
        #loc_pred (N * T * W * H * loc)
        # import pdb; pdb.set_trace()
        result = {'loc': loc_preds,
                  'cls': cls_preds}

        #MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0],-1,motion_cat)
            result['state'] = motion_cls_preds

        if self.kd_flag == 1:
            return result, x_8, x_7, x_6, x_5, feat_fuse_mat
        else:
            return result       

class UMC_GrainSelection_2_3(nn.Module):
    def __init__(self, config, layer=3, in_channels=13, kd_flag=False):
        super(UMC_GrainSelection_2_3, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)
        self.u_encoder = lidar_encoder(height_feat_size=in_channels)
        self.agent_num = 5
        self.kd_flag = kd_flag
        self.layer = layer
        self.ModulationLayer3 = ModulationLayer3(config)
        
        if self.layer ==3:
            self.PixelWeightedFusion = PixelWeightedFusionSoftmax(128*3)
            self.PixelWeightedFusion_v1 = PixelWeightedFusionSoftmax(64*3)
        elif self.layer ==2:
            self.PixelWeightedFusion = PixelWeightedFusionSoftmax(128)
            
        self.decoder = lidar_decoder_v2_2_3(32) 
        self.hidden_f_last = self.init_hidden_f(4)
        
        self.cnn_w_hid = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_w_hid = nn.BatchNorm2d(128)
        self.cnn_w_hid_v1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_w_hid_v1 = nn.BatchNorm2d(64)
        
        self.gru_r_z = GRU_R_Z(128)
        self.gru_r_z_v1 = GRU_R_Z(64)
        
        self.compress = CompressNet(128)
        self.compress_v1 = CompressNet(64)
        
        self.masker = conv_mask_uniform(128, 128, kernel_size=3, padding=1)
        self.masker_v1= conv_mask_uniform(64, 64, kernel_size=3, padding=1)
        
        self.stack = stack_channel(1, 9, kernel_size=3, padding=1)
        
    def init_hidden_f(self, batch_size):
        hidden_f = list()
        hidden_f_v1 = list()
        for i in range(batch_size):
            tmp_hidden = list()
            for j in range(5):
                tmp_hidden.append(torch.nn.Parameter(torch.nn.init.zeros_(torch.empty(128, 64, 64))))
            hidden_f.append(tmp_hidden)

        for k in range(batch_size):
            tmp_hidden = list()
            for m in range(5):
                tmp_hidden.append(torch.nn.Parameter(torch.nn.init.zeros_(torch.empty(64, 128, 128))))
            hidden_f_v1.append(tmp_hidden)
        
        return [hidden_f, hidden_f_v1]

    def agents2batch(self, feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat
    
    def acc_entropy_selection(self, tg_agent, nb_agent, M=3, N=3):
        
        w = nb_agent.shape[-2]
        h = nb_agent.shape[-1]
        batch_nb = nb_agent.reshape(-1, 1, 1, 1) # 32*32 x 1 x 1 x 1
        stack = self.stack(nb_agent).permute(2,3,1,0).contiguous().reshape(-1, 9, 1, 1)
        p = F.sigmoid((stack - batch_nb)).mean(dim=1).reshape(w, h)
        entropy_tmp = p * torch.log(p)
         
        self_holder = entropy_tmp.mean()
        masker = torch.where(entropy_tmp>=self_holder)
        
        stack_tg = self.stack(tg_agent).permute(2,3,1,0).contiguous().reshape(-1, 9, 1, 1)
        p_t = F.sigmoid((stack_tg - batch_nb)).mean(dim=1).reshape(w, h)
        entropy_t = p_t * torch.log(p_t)
        
        # masker
        tmp_masker = - torch.ones_like(entropy_t)
        tmp_masker[masker] = entropy_t[masker]
     
        thresholds = tmp_masker[tmp_masker!=-1].mean()
        
        return torch.where(tmp_masker>=thresholds)
    
    def interplot_f(self, feature, masker):
        
        masker_t = torch.zeros_like(feature)
        masker_t[:, masker[0], masker[1]] = 1
        masker_f = masker_t[None, :, :, :].float()
        
        if feature.shape[0] == 128:
            inter = self.masker(feature.unsqueeze(0), masker_f)
        elif feature.shape[0] == 64:
            inter = self.masker_v1(feature.unsqueeze(0), masker_f)
        
        return torch.squeeze(inter)
    
    def forward(self, bevs, trans_matrices, num_agent_tensor, gnss, imu, batch_size=1):
        bevs = bevs.permute(0, 1, 4, 2, 3) # (Batch * agents, seq, z, h, w)
        x_0,x_1,x_2,x_3,x_4 = self.u_encoder(bevs)
        device = bevs.device
        
        if batch_size == 1:
            imu = torch.unsqueeze(imu, 0)
            gnss = torch.unsqueeze(gnss, 0)
            
        feat_maps = x_2
        size = (1, 128, 64, 64)
        
        feat_map = {}
        feat_map_v1 = {}
        feat_list = []
        feat_list_v1 = []
        feat_maps_v1 = x_1
        
        for i in range(self.agent_num):
            feat_map[i] = torch.unsqueeze(feat_maps[batch_size * i:batch_size * (i + 1)], 1)
            feat_map_v1[i] = torch.unsqueeze(feat_maps_v1[batch_size * i:batch_size * (i + 1)], 1)
            # shape of feat_maps [batch * agents, 256c, 32w, 32h] 
            feat_list.append(feat_map[i])
            feat_list_v1.append(feat_map_v1[i])
            # import pdb; pdb.set_trace()
            
        local_com_mat = torch.cat(tuple(feat_list), 1) # [2 5 512 16 16] [batch, agent, channel, height, width]
        local_com_mat_update = torch.cat(tuple(feat_list), 1) # to avoid the inplace operation
        local_com_mat_v1 = torch.cat(tuple(feat_list_v1), 1)
        local_com_mat_update_v1 = torch.cat(tuple(feat_list_v1), 1)
        
        p = np.array([1.0, 0.0])
        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]
            for i in range(num_agent):
                tg_agent = local_com_mat[b, i] # 256x32x32
                tg_agent_v1 = local_com_mat_v1[b, i] # 128x64x64

                rot_infor = torch.unsqueeze(imu[b, i], 0).to(device)
                trans_infor = torch.unsqueeze(gnss[b, i], 0).to(device)
                hd_grid_rot = F.affine_grid(rot_infor, size=torch.Size(size))
                hd_grid_rot_v1 = F.affine_grid(rot_infor, size=torch.Size((1, 64, 128, 128)))
                hd_grid_trans = F.affine_grid(trans_infor, size=torch.Size(size))
                hd_grid_trans_v1 = F.affine_grid(trans_infor, size=torch.Size((1, 64, 128, 128))) 

                hid_infor = torch.unsqueeze(self.hidden_f_last[0][b][i], 0).to(device)
                hid_infor_v1 = torch.unsqueeze(self.hidden_f_last[1][b][i], 0).to(device)

                warp_hd_rot = F.grid_sample(hid_infor, hd_grid_rot, mode='bilinear')
                warp_hd_rot_v1 = F.grid_sample(hid_infor_v1, hd_grid_rot_v1, mode='bilinear')
                warp_hd_trans = F.grid_sample(warp_hd_rot, hd_grid_trans, mode='bilinear')
                warp_hd_trans_v1 = F.grid_sample(warp_hd_rot_v1, hd_grid_trans_v1, mode='bilinear')
                
                z, hd_agent = self.gru_r_z(warp_hd_trans, tg_agent)
                z_v1, hd_agent_v1 = self.gru_r_z_v1(warp_hd_trans_v1, tg_agent_v1)
                
                all_warp = trans_matrices[b, i]
                
                neighbor_feat_list = list()
                neighbor_feat_list.append(tg_agent)
                neighbor_feat_list_v1 = list()
                neighbor_feat_list_v1.append(tg_agent_v1)

                p_com_outage = np.random.choice([0, 1], p=p.ravel())
                if p_com_outage == 1:
                    agent_wise_weight_feat = neighbor_feat_list[0]
                else:
                    for j in range(num_agent):
                        if j != i:
                            nb_agent = torch.unsqueeze(local_com_mat[b, j], 0)
                            nb_agent_v1 = torch.unsqueeze(local_com_mat_v1[b, j], 0)
                            
                            nb_warp = all_warp[j]
                            x_trans = (4*nb_warp[0, 3])/128
                            y_trans = -(4*nb_warp[1, 3])/128
                            theta_rot = torch.tensor([[nb_warp[0,0], nb_warp[0,1], 0.0], [nb_warp[1,0], nb_warp[1,1], 0.0]]).type(dtype=torch.float).to(device)
                            theta_rot = torch.unsqueeze(theta_rot, 0)
                            grid_rot = F.affine_grid(theta_rot, size=torch.Size(size))  # for grid sample
                            grid_rot_v1 = F.affine_grid(theta_rot, size=torch.Size((1, 64, 128, 128)))
                            theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(device)
                            theta_trans = torch.unsqueeze(theta_trans, 0)
                            grid_trans = F.affine_grid(theta_trans, size=torch.Size(size))  # for grid sample
                            grid_trans_v1 = F.affine_grid(theta_trans, size=torch.Size((1, 64, 128, 128)))

                            warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
                            warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
                            warp_feat = torch.squeeze(warp_feat_trans)
                            
                            warp_feat_rot_v1 = F.grid_sample(nb_agent_v1, grid_rot_v1, mode='bilinear')
                            warp_feat_trans_v1 = F.grid_sample(warp_feat_rot_v1, grid_trans_v1, mode='bilinear')
                            
                            warp_feat_v1 = torch.squeeze(warp_feat_trans_v1) 
                            
                            if warp_feat.min() + warp_feat.max() == 0:
                                neighbor_feat_list.append(warp_feat)
                                neighbor_feat_list_v1.append(warp_feat_v1)
                            else:
                                tg_agent_com = self.compress(torch.unsqueeze(tg_agent, 0))
                                tg_agent_com_v1 = self.compress_v1(torch.unsqueeze(tg_agent_v1, 0))
                                warp_feat_trans_com = self.compress(warp_feat_trans)
                                warp_feat_trans_com_v1 = self.compress_v1(warp_feat_trans_v1)
                                selection = self.acc_entropy_selection(tg_agent_com, warp_feat_trans_com, M=3, N=3)
                                selection_v1 = self.acc_entropy_selection(tg_agent_com_v1, warp_feat_trans_com_v1, M=3, N=3)
                                
                                warp_feat_interplot = self.interplot_f(warp_feat, selection)

                                warp_feat_interplot_v1 = self.interplot_f(warp_feat_v1, selection_v1)
                                neighbor_feat_list.append(warp_feat_interplot)
                                neighbor_feat_list_v1.append(warp_feat_interplot_v1)
                                
                    tmp_agent_weight_list = list()
                    tmp_agent_weight_list_v1 = list()
                    sum_weight = 0
                    sum_weight_v1 = 0
                    for k in range(num_agent):
                        cat_feat = torch.cat([hd_agent, tg_agent, neighbor_feat_list[k]], dim=0)
                        cat_feat_v1 = torch.cat([hd_agent_v1, tg_agent_v1, neighbor_feat_list_v1[k]], dim=0)
                        AgentWeight = torch.squeeze(self.PixelWeightedFusion(cat_feat))
                        AgentWeight_v1 = torch.squeeze(self.PixelWeightedFusion_v1(cat_feat_v1))
                        tmp_agent_weight_list.append(torch.exp(AgentWeight))
                        tmp_agent_weight_list_v1.append(torch.exp(AgentWeight_v1))
                        sum_weight = sum_weight + torch.exp(AgentWeight)
                        sum_weight_v1 = sum_weight_v1 + torch.exp(AgentWeight_v1)
                    
                    agent_weight_list = list()
                    agent_weight_list_v1 = list()
                    for k in range(num_agent):
                        AgentWeight = torch.div(tmp_agent_weight_list[k], sum_weight)
                        AgentWeight_v1 = torch.div(tmp_agent_weight_list_v1[k], sum_weight_v1)
                        AgentWeight.expand([256, -1, -1])
                        AgentWeight_v1.expand([64, -1, -1])
                        agent_weight_list.append(AgentWeight)
                        agent_weight_list_v1.append(AgentWeight_v1)
                        
                    agent_wise_weight_feat = 0
                    agent_wise_weight_feat_v1 = 0
                    
                    for k in range(num_agent):
                        agent_wise_weight_feat = agent_wise_weight_feat + agent_weight_list[k]*neighbor_feat_list[k]
                        agent_wise_weight_feat_v1 = agent_wise_weight_feat_v1 + agent_weight_list_v1[k]*neighbor_feat_list_v1[k]

                out_stage = torch.mul(1-z, agent_wise_weight_feat.unsqueeze(0)) + torch.mul(z, hd_agent.unsqueeze(0))
                out_stage_v1 = torch.mul(1-z_v1, agent_wise_weight_feat_v1.unsqueeze(0)) + torch.mul(z_v1, hd_agent_v1.unsqueeze(0))
                local_com_mat_update[b, i] = out_stage.squeeze(0)
                local_com_mat_update_v1[b, i] = out_stage_v1.squeeze(0)
                self.hidden_f_last[0][b][i] = (F.relu(self.bn_w_hid(self.cnn_w_hid(out_stage))).squeeze(0)).detach()
                self.hidden_f_last[1][b][i] = (F.relu(self.bn_w_hid_v1(self.cnn_w_hid_v1(out_stage_v1))).squeeze(0)).detach()
        
        feat_fuse_mat = self.agents2batch(local_com_mat_update)
        feat_fuse_mat_v1 = self.agents2batch(local_com_mat_update_v1)                              

        if self.kd_flag == 1:
            if self.layer ==4:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,x_2,x_3,feat_fuse_mat,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 3:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,feat_fuse_mat_v1,feat_fuse_mat,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 2:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,feat_fuse_mat,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 1:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,feat_fuse_mat,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 0:
                x_8, x_7, x_6, x_5 = self.decoder(feat_fuse_mat,x_1,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)

            x = x_8  
        else:
            if self.layer ==4:
                x = self.decoder(x_0,x_1,x_2,x_3,feat_fuse_mat,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 3:
                x = self.decoder(x_0, x_1, x_2, x_3, x_4, feat_fuse_mat_v1, feat_fuse_mat, batch_size, kd_flag = self.kd_flag)
            elif self.layer == 2:
                x = self.decoder(x_0,x_1,feat_fuse_mat,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 1:
                x = self.decoder(x_0,feat_fuse_mat,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 0:
                x = self.decoder(feat_fuse_mat,x_1,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)

        # Cell Classification head
        cls_preds = self.classification(x)
        # import pdb; pdb.set_trace()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0],-1,self.category_num)

        # Detection head
        loc_preds =self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1,loc_preds.size(1),loc_preds.size(2),self.anchor_num_per_loc,self.out_seq_len,self.box_code_size)
        #loc_pred (N * T * W * H * loc)
        # import pdb; pdb.set_trace()
        result = {'loc': loc_preds,
                  'cls': cls_preds}

        #MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0],-1,motion_cat)
            result['state'] = motion_cls_preds

        if self.kd_flag == 1:
            return result, x_8, x_7, x_6, x_5, feat_fuse_mat
        else:
            return result        

#################################### ablation experiments ####################################

class MGFE_GCGRU(nn.Module):
    def __init__(self, config, layer=3, in_channels=13, kd_flag=False):
        super(MGFE_GCGRU, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)
        self.u_encoder = lidar_encoder(height_feat_size=in_channels)
        self.agent_num = 5
        self.kd_flag = kd_flag
        self.layer = layer
        self.ModulationLayer3 = ModulationLayer3(config)
        if self.layer ==3:
            self.PixelWeightedFusion = PixelWeightedFusionSoftmax(256*3)
            self.PixelWeightedFusion_v1 = PixelWeightedFusionSoftmax(128*3)
        elif self.layer ==2:
            self.PixelWeightedFusion = PixelWeightedFusionSoftmax(128)

        # Detection decoder
        self.decoder = lidar_decoder_v2(32)
        self.hidden_f_last = self.init_hidden_f(4)
        self.cnn_w_hid = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_w_hid = nn.BatchNorm2d(256)
        self.cnn_w_hid_v1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_w_hid_v1 = nn.BatchNorm2d(128)
        self.gru_r_z = GRU_R_Z(256)
        self.gru_r_z_v1 = GRU_R_Z(128)


    def init_hidden_f(self, batch_size):
        hidden_f = list()
        hidden_f_v1 = list()
        for i in range(batch_size):
            tmp_hidden = list()
            for j in range(5):
                tmp_hidden.append(torch.nn.Parameter(torch.nn.init.zeros_(torch.empty(256, 32, 32))))
            hidden_f.append(tmp_hidden)

        for k in range(batch_size):
            tmp_hidden = list()
            for m in range(5):
                tmp_hidden.append(torch.nn.Parameter(torch.nn.init.zeros_(torch.empty(128, 64, 64))))
            hidden_f_v1.append(tmp_hidden)
        
        return [hidden_f, hidden_f_v1]

    def agents2batch(self, feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat

    def forward(self, bevs, trans_matrices, num_agent_tensor, gnss, imu, batch_size=1):
     
        bevs = bevs.permute(0, 1, 4, 2, 3) # (Batch * agents, seq, z, h, w)
        x_0,x_1,x_2,x_3,x_4 = self.u_encoder(bevs)
        device = bevs.device

        if batch_size == 1:
            imu = torch.unsqueeze(imu, 0)
            gnss = torch.unsqueeze(gnss, 0)

        if self.layer ==4:
            feat_maps = x_4
            size = (1, 512, 16, 16)
        elif self.layer ==3:
            feat_maps = x_3
            size = (1, 256, 32, 32)
        elif self.layer == 2:
            feat_maps = x_2
            size = (1, 128, 64, 64)
        elif self.layer == 1:
            feat_maps = x_1
            size = (1, 64, 128, 128)
        elif self.layer == 0:
            feat_maps = x_0
            size = (1, 32, 256, 256)

        feat_map = {}
        feat_map_v1 = {}
        feat_list = []
        feat_list_v1 = []
        feat_maps_v1 = x_2

        for i in range(self.agent_num):
            feat_map[i] = torch.unsqueeze(feat_maps[batch_size * i:batch_size * (i + 1)], 1)
            feat_map_v1[i] = torch.unsqueeze(feat_maps_v1[batch_size * i:batch_size * (i + 1)], 1)
            # shape of feat_maps [batch * agents, 256c, 32w, 32h] 
            feat_list.append(feat_map[i])
            feat_list_v1.append(feat_map_v1[i])
            # import pdb; pdb.set_trace()

        # (256, 32, 32) 
        local_com_mat = torch.cat(tuple(feat_list), 1) # [2 5 512 16 16] [batch, agent, channel, height, width]
        local_com_mat_update = torch.cat(tuple(feat_list), 1) # to avoid the inplace operation
        # (128, 64, 64)
        local_com_mat_v1 = torch.cat(tuple(feat_list_v1), 1)
        local_com_mat_update_v1 = torch.cat(tuple(feat_list_v1), 1)

        p = np.array([1.0, 0.0])
        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]
            for i in range(num_agent):
                tg_agent = local_com_mat[b, i] # 256x32x32
                tg_agent_v1 = local_com_mat_v1[b, i] # 128x64x64

                rot_infor = torch.unsqueeze(imu[b, i], 0).to(device)
                trans_infor = torch.unsqueeze(gnss[b, i], 0).to(device)
                hd_grid_rot = F.affine_grid(rot_infor, size=torch.Size(size))
                hd_grid_rot_v1 = F.affine_grid(rot_infor, size=torch.Size((1, 128, 64, 64)))
                hd_grid_trans = F.affine_grid(trans_infor, size=torch.Size(size))
                hd_grid_trans_v1 = F.affine_grid(trans_infor, size=torch.Size((1, 128, 64, 64)))

                hid_infor = torch.unsqueeze(self.hidden_f_last[0][b][i], 0).to(device)
                hid_infor_v1 = torch.unsqueeze(self.hidden_f_last[1][b][i], 0).to(device)

                warp_hd_rot = F.grid_sample(hid_infor, hd_grid_rot, mode='bilinear')
                warp_hd_rot_v1 = F.grid_sample(hid_infor_v1, hd_grid_rot_v1, mode='bilinear')
                warp_hd_trans = F.grid_sample(warp_hd_rot, hd_grid_trans, mode='bilinear')
                warp_hd_trans_v1 = F.grid_sample(warp_hd_rot_v1, hd_grid_trans_v1, mode='bilinear')

                z, hd_agent = self.gru_r_z(warp_hd_trans, tg_agent)
                z_v1, hd_agent_v1 = self.gru_r_z_v1(warp_hd_trans_v1, tg_agent_v1)
                
                all_warp = trans_matrices[b, i]
                
                neighbor_feat_list = list()
                neighbor_feat_list.append(tg_agent)
                neighbor_feat_list_v1 = list()
                neighbor_feat_list_v1.append(tg_agent_v1)

                p_com_outage = np.random.choice([0, 1], p=p.ravel())
                if p_com_outage == 1:
                    agent_wise_weight_feat = neighbor_feat_list[0]
                else:
                    for j in range(num_agent):
                        if j != i:
                            nb_agent = torch.unsqueeze(local_com_mat[b, j], 0)
                            nb_agent_v1 = torch.unsqueeze(local_com_mat_v1[b, j], 0)
                            
                            nb_warp = all_warp[j]
                            x_trans = (4*nb_warp[0, 3])/128
                            y_trans = -(4*nb_warp[1, 3])/128
                            theta_rot = torch.tensor([[nb_warp[0,0], nb_warp[0,1], 0.0], [nb_warp[1,0], nb_warp[1,1], 0.0]]).type(dtype=torch.float).to(device)
                            theta_rot = torch.unsqueeze(theta_rot, 0)
                            grid_rot = F.affine_grid(theta_rot, size=torch.Size(size))  # for grid sample
                            grid_rot_v1 = F.affine_grid(theta_rot, size=torch.Size((1, 128, 64, 64)))
                            theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(device)
                            theta_trans = torch.unsqueeze(theta_trans, 0)
                            grid_trans = F.affine_grid(theta_trans, size=torch.Size(size))  # for grid sample
                            grid_trans_v1 = F.affine_grid(theta_trans, size=torch.Size((1, 128, 64, 64)))

                            warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
                            warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
                            warp_feat = torch.squeeze(warp_feat_trans)
                            warp_feat_rot_v1 = F.grid_sample(nb_agent_v1, grid_rot_v1, mode='bilinear')
                            warp_feat_trans_v1 = F.grid_sample(warp_feat_rot_v1, grid_trans_v1, mode='bilinear')
                            warp_feat_v1 = torch.squeeze(warp_feat_trans_v1)

                            neighbor_feat_list.append(warp_feat)
                            neighbor_feat_list_v1.append(warp_feat_v1)

                    tmp_agent_weight_list = list()
                    tmp_agent_weight_list_v1 = list()
                    sum_weight = 0
                    sum_weight_v1 = 0
                    for k in range(num_agent):
                        cat_feat = torch.cat([hd_agent, tg_agent, neighbor_feat_list[k]], dim=0)
                        cat_feat_v1 = torch.cat([hd_agent_v1, tg_agent_v1, neighbor_feat_list_v1[k]], dim=0)
                        AgentWeight = torch.squeeze(self.PixelWeightedFusion(cat_feat))
                        AgentWeight_v1 = torch.squeeze(self.PixelWeightedFusion_v1(cat_feat_v1))
                        tmp_agent_weight_list.append(torch.exp(AgentWeight))
                        tmp_agent_weight_list_v1.append(torch.exp(AgentWeight_v1))
                        sum_weight = sum_weight + torch.exp(AgentWeight)
                        sum_weight_v1 = sum_weight_v1 + torch.exp(AgentWeight_v1)

                    agent_weight_list = list()
                    agent_weight_list_v1 = list()
                    for k in range(num_agent):
                        AgentWeight = torch.div(tmp_agent_weight_list[k], sum_weight)
                        AgentWeight_v1 = torch.div(tmp_agent_weight_list_v1[k], sum_weight_v1)
                        AgentWeight.expand([256, -1, -1])
                        AgentWeight_v1.expand([128, -1, -1])
                        agent_weight_list.append(AgentWeight)
                        agent_weight_list_v1.append(AgentWeight_v1)

                    agent_wise_weight_feat = 0
                    agent_wise_weight_feat_v1 = 0
                    
                    for k in range(num_agent):
                        agent_wise_weight_feat = agent_wise_weight_feat + agent_weight_list[k]*neighbor_feat_list[k]
                        agent_wise_weight_feat_v1 = agent_wise_weight_feat_v1 + agent_weight_list_v1[k]*neighbor_feat_list_v1[k]

                # feature update and hid renew
                out_stage = torch.mul(1-z, agent_wise_weight_feat.unsqueeze(0)) + torch.mul(z, hd_agent.unsqueeze(0))
                out_stage_v1 = torch.mul(1-z_v1, agent_wise_weight_feat_v1.unsqueeze(0)) + torch.mul(z_v1, hd_agent_v1.unsqueeze(0))
                local_com_mat_update[b, i] = out_stage.squeeze(0)
                local_com_mat_update_v1[b, i] = out_stage_v1.squeeze(0)
                self.hidden_f_last[0][b][i] = (F.relu(self.bn_w_hid(self.cnn_w_hid(out_stage))).squeeze(0)).detach()
                self.hidden_f_last[1][b][i] = (F.relu(self.bn_w_hid_v1(self.cnn_w_hid_v1(out_stage_v1))).squeeze(0)).detach()
        
        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents2batch(local_com_mat_update)
        feat_fuse_mat_v1 = self.agents2batch(local_com_mat_update_v1)

        if self.kd_flag == 1:
            if self.layer ==4:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,x_2,x_3,feat_fuse_mat,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 3:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,feat_fuse_mat_v1,feat_fuse_mat,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 2:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,feat_fuse_mat,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 1:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,feat_fuse_mat,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 0:
                x_8, x_7, x_6, x_5 = self.decoder(feat_fuse_mat,x_1,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)

            x = x_8
        else:
            if self.layer ==4:
                x = self.decoder(x_0,x_1,x_2,x_3,feat_fuse_mat,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 3:
                x = self.decoder(x_0, x_1, x_2, x_3, x_4, feat_fuse_mat_v1, feat_fuse_mat, batch_size, kd_flag = self.kd_flag)
            elif self.layer == 2:
                x = self.decoder(x_0,x_1,feat_fuse_mat,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 1:
                x = self.decoder(x_0,feat_fuse_mat,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 0:
                x = self.decoder(feat_fuse_mat,x_1,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)

        # Cell Classification head
        cls_preds = self.classification(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0],-1,self.category_num)

        # Detection head
        loc_preds = self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1,loc_preds.size(1),loc_preds.size(2),self.anchor_num_per_loc,self.out_seq_len,self.box_code_size)
        # loc_pred (N * T * W * H * loc)
        result = {'loc': loc_preds,
                  'cls': cls_preds}

        #MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0],-1,motion_cat)
            result['state'] = motion_cls_preds

        if self.kd_flag == 1:
            return result, x_8, x_7, x_6, x_5, feat_fuse_mat
        else:
            return result  

class GCGRU(nn.Module):
    def __init__(self, config, layer=3, in_channels=13, kd_flag=False):
        super(GCGRU, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)
        self.u_encoder = lidar_encoder(height_feat_size=in_channels)
        self.agent_num = 5
        self.kd_flag = kd_flag
        self.layer = layer
        self.ModulationLayer3 = ModulationLayer3(config)
        if self.layer ==3:
            self.PixelWeightedFusion = PixelWeightedFusionSoftmax(256*3)
        elif self.layer ==2:
            self.PixelWeightedFusion = PixelWeightedFusionSoftmax(128)

        # Detection decoder
        self.decoder = lidar_decoder(height_feat_size=in_channels)
        self.hidden_f_last = self.init_hidden_f(4)
        self.cnn_w_ir = nn.Conv2d(256*2, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.cnn_w_hz = nn.Conv2d(256*2, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_w_ir = nn.BatchNorm2d(256)
        self.bn_w_hz = nn.BatchNorm2d(256)
        self.cnn_w_hid = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_w_hid = nn.BatchNorm2d(256)


    def init_hidden_f(self, batch_size):
        hidden_f = list()
        for i in range(batch_size):
            tmp_hidden = list()
            for j in range(5):
                tmp_hidden.append(torch.nn.Parameter(torch.nn.init.zeros_(torch.empty(256, 32, 32))))
            hidden_f.append(tmp_hidden)
        return hidden_f

    def agents2batch(self, feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat

    def forward(self, bevs, trans_matrices, num_agent_tensor, gnss, imu, batch_size=1):

        if batch_size == 1:
            imu = torch.unsqueeze(imu, 0)
            gnss = torch.unsqueeze(gnss, 0)
        
        bevs = bevs.permute(0, 1, 4, 2, 3) # (Batch * agents, seq, z, h, w)
        x_0,x_1,x_2,x_3,x_4 = self.u_encoder(bevs)
        device = bevs.device

        if self.layer ==4:
            feat_maps = x_4
            size = (1, 512, 16, 16)
        elif self.layer ==3:
            feat_maps = x_3
            size = (1, 256, 32, 32)
        elif self.layer == 2:
            feat_maps = x_2
            size = (1, 128, 64, 64)
        elif self.layer == 1:
            feat_maps = x_1
            size = (1, 64, 128, 128)
        elif self.layer == 0:
            feat_maps = x_0
            size = (1, 32, 256, 256)

        feat_map = {}
        feat_list = []

        for i in range(self.agent_num):
            feat_map[i] = torch.unsqueeze(feat_maps[batch_size * i:batch_size * (i + 1)], 1)
            # shape of feat_maps [batch * agents, 256c, 32w, 32h] 
            feat_list.append(feat_map[i])
            # import pdb; pdb.set_trace()

        local_com_mat = torch.cat(tuple(feat_list), 1) # [2 5 512 16 16] [batch, agent, channel, height, width]
        local_com_mat_update = torch.cat(tuple(feat_list), 1) # to avoid the inplace operation

        save_agent_weight_list = list()
        p = np.array([1.0, 0.0])

        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]
            for i in range(num_agent):
                tg_agent = local_com_mat[b, i]

                hid_infor = torch.unsqueeze(self.hidden_f_last[b][i], 0).to(device)
                rot_infor = torch.unsqueeze(imu[b, i], 0).to(device)
                hd_grid_rot = F.affine_grid(rot_infor, size=torch.Size(size))
                trans_infor = torch.unsqueeze(gnss[b, i], 0).to(device)
                hd_grid_trans = F.affine_grid(trans_infor, size=torch.Size(size))

                warp_hd_rot = F.grid_sample(hid_infor, hd_grid_rot, mode='bilinear')
                warp_hd_trans = F.grid_sample(warp_hd_rot, hd_grid_trans, mode='bilinear')
                
                concat_f = torch.cat((tg_agent.unsqueeze(0), warp_hd_trans), dim=1)
                w_ir = torch.sigmoid(self.bn_w_ir(self.cnn_w_ir(concat_f)))
                w_hr = 1 - w_ir
                r = torch.sigmoid(torch.mul(w_ir, warp_hd_trans) + torch.mul(w_hr, tg_agent.unsqueeze(0)))
                w_hz = torch.sigmoid(self.bn_w_hz(self.cnn_w_hz(concat_f)))
                w_iz = 1 - w_hz
                z = torch.sigmoid(torch.mul(w_iz, warp_hd_trans) + torch.mul(w_hz, tg_agent.unsqueeze(0)))
                # import pdb; pdb.set_trace()
                hd_agent = torch.squeeze(torch.mul(r, warp_hd_trans))
                # print(hd_agent)
                all_warp = trans_matrices[b, i]

                neighbor_feat_list = list()
                neighbor_feat_list.append(tg_agent)

                p_com_outage = np.random.choice([0, 1], p=p.ravel())
                if p_com_outage==1:
                    agent_wise_weight_feat = neighbor_feat_list[0]
                else:
                    for j in range(num_agent):
                        if j != i:
                            nb_agent = torch.unsqueeze(local_com_mat[b, j], 0)
                            nb_warp = all_warp[j] # [4 4]  corresponding trans_matrices
                            # normalize the translation vector
                            x_trans = (4*nb_warp[0, 3])/128
                            y_trans = -(4*nb_warp[1, 3])/128

                            theta_rot = torch.tensor([[nb_warp[0,0], nb_warp[0,1], 0.0], [nb_warp[1,0], nb_warp[1,1], 0.0]]).type(dtype=torch.float).to(device)
                            theta_rot = torch.unsqueeze(theta_rot, 0)
                            grid_rot = F.affine_grid(theta_rot, size=torch.Size(size))  # for grid sample

                            theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(device)
                            theta_trans = torch.unsqueeze(theta_trans, 0)
                            grid_trans = F.affine_grid(theta_trans, size=torch.Size(size))  # for grid sample

                            # first rotate the feature map, then translate it
                            warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
                            warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
                            warp_feat = torch.squeeze(warp_feat_trans)
                            neighbor_feat_list.append(warp_feat)

                    tmp_agent_weight_list =list()
                    sum_weight = 0
                    for k in range(num_agent):
                        cat_feat = torch.cat([hd_agent, tg_agent, neighbor_feat_list[k]], dim=0)
                        cat_feat = cat_feat.unsqueeze(0)
                        AgentWeight = torch.squeeze(self.PixelWeightedFusion(cat_feat))
                        tmp_agent_weight_list.append(torch.exp(AgentWeight))
                        sum_weight = sum_weight + torch.exp(AgentWeight)
                    
                    agent_weight_list = list()
                    for k in range(num_agent):
                        AgentWeight = torch.div(tmp_agent_weight_list[k], sum_weight)
                        AgentWeight.expand([256, -1, -1])
                        agent_weight_list.append(AgentWeight)
                    
                    agent_wise_weight_feat = 0

                    for k in range(num_agent):
                        agent_wise_weight_feat = agent_wise_weight_feat + agent_weight_list[k]*neighbor_feat_list[k]

                # feature update and hid renew
                out_stage = torch.mul(1-z, agent_wise_weight_feat.unsqueeze(0)) + torch.mul(z, hd_agent.unsqueeze(0))
                # re_m, next_hid = self.grucell(out_stage)
                local_com_mat_update[b, i] = out_stage.squeeze(0)
                # self.hidden_f_last[b][i] = (F.relu(self.bn_w_hid(self.cnn_w_hid(out_stage))).squeeze(0)).detach()
                self.hidden_f_last[b][i] = (out_stage).squeeze(0).detach()

        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents2batch(local_com_mat_update)

        if self.kd_flag == 1:
            if self.layer ==4:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,x_2,x_3,feat_fuse_mat,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 3:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,x_2,feat_fuse_mat,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 2:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,feat_fuse_mat,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 1:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,feat_fuse_mat,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 0:
                x_8, x_7, x_6, x_5 = self.decoder(feat_fuse_mat,x_1,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)

            x = x_8
        else:
            if self.layer ==4:
                x = self.decoder(x_0,x_1,x_2,x_3,feat_fuse_mat,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 3:
                x = self.decoder(x_0,x_1,x_2,feat_fuse_mat,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 2:
                x = self.decoder(x_0,x_1,feat_fuse_mat,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 1:
                x = self.decoder(x_0,feat_fuse_mat,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 0:
                x = self.decoder(feat_fuse_mat,x_1,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)

        # Cell Classification head
        cls_preds = self.classification(x)
        # import pdb; pdb.set_trace()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0],-1,self.category_num)

        # Detection head
        loc_preds = self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1,loc_preds.size(1),loc_preds.size(2),self.anchor_num_per_loc,self.out_seq_len,self.box_code_size)
        #loc_pred (N * T * W * H * loc)
        result = {'loc': loc_preds,
                  'cls': cls_preds}

        #MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0],-1,motion_cat)
            result['state'] = motion_cls_preds

        if self.kd_flag == 1:
            return result, x_8, x_7, x_6, x_5, feat_fuse_mat
        else:
            return result

class EntropyCS_GCGRU(nn.Module):
    def __init__(self, config, layer=3, in_channels=13, kd_flag=False):
        super(EntropyCS_GCGRU, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)
        self.u_encoder = lidar_encoder(height_feat_size=in_channels)
        self.agent_num = 5
        self.kd_flag = kd_flag
        self.layer = layer
        self.ModulationLayer3 = ModulationLayer3(config)
        if self.layer ==3:
            self.PixelWeightedFusion = PixelWeightedFusionSoftmax(256*3)
        elif self.layer ==2:
            self.PixelWeightedFusion = PixelWeightedFusionSoftmax(128)

        # Detection decoder
        self.decoder = lidar_decoder(height_feat_size=in_channels)
        self.hidden_f_last = self.init_hidden_f(4)

        self.cnn_w_hid = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_w_hid = nn.BatchNorm2d(256)
        self.gru_r_z = GRU_R_Z(256)
        self.compress = CompressNet(256)
        self.masker = conv_mask_uniform(256, 256, kernel_size=3, padding=1)
        self.stack = stack_channel(1, 9, kernel_size=3, padding=1)

    def init_hidden_f(self, batch_size):
        hidden_f = list()
        for i in range(batch_size):
            tmp_hidden = list()
            for j in range(5):
                tmp_hidden.append(torch.nn.Parameter(torch.nn.init.zeros_(torch.empty(256, 32, 32))))
            hidden_f.append(tmp_hidden)
        return hidden_f

    def agents2batch(self, feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat

    def acc_entropy_selection(self, tg_agent, nb_agent, M=3, N=3):
        
        w = nb_agent.shape[-2]
        h = nb_agent.shape[-1]
        batch_nb = nb_agent.reshape(-1, 1, 1, 1) # 32*32 x 1 x 1 x 1
        stack = self.stack(nb_agent).permute(2,3,1,0).contiguous().reshape(-1, 9, 1, 1)
        p = F.sigmoid((stack - batch_nb)).mean(dim=1).reshape(w, h)
        entropy_tmp = p * torch.log(p)
         
        self_holder = entropy_tmp.mean()
        masker = torch.where(entropy_tmp>=self_holder)
        
        stack_tg = self.stack(tg_agent).permute(2,3,1,0).contiguous().reshape(-1, 9, 1, 1)
        p_t = F.sigmoid((stack_tg - batch_nb)).mean(dim=1).reshape(w, h)
        entropy_t = p_t * torch.log(p_t)
        
        # masker
        tmp_masker = - torch.ones_like(entropy_t)
        tmp_masker[masker] = entropy_t[masker]
     
        thresholds = tmp_masker[tmp_masker!=-1].mean()
        
        return torch.where(tmp_masker>=thresholds)
    
    def interplot_f(self, feature, masker):
        
        masker_t = torch.zeros_like(feature)
        masker_t[:, masker[0], masker[1]] = 1
        masker_f = masker_t[None, :, :, :].float()
        
        if feature.shape[0] == 256:
            inter = self.masker(feature.unsqueeze(0), masker_f)
        elif feature.shape[0] == 128:
            inter = self.masker_v1(feature.unsqueeze(0), masker_f)
        
        return torch.squeeze(inter)   

    def forward(self, bevs, trans_matrices, num_agent_tensor, gnss, imu, batch_size=1):

        if batch_size == 1:
            imu = torch.unsqueeze(imu, 0)
            gnss = torch.unsqueeze(gnss, 0)
        
        bevs = bevs.permute(0, 1, 4, 2, 3) # (Batch * agents, seq, z, h, w)
        x_0,x_1,x_2,x_3,x_4 = self.u_encoder(bevs)
        device = bevs.device

        if self.layer ==4:
            feat_maps = x_4
            size = (1, 512, 16, 16)
        elif self.layer ==3:
            feat_maps = x_3
            size = (1, 256, 32, 32)
        elif self.layer == 2:
            feat_maps = x_2
            size = (1, 128, 64, 64)
        elif self.layer == 1:
            feat_maps = x_1
            size = (1, 64, 128, 128)
        elif self.layer == 0:
            feat_maps = x_0
            size = (1, 32, 256, 256)

        feat_map = {}
        feat_list = []

        for i in range(self.agent_num):
            feat_map[i] = torch.unsqueeze(feat_maps[batch_size * i:batch_size * (i + 1)], 1)
            # shape of feat_maps [batch * agents, 256c, 32w, 32h] 
            feat_list.append(feat_map[i])
            # import pdb; pdb.set_trace()

        local_com_mat = torch.cat(tuple(feat_list), 1) # [2 5 512 16 16] [batch, agent, channel, height, width]
        local_com_mat_update = torch.cat(tuple(feat_list), 1) # to avoid the inplace operation

        save_agent_weight_list = list()
        p = np.array([1.0, 0.0])

        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]
            for i in range(num_agent):
                tg_agent = local_com_mat[b, i]

                hid_infor = torch.unsqueeze(self.hidden_f_last[b][i], 0).to(device)
                rot_infor = torch.unsqueeze(imu[b, i], 0).to(device)
                hd_grid_rot = F.affine_grid(rot_infor, size=torch.Size(size))
                trans_infor = torch.unsqueeze(gnss[b, i], 0).to(device)
                hd_grid_trans = F.affine_grid(trans_infor, size=torch.Size(size))

                warp_hd_rot = F.grid_sample(hid_infor, hd_grid_rot, mode='bilinear')
                warp_hd_trans = F.grid_sample(warp_hd_rot, hd_grid_trans, mode='bilinear')

                z, hd_agent = self.gru_r_z(warp_hd_trans, tg_agent)

                all_warp = trans_matrices[b, i]

                neighbor_feat_list = list()
                neighbor_feat_list.append(tg_agent)

                p_com_outage = np.random.choice([0, 1], p=p.ravel())
                if p_com_outage==1:
                    agent_wise_weight_feat = neighbor_feat_list[0]
                else:
                    for j in range(num_agent):
                        if j != i:
                            nb_agent = torch.unsqueeze(local_com_mat[b, j], 0)
                            nb_warp = all_warp[j] # [4 4]  corresponding trans_matrices
                            # normalize the translation vector
                            x_trans = (4*nb_warp[0, 3])/128
                            y_trans = -(4*nb_warp[1, 3])/128

                            theta_rot = torch.tensor([[nb_warp[0,0], nb_warp[0,1], 0.0], [nb_warp[1,0], nb_warp[1,1], 0.0]]).type(dtype=torch.float).to(device)
                            theta_rot = torch.unsqueeze(theta_rot, 0)
                            grid_rot = F.affine_grid(theta_rot, size=torch.Size(size))  # for grid sample

                            theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(device)
                            theta_trans = torch.unsqueeze(theta_trans, 0)
                            grid_trans = F.affine_grid(theta_trans, size=torch.Size(size))  # for grid sample

                            # first rotate the feature map, then translate it
                            warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
                            warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
                            warp_feat = torch.squeeze(warp_feat_trans)

                            if warp_feat.min() + warp_feat.max() == 0:
                                neighbor_feat_list.append(warp_feat)
                            else:
                                tg_agent_com = self.compress(torch.unsqueeze(tg_agent, 0))
                                warp_feat_trans_com = self.compress(warp_feat_trans)
                                import time
                                s = time.time()
                                selection = self.acc_entropy_selection(tg_agent_com, warp_feat_trans_com, M=3, N=3)
                                print(time.time()-s)
                                warp_feat_interplot = self.interplot_f(warp_feat, selection)
                                neighbor_feat_list.append(warp_feat_interplot)

                    tmp_agent_weight_list = list()
                    sum_weight = 0
                    for k in range(num_agent):
                        cat_feat = torch.cat([hd_agent, tg_agent, neighbor_feat_list[k]], dim=0)
                        cat_feat = cat_feat.unsqueeze(0)
                        AgentWeight = torch.squeeze(self.PixelWeightedFusion(cat_feat))
                        tmp_agent_weight_list.append(torch.exp(AgentWeight))
                        sum_weight = sum_weight + torch.exp(AgentWeight)
                    
                    agent_weight_list = list()
                    for k in range(num_agent):
                        AgentWeight = torch.div(tmp_agent_weight_list[k], sum_weight)
                        AgentWeight.expand([256, -1, -1])
                        agent_weight_list.append(AgentWeight)
                    
                    agent_wise_weight_feat = 0

                    for k in range(num_agent):
                        agent_wise_weight_feat = agent_wise_weight_feat + agent_weight_list[k]*neighbor_feat_list[k]

                # feature update and hid renew
                out_stage = torch.mul(1-z, agent_wise_weight_feat.unsqueeze(0)) + torch.mul(z, hd_agent.unsqueeze(0))
                # re_m, next_hid = self.grucell(out_stage)
                local_com_mat_update[b, i] = out_stage.squeeze(0)
                self.hidden_f_last[b][i] = (F.relu(self.bn_w_hid(self.cnn_w_hid(out_stage))).squeeze(0)).detach()

        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents2batch(local_com_mat_update)

        if self.kd_flag == 1:
            if self.layer ==4:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,x_2,x_3,feat_fuse_mat,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 3:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,x_2,feat_fuse_mat,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 2:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,x_1,feat_fuse_mat,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 1:
                x_8, x_7, x_6, x_5 = self.decoder(x_0,feat_fuse_mat,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 0:
                x_8, x_7, x_6, x_5 = self.decoder(feat_fuse_mat,x_1,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)

            x = x_8
        else:
            if self.layer ==4:
                x = self.decoder(x_0,x_1,x_2,x_3,feat_fuse_mat,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 3:
                x = self.decoder(x_0,x_1,x_2,feat_fuse_mat,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 2:
                x = self.decoder(x_0,x_1,feat_fuse_mat,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 1:
                x = self.decoder(x_0,feat_fuse_mat,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)
            elif self.layer == 0:
                x = self.decoder(feat_fuse_mat,x_1,x_2,x_3,x_4,batch_size, kd_flag = self.kd_flag)

        # Cell Classification head
        cls_preds = self.classification(x)
        # import pdb; pdb.set_trace()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0],-1,self.category_num)

        # Detection head
        loc_preds =self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1,loc_preds.size(1),loc_preds.size(2),self.anchor_num_per_loc,self.out_seq_len,self.box_code_size)
        # loc_pred (N * T * W * H * loc)
        result = {'loc': loc_preds,
                  'cls': cls_preds}

        #MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0],-1,motion_cat)
            result['state'] = motion_cls_preds

        if self.kd_flag == 1:
            return result, x_8, x_7, x_6, x_5, feat_fuse_mat
        else:
            return result  

##############################################################################################

class ModulationLayer3(nn.Module):
    def __init__(self,config):
        super(ModulationLayer3, self).__init__()

        self.conv1_1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))

        return x_1

class MapScore(nn.Module):
    def __init__(self, channel):
        super(MapScore, self).__init__()
        self.conv1_1 = nn.Conv2d(channel, 32, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(32)

        self.conv1_2 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(8)

        self.conv1_3 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_1 = F.sigmoid(self.conv1_3(x_1))

        return x_1

class MapScoreV2(nn.Module):
    def __init__(self, channel):
        super(MapScoreV2, self).__init__()
        self.conv1_1 = nn.Conv2d(channel, 32, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)

        self.conv1_2 = nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(8)

        self.conv1_3 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_1 = F.sigmoid(self.conv1_3(x_1))

        return x_1

class CompressNet(nn.Module):
    def __init__(self, channel):
        super(CompressNet, self).__init__()
        self.conv1_1 = nn.Conv2d(channel, 64, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(64)
        
        self.conv1_2 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.conv1_2(x_1))
        
        return x_1

class GRU_R_Z(nn.Module):
    def __init__(self, input_channel):
        super(GRU_R_Z, self).__init__()
        self.cnn_w_ir = nn.Conv2d(input_channel*2, input_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.cnn_w_hz = nn.Conv2d(input_channel*2, input_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_w_ir = nn.BatchNorm2d(input_channel)
        self.bn_w_hz = nn.BatchNorm2d(input_channel)


    def forward(self, hidden_f, tg):
        fuse_f = torch.cat((tg.unsqueeze(0), hidden_f), dim=1)
        w_ir = torch.sigmoid(self.bn_w_ir(self.cnn_w_ir(fuse_f)))
        w_hr = 1 - w_ir
        r = torch.sigmoid(torch.mul(w_ir, hidden_f) + torch.mul(w_hr, tg.unsqueeze(0)))
        w_hz = torch.sigmoid(self.bn_w_hz(self.cnn_w_hz(fuse_f)))
        w_iz = 1 - w_hz
        z = torch.sigmoid(torch.mul(w_iz, hidden_f) + torch.mul(w_hz, tg.unsqueeze(0)))
        hd_agent = torch.squeeze(torch.mul(r, hidden_f))

        return z, hd_agent

class PixelWeightedFusionSoftmax(nn.Module):
    def __init__(self,channel):
        super(PixelWeightedFusionSoftmax, self).__init__()

        self.conv1_1 = nn.Conv2d(channel, 128, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(128)

        self.conv1_2 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(32)

        self.conv1_3 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0)
        self.bn1_3 = nn.BatchNorm2d(8)

        self.conv1_4 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
        # self.bn1_4 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_1 = F.relu(self.bn1_3(self.conv1_3(x_1)))
        x_1 = F.relu(self.conv1_4(x_1))

        return x_1

class AgentWeightedFusion(nn.Module):
    def __init__(self,config):
        super(AgentWeightedFusion, self).__init__()

        self.conv1_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(128)

        self.conv1_2 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(32)

        self.conv1_3 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0)
        self.bn1_3 = nn.BatchNorm2d(8)

        self.conv1_4 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)

        # self.conv1_1 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
        # self.bn1_1 = nn.BatchNorm2d(1)
        self.conv1_5 = nn.Conv2d(1, 1, kernel_size=32, stride=1, padding=0)
        # # self.bn1_2 = nn.BatchNorm2d(1)

    def forward(self, x):
        # x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        # x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        # x_1 = F.sigmoid(self.conv1_2(x_1))
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_1 = F.relu(self.bn1_3(self.conv1_3(x_1)))
        x_1 = F.relu(self.conv1_4(x_1))
        x_1 = F.relu(self.conv1_5(x_1))

        return x_1

class ClassificationHead(nn.Module):

    def __init__(self, config):

        super(ClassificationHead, self).__init__()
        category_num = config.category_num
        channel = 32
        if config.use_map:
            channel += 6
        if config.use_vis:
            channel += 13

        anchor_num_per_loc = len(config.anchor_size)

        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channel, category_num*anchor_num_per_loc, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(channel)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x

class SingleRegressionHead(nn.Module):
    def __init__(self,config):
        super(SingleRegressionHead,self).__init__()
        category_num = config.category_num
        channel = 32
        if config.use_map:
            channel += 6
        if config.use_vis:
            channel += 13

        anchor_num_per_loc = len(config.anchor_size)
        box_code_size = config.box_code_size
        if config.only_det:
            out_seq_len = 1
        else:
            out_seq_len = config.pred_len
    
        if config.binary:
            if config.only_det:
                self.box_prediction = nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(channel),
                    nn.ReLU(),
                    nn.Conv2d(channel, anchor_num_per_loc * box_code_size * out_seq_len, kernel_size=1, stride=1, padding=0))        
            else:      
                self.box_prediction = nn.Sequential(
                    nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, anchor_num_per_loc * box_code_size * out_seq_len, kernel_size=1, stride=1, padding=0))
                

    def forward(self,x):
        box = self.box_prediction(x)

        return box

class TeacherNet(nn.Module):
    def __init__(self, config):
        super(TeacherNet, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        # self.RegressionList = nn.ModuleList([RegressionHead for i in range(seq_len)])
        self.regression = SingleRegressionHead(config)
        # self.fusion_method = config.fusion_method

        # if self.use_map:
        #     if self.fusion_method == 'early_fusion':
        #         self.stpn = STPN(height_feat_size=config.map_dims[2]+config.map_channel)
        #     elif self.fusion_method == 'middle_fusion':
        #         self.stpn = STPN(height_feat_size=config.map_dims[2],use_map=True)
        #     elif self.fusion_method == 'late_fusion':
        #         self.map_encoder = MapExtractor(map_channel=config.map_channel)
        #         self.stpn = STPN(height_feat_size=config.map_dims[2])
        # else:
        self.stpn = STPN_KD(height_feat_size=config.map_dims[2])

        # if self.motion_state:
        #     self.motion_cls = MotionStateHead(config)

    def forward(self, bevs, maps=None, vis=None):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)
        # vis = vis.permute(0, 3, 1, 2)
        x_8, x_7, x_6, x_5, x_3, x_2 = self.stpn(bevs)
        return x_8, x_7, x_6, x_5, x_3, x_2

class FaFNet(nn.Module):
    def __init__(self, config):
        super(FaFNet, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)

        self.stpn = STPN_KD(height_feat_size=config.map_dims[2])

    def forward(self, bevs, frame_id, maps=None, vis=None, batch_size=None):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)
        # vis = vis.permute(0, 3, 1, 2)
        x_8, x_7, x_6, x_5, x_3, x_2 = self.stpn(bevs)

        x = x_8
        # if not maps is None:
        #     x = torch.cat([x,maps],axis=-1)
        # if not vis is None:
        #     x = torch.cat([x,vis],axis=1)

        #Cell Classification head
        
        # desire_frame = 59
        
        # save_dir = "./89_59_teacher_det_figure/"
        # check_folder(save_dir)
        # import matplotlib.pyplot as plt
        
        # if frame_id == desire_frame:
        #     for tmp_i in range(5):
        #         plt.matshow(x[tmp_i].sum(axis=0).detach().cpu().numpy(),cmap='hot')
        #         plt.axis('off')
        #         plt.savefig(save_dir+str(frame_id) + "_x_" + str(tmp_i) + "_.png",bbox_inches='tight',pad_inches=0)
        
        # if frame_id == desire_frame+1:
        #     import pdb; pdb.set_trace()
            
        cls_preds = self.classification(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0],-1,self.category_num)

        # Detection head
        loc_preds =self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1,loc_preds.size(1),loc_preds.size(2),self.anchor_num_per_loc,self.out_seq_len,self.box_code_size)
        #loc_pred (N * T * W * H * loc)
        result = {'loc': loc_preds,
                  'cls': cls_preds}

        #MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0],-1,motion_cat)
            result['state'] = motion_cls_preds
        return result

class policy_net4(nn.Module):
    def __init__(self, in_channels=13, input_feat_sz=32):
        super(policy_net4, self).__init__()
        feat_map_sz = input_feat_sz // 4
        self.n_feat = int(256 * feat_map_sz * feat_map_sz)
        self.lidar_encoder = lidar_encoder(height_feat_size=in_channels)

        # Encoder
        # down 1 
        self.conv1 = conv2DBatchNormRelu(512, 512, k_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(512, 256, k_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(256, 256, k_size=3, stride=2, padding=1)

        # down 2
        self.conv4 = conv2DBatchNormRelu(256, 256, k_size=3, stride=1, padding=1)
        self.conv5 = conv2DBatchNormRelu(256, 256, k_size=3, stride=2, padding=1)

    def forward(self, features_map):
        _, _, _, _, outputs1 = self.lidar_encoder(features_map)
        outputs = self.conv1(outputs1)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)
        return outputs

class MIMOGeneralDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, query_size, key_size, warp_flag, attn_dropout=0.1):
        super().__init__()
        self.sparsemax = Sparsemax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(query_size, key_size)
        self.warp_flag = warp_flag
        print('Msg size: ',query_size,'  Key size: ', key_size)

    def forward(self, qu, k, v, sparse=True):
        # qu (batch,5,32)
        # k (batch,5,1024)
        # v (batch,5,channel,size,size)
        query = self.linear(qu)  # (batch,5,key_size)

        # normalization
        # query_norm = query.norm(p=2,dim=2).unsqueeze(2).expand_as(query)
        # query = query.div(query_norm + 1e-9)

        # k_norm = k.norm(p=2,dim=2).unsqueeze(2).expand_as(k)
        # k = k.div(k_norm + 1e-9)
        # generate the
        attn_orig = torch.bmm(k, query.transpose(2, 1))  # (batch,5,5)  column: differnt keys and the same query

        # scaling [not sure]
        # scaling = torch.sqrt(torch.tensor(k.shape[2],dtype=torch.float32)).cuda()
        # attn_orig = attn_orig/ scaling # (batch,5,5)  column: differnt keys and the same query

        attn_orig_softmax = self.softmax(attn_orig)  # (batch,5,5)

        attn_shape = attn_orig_softmax.shape
        bats, key_num, query_num = attn_shape[0], attn_shape[1], attn_shape[2]
        attn_orig_softmax_exp = attn_orig_softmax.view(bats, key_num, query_num, 1, 1, 1)

        if self.warp_flag==1:
            v_exp = v
        else:
            v_exp = torch.unsqueeze(v, 2)
            v_exp = v_exp.expand(-1, -1, query_num, -1, -1, -1)

        output = attn_orig_softmax_exp * v_exp  # (batch,5,channel,size,size)
        output_sum = output.sum(1)  # (batch,1,channel,size,size)

        return output_sum, attn_orig_softmax

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path