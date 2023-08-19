# from nuscenes.nuscenes import NuScenes as V2XSimDataset
# from v2x_sim_visualizer import render_sample_data, render_scene_lidar
#
# v2x_sim = V2XSimDataset(version='v1.0-mini', dataroot='/media/th/T7-1/dataset/v2x-sim', verbose=True)
#
# # To investigate a single scene
# my_scene = v2x_sim.scene[0]
# print("my_scene", my_scene)
# first_sample_token = my_scene['first_sample_token']
# print("first_sample_token", first_sample_token)
# my_sample = v2x_sim.get('sample', first_sample_token)
# print("my_sample", my_sample)
# channel = 'LIDAR_TOP_id_0'
#
# sample_data_token = my_sample['data'][channel]
# print("sample_data_token", sample_data_token)
# render_sample_data(v2x_sim, sample_data_token, with_anns=True, underlay_map=False, pointsensor_channel=channel, axes_limit=32)

from nuscenes.nuscenes import NuScenes as V2XSimDataset
from v2x_sim_visualizer import render_sample_data, render_scene_lidar

v2x_sim = V2XSimDataset(version='v1.0-mini', dataroot='/media/th/T7-1/dataset/opv2v-test_culver_city-100', verbose=True)

# To investigate a single scene
my_scene = v2x_sim.scene[2]
print("my_scene", my_scene)
first_sample_token = my_scene['first_sample_token']
print("first_sample_token", first_sample_token)
my_sample = v2x_sim.get('sample', first_sample_token)
print("my_sample", my_sample)
channel = 'LIDAR_TOP_id_0'

sample_data_token = my_sample['data'][channel]
print("sample_data_token", sample_data_token)
render_sample_data(v2x_sim, sample_data_token, with_anns=True, underlay_map=False, pointsensor_channel=channel)
