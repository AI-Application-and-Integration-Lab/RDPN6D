import pickle
import numpy as np

# f = open('/disk2/RGBD-6dpose/GDR-Net/output/gdrn/delta/test_epoch_120_v3/inference_model_1985609/delta_test/a-test_delta_test_errors.pkl', 'rb')
# data = pickle.load(f)
# print(np.array(data['CHASSIS_1300']['ad'])[(np.array(data['CHASSIS_1300']['ad']) > 0.1)].size)

ff = open('/disk2/RGBD-6dpose/GDR-Net/output/gdrn/delta/test_epoch_120_v3/inference_model_2075864/delta_test/a-test_delta_test_recalls.pkl', 'rb')
data = pickle.load(ff)
index = np.array(data['CHASSIS_1300']['ad_10']) < 0.1
print(np.array(data['CHASSIS_1300']['ad_10']) < 0.1)
f = open('/disk2/RGBD-6dpose/GDR-Net/output/gdrn/delta/test_epoch_120_pbr_v3/inference_model_final/delta_test/a-test_delta_test_preds.pkl', 'rb')
data = pickle.load(f)
print(data['Assemble'].keys())
print(data['Assemble']['datasets/BOP_DATASETS/delta/test/000050/rgb/000000.jpg']['R'])
print(data['Assemble']['datasets/BOP_DATASETS/delta/test/000050/rgb/000000.jpg']['t'])
#print(list(data['Assemble']))

#print(np.array(list(data['Assemble']))[0])