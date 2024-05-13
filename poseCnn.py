
import scipy.io
import json
import glob
# "48/1": [{"bbox_est": [207.6616668701172, 140.1031494140625, 128.17332458496094, 188.31982421875], "obj_id": 1, "score": 0.908555269241333, "time": 0.05265476700151339}, {"bbox_est": [281.0294494628906, 281.28741455078125, 113.8648681640625, 74.43646240234375], "obj_id": 6, "score": 0.8901379108428955, "time": 0.05265476700151339}, {"bbox_est": [210.63357543945312, 41.13779830932617, 133.0020751953125, 111.98211669921875], "obj_id": 14, "score": 0.7543352842330933, "time": 0.05265476700151339}, {"bbox_est": [94.84809112548828, 329.53082275390625, 344.0975646972656, 121.43252563476562], "obj_id": 20, "score": 0.7254160642623901, "time": 0.05265476700151339}, {"bbox_est": [325.288818359375, 115.03593444824219, 87.847900390625, 213.0474395751953], "obj_id": 19, "score": 0.6487630009651184, "time": 0.05265476700151339}, {"bbox_est": [94.42133331298828, 328.5173645019531, 345.0547180175781, 122.87893676757812], "obj_id": 19, "score": 0.42629653215408325, "time": 0.05265476700151339}, {"bbox_est": [324.77960205078125, 116.16062927246094, 89.93930053710938, 213.80372619628906], "obj_id": 20, "score": 0.3502214252948761, "time": 0.05265476700151339}, {"bbox_est": [279.1546630859375, 282.34710693359375, 110.59918212890625, 119.76065063476562], "obj_id": 6, "score": 0.2591652572154999, "time": 0.05265476700151339}, {"bbox_est": [324.77960205078125, 116.16062927246094, 89.93930053710938, 213.80372619628906], "obj_id": 18, "score": 0.1754877120256424, "time": 0.05265476700151339}, {"bbox_est": [322.88604736328125, 115.99964141845703, 91.47503662109375, 215.96865844726562], "obj_id": 1, "score": 0.08767641335725784, "time": 0.05265476700151339}],


resJson = {}
keyframes = []
with open('/media/sda1/r10922190/BOP_DATASETS/ycbv_origin/keyframe.txt') as f:
    for line in f.readlines():
        keyframes.append(line[:-1])

paths = sorted(glob.glob(
    '/media/sda1/r10922190/BOP_DATASETS/ycbv_origin/results_PoseCNN/*.mat'))

for id, path in enumerate(paths):
    scene_id = str(int(keyframes[id].split('/')[0]))
    im_id = str(int(keyframes[id].split('/')[1]))
    cur_res = []
    res = scipy.io.loadmat(path)
    for roi in res['rois']:
        bbox = {}
        obj_id = roi[1]
        x1 = roi[2]
        y1 = roi[3]
        w = roi[4] - roi[2]
        h = roi[5] - roi[3]
        bbox['bbox_est'] = [float(x1), float(y1), float(w), float(h)]
        bbox['obj_id'] = int(obj_id)
        bbox['score'] = 1
        bbox['time'] = 0.1
        cur_res.append(bbox)
    resJson[f"{scene_id}/{im_id}"] = cur_res
# Serializing json
json_object = json.dumps(resJson, indent=4)
# Writing to sample.json
with open("sample.json", "w") as outfile:
    outfile.write(json_object)
