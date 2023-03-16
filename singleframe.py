import cv2
import json
import numpy as np
import onnxruntime
import pandas as pd

from matplotlib import pyplot as plt

X_IDXS = np.array([ 0. ,   0.1875,   0.75  ,   1.6875,   3.    ,   4.6875,
         6.75  ,   9.1875,  12.    ,  15.1875,  18.75  ,  22.6875,
        27.    ,  31.6875,  36.75  ,  42.1875,  48.    ,  54.1875,
        60.75  ,  67.6875,  75.    ,  82.6875,  90.75  ,  99.1875,
       108.    , 117.1875, 126.75  , 136.6875, 147.    , 157.6875,
       168.75  , 180.1875, 192.])

def parse_image(frame):
	H = (frame.shape[0]*2)//3
	W = frame.shape[1]
	parsed = np.zeros((6, H//2, W//2), dtype=np.uint8)

	parsed[0] = frame[0:H:2, 0::2]
	parsed[1] = frame[1:H:2, 0::2]
	parsed[2] = frame[0:H:2, 1::2]
	parsed[3] = frame[1:H:2, 1::2]
	parsed[4] = frame[H:H+H//4].reshape((-1, H//2,W//2))
	parsed[5] = frame[H+H//4:H+H//2].reshape((-1, H//2,W//2))

	return parsed

def seperate_points_and_std_values(df):
	points = df.iloc[lambda x: x.index % 2 == 0]
	std = df.iloc[lambda x: x.index % 2 != 0]
	points = pd.concat([points], ignore_index = True)
	std = pd.concat([std], ignore_index = True)

	return points, std

def getSuperComboOutput(vision_matrix):
	model = "supercombo_0.8.9.onnx"
	
	cap = vision_matrix
	parsed_images = []

	width = 512
	height = 256
	dim = (width, height)
	
	plan_start_idx = 0
	plan_end_idx = 4955
	
	lanes_start_idx = plan_end_idx
	lanes_end_idx = lanes_start_idx + 528
	
	lane_lines_prob_start_idx = lanes_end_idx
	lane_lines_prob_end_idx = lane_lines_prob_start_idx + 8
	
	road_start_idx = lane_lines_prob_end_idx
	road_end_idx = road_start_idx + 264

# 	lead_start_idx = road_end_idx
# 	lead_end_idx = lead_start_idx + 55
# 	
# 	lead_prob_start_idx = lead_end_idx
# 	lead_prob_end_idx = lead_prob_start_idx + 3
# 	
# 	desire_start_idx = lead_prob_end_idx
# 	desire_end_idx = desire_start_idx + 72
# 	
# 	meta_start_idx = desire_end_idx
# 	meta_end_idx = meta_start_idx + 32
# 	
# 	desire_pred_start_idx = meta_end_idx
# 	desire_pred_end_idx = desire_pred_start_idx + 32
# 	
# 	pose_start_idx = desire_pred_end_idx
# 	pose_end_idx = pose_start_idx + 12
# 	
# 	rnn_start_idx = pose_end_idx
# 	rnn_end_idx = rnn_start_idx + 908
	
	session = onnxruntime.InferenceSession(model, None)

	print(cap.shape)

	img = cv2.resize(cap, dim)
	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
	parsed = parse_image(img_yuv)
	
	
	parsed_images.append(parsed)

	
		
	parsed_arr = np.array(parsed_images)
	parsed_arr.resize((1,12,128,256))

	data = json.dumps({'data': parsed_arr.tolist()})
	data = np.array(json.loads(data)['data']).astype('float32')
			
	input_imgs = session.get_inputs()[0].name
	desire = session.get_inputs()[1].name
	initial_state = session.get_inputs()[2].name
	traffic_convention = session.get_inputs()[3].name
	output_name = session.get_outputs()[0].name
			
	desire_data = np.array([0]).astype('float32')
	desire_data.resize((1,8))
			
	traffic_convention_data = np.array([0]).astype('float32')
	traffic_convention_data.resize((1,512))
			
	initial_state_data = np.array([0]).astype('float32')
	initial_state_data.resize((1,2))

	result = session.run([output_name], {input_imgs: data,
												desire: desire_data,
												traffic_convention: traffic_convention_data,
												initial_state: initial_state_data
												})

	res = np.array(result)

	print(res[0])




