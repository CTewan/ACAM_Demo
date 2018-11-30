import numpy as np
import cv2
import imageio
import tensorflow as tf
import json

import os
import sys
import argparse

import object_detection.object_detector as obj
import action_detection.action_detector as act

from multiprocessing import Process, Queue

import time
DISPLAY = False
SHOW_CAMS = False

ACTION_FREQ = 8

# separate process definitions

# frame reader
def read_frames(reader, frame_q):
    for cur_img in reader:
        frame_q.put(cur_img)

# object detector and tracker
def run_obj_det_and_track(frame_q, detection_q, det_vis_q):
    ## Best
    # obj_detection_graph =  os.path.join(main_folder, 'object_detection/weights/batched_zoo/faster_rcnn_nas_coco_2018_01_28/batched_graph/frozen_inference_graph.pb')
    ## Good and Faster
    #obj_detection_graph =  os.path.join(main_folder, 'object_detection/weights/batched_zoo/faster_rcnn_nas_lowproposals_coco_2018_01_28/batched_graph/frozen_inference_graph.pb')
    ## Fastest
    #obj_detection_graph =  os.path.join(main_folder, 'object_detection/weights/batched_zoo/faster_rcnn_resnet50_coco_2018_01_28/batched_graph/frozen_inference_graph.pb')

    # NAS
    obj_detection_graph =  '/home/oytun/work/tensorflow_object/zoo/batched_zoo/faster_rcnn_nas_coco_2018_01_28_lowth/batched_graph/frozen_inference_graph.pb'


    print("Loading object detection model at %s" % obj_detection_graph)


    obj_detector = obj.Object_Detector(obj_detection_graph)
    tracker = obj.Tracker()
    while True:
        cur_img = frame_q.get()
        expanded_img = np.expand_dims(cur_img, axis=0)
        detection_list = obj_detector.detect_objects_in_np(expanded_img)
        detection_info = [info[0] for info in detection_list]
        tracker.update_tracker(detection_info, cur_img)
        rois_np, temporal_rois_np = tracker.generate_all_rois()
        detection_q.put([cur_img, tracker.active_actors[:], rois_np, temporal_rois_np])
        det_vis_q.put([cur_img, tracker.active_actors[:]])


# Action detector
def run_act_detector(act_detector, shape, detection_q, actions_q):
    # act_detector = act.Action_Detector('i3d_tail')
    # ckpt_name = 'model_ckpt_RGB_i3d_pooled_tail-4'
    act_detector = act.Action_Detector('soft_attn')
    #ckpt_name = 'model_ckpt_RGB_soft_attn-16'
    #ckpt_name = 'model_ckpt_soft_attn_ava-23'
    ckpt_name = 'model_ckpt_soft_attn_pooled_ava-52'

    #input_frames, temporal_rois, temporal_roi_batch_indices, cropped_frames = act_detector.crop_tubes_in_tf([T,H,W,3])
    memory_size = act_detector.timesteps - ACTION_FREQ
    updated_frames, temporal_rois, temporal_roi_batch_indices, cropped_frames = act_detector.crop_tubes_in_tf_with_memory(shape, memory_size)
    
    rois, roi_batch_indices, pred_probs = act_detector.define_inference_with_placeholders_noinput(cropped_frames)
    

    ckpt_path = os.path.join(main_folder, 'action_detection', 'weights', ckpt_name)
    act_detector.restore_model(ckpt_path)


    while True:
        images = []
        for _ in range(ACTION_FREQ):
            cur_img, active_actors, rois_np, temporal_rois_np = detection_q.get()
            images.append(cur_img)
        
        if not active_actors:
            continue
        # use the last active actors and rois vectors
        no_actors = len(active_actors)

        cur_input_sequence = np.expand_dims(np.stack(images, axis=0), axis=0)

        if no_actors > 14:
            no_actors = 14
            rois_np = rois_np[:14]
            temporal_rois_np = temporal_rois_np[:14]
            active_actors = active_actors[:14]

        #feed_dict = {input_frames:cur_input_sequence, 
        feed_dict = {updated_frames:cur_input_sequence, # only update last #action_freq frames
                        temporal_rois: temporal_rois_np,
                        temporal_roi_batch_indices: np.zeros(no_actors),
                        rois:rois_np, 
                        roi_batch_indices:np.arange(no_actors)}
        run_dict = {'pred_probs': pred_probs}

        out_dict = act_detector.session.run(run_dict, feed_dict=feed_dict)
        probs = out_dict['pred_probs']

        # associate probs with actor ids
        print_top_k = 5
        prob_dict = {}
        for bb in range(no_actors):
            act_probs = probs[bb]
            order = np.argsort(act_probs)[::-1]
            cur_actor_id = tracker.active_actors[bb]['actor_id']
            print("Person %i" % cur_actor_id)
            cur_results = []
            for pp in range(print_top_k):
                print('\t %s: %.3f' % (act.ACTION_STRINGS[order[pp]], act_probs[order[pp]]))
                cur_results.append((act.ACTION_STRINGS[order[pp]], act_probs[order[pp]]))
            prob_dict[cur_actor_id] = cur_results
        
        actions_q.put(prob_dict)



# Visualization
def run_visualization(writer, det_vis_q, actions_q):
    frame_cnt = 0
    prob_dict = actions_q.get() # skip the first one

    while True:
        frame_cnt += 1
        cur_img, active_actors = det_vis_q.get()
        if frame_cnt % ACTION_FREQ == 0:
            prob_dict = actions_q.get()

        out_img = visualize_detection_results(cur_img, active_actors, prob_dict)
    
        if DISPLAY: 
            cv2.imshow('results', out_img[:,:,::-1])
            cv2.waitKey(10)
        else:
            writer.append_data(out_img)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--video_path', type=str, required=True)

    args = parser.parse_args()
    
    actor_to_display = 6 # for cams

    video_path = args.video_path
    basename = os.path.basename(video_path).split('.')[0]
    #out_vid_path = "./output_videos/%s_output.mp4" % (basename if not SHOW_CAMS else basename+'_cams_actor_%.2d' % actor_to_display)
    out_vid_path = './output_videos/testing.mp4'

    # video_path = "./tests/chase1Person1View3Point0.mp4"
    # out_vid_path = 'output.mp4'

    main_folder = './'
    

    print("Reading video file %s" % video_path)
    reader = imageio.get_reader(video_path, 'ffmpeg')
    
    # fps_divider = 1
    print('Running actions every %i frame' % ACTION_FREQ)
    fps = reader.get_meta_data()['fps'] #// fps_divider
    W, H = reader.get_meta_data()['size']
    T = tracker.timesteps
    if not DISPLAY:
        writer = imageio.get_writer(out_vid_path, fps=fps)
        print("Writing output to %s" % out_vid_path)

    shape = [T,H,W,3]

    

    frame_q = Queue()
    detection_q = Queue()
    det_vis_q = Queue()
    actions_q = Queue()

    frame_reader_p = Process(target=read_frames, args=(reader, frame_q))
    obj_detector_p = Process(target=run_obj_det_and_track, args=(frame_q, detection_q, det_vis_q))
    action_detector_p = Process(target=run_act_detector, args=(act_detector, shape, tf_pointers, detection_q, actions_q))
    visualization_p = Process(target=run_visualization, args=(writer, det_vis_q, actions_q))

    processes = [frame_reader_p, obj_detector_p, action_detector_p, visualization_p]

    for process in processes:
        process.daemon = True
        process.start()

    while True:
        time.sleep(1)
    
        
    if not DISPLAY:
        writer.close()


np.random.seed(10)
COLORS = np.random.randint(0, 255, [1000, 3])
def visualize_detection_results(img_np, active_actors, prob_dict):
    score_th = 0.30
    action_th = 0.20

    # copy the original image first
    disp_img = np.copy(img_np)
    H, W, C = img_np.shape
    #for ii in range(len(active_actors)):
    for ii in range(len(active_actors)):
        cur_actor = active_actors[ii]
        actor_id = cur_actor['actor_id']
        cur_act_results = prob_dict[actor_id] if actor_id in prob_dict else []
        cur_box, cur_score, cur_class = cur_actor['all_boxes'], cur_actor['all_scores'], 1
        
        if cur_score < score_th: 
            continue

        top, left, bottom, right = cur_box


        left = int(W * left)
        right = int(W * right)

        top = int(H * top)
        bottom = int(H * bottom)

        conf = cur_score
        #label = bbox['class_str']
        # label = 'Class_%i' % cur_class
        label = obj.OBJECT_STRINGS[cur_class]['name']
        message = '%s_%i: %% %.2f' % (label, actor_id,conf)
        action_message_list = ["%s:%.3f" % (actres[0][0:7], actres[1]) for actres in cur_act_results if actres[1]>action_th]
        # action_message = " ".join(action_message_list)

        color = COLORS[actor_id]

        cv2.rectangle(disp_img, (left,top), (right,bottom), color, 3)

        font_size =  max(0.5,(right - left)/50.0/float(len(message)))
        cv2.rectangle(disp_img, (left, top-int(font_size*40)), (right,top), color, -1)
        cv2.putText(disp_img, message, (left, top-12), 0, font_size, (255,255,255)-color, 1)

        #action message writing
        cv2.rectangle(disp_img, (left, top), (right,top+10*len(action_message_list)), color, -1)
        for aa, action_message in enumerate(action_message_list):
            offset = aa*10
            cv2.putText(disp_img, action_message, (left, top+5+offset), 0, 0.5, (255,255,255)-color, 1)

    return disp_img


def visualize_cams(image, input_frames, out_dict, actor_idx):
    #classes = ["walk", "bend", "carry"]
    #classes = ["sit", "ride"]
    classes = ["talk to", "watch (a", "listen to"]
    action_classes = [cc for cc in range(60) if any([cname in act.ACTION_STRINGS[cc] for cname in classes])]

    feature_activations = out_dict['final_i3d_feats']
    cls_weights = out_dict['cls_weights']
    input_frames = out_dict['cropped_frames'].astype(np.uint8)
    probs = out_dict["pred_probs"]

    class_maps = np.matmul(feature_activations, cls_weights)
    min_val = np.min(class_maps[:,:, :, :, :])
    max_val = np.max(class_maps[:,:, :, :, :]) - min_val

    normalized_cmaps = np.uint8((class_maps-min_val)/max_val * 255.)

    t_feats = feature_activations.shape[1]
    t_input = input_frames.shape[1]
    index_diff = (t_input) // (t_feats+1)

    img_new_height = 400
    img_new_width = int(image.shape[1] / float(image.shape[0]) * img_new_height)
    img_to_show = cv2.resize(image.copy(), (img_new_width,img_new_height))[:,:,::-1]
    #img_to_concat = np.zeros((400, 800, 3), np.uint8)
    img_to_concat = np.zeros((400, 400, 3), np.uint8)
    for cc in range(len(action_classes)):
        cur_cls_idx = action_classes[cc]
        act_str = act.ACTION_STRINGS[action_classes[cc]]
        message = "%s:%%%.2d" % (act_str[:20], 100*probs[actor_idx, cur_cls_idx])
        for tt in range(t_feats):
            cur_cam = normalized_cmaps[actor_idx, tt,:,:, cur_cls_idx]
            cur_frame = input_frames[actor_idx, (tt+1) * index_diff, :,:,::-1]

            resized_cam = cv2.resize(cur_cam, (100,100))
            colored_cam = cv2.applyColorMap(resized_cam, cv2.COLORMAP_JET)

            overlay = cv2.resize(cur_frame.copy(), (100,100))
            overlay = cv2.addWeighted(overlay, 0.5, colored_cam, 0.5, 0)

            img_to_concat[cc*125:cc*125+100, tt*100:(tt+1)*100, :] = overlay
        cv2.putText(img_to_concat, message, (20, 13+100+125*cc), 0, 0.5, (255,255,255), 1)

    final_image = np.concatenate([img_to_show, img_to_concat], axis=1)
    return final_image[:,:,::-1]






    


if __name__ == '__main__':
    main()