import numpy as np
import cv2
import imageio
import tensorflow as tf
import json
import csv

import os
import sys
sys.path.append("object_detection")
sys.path.append("object_detection/deep_sort")
sys.path.append("action_detection")

import argparse

import object_detection.object_detector as obj
import action_detection.action_detector as act

import time
DISPLAY = False
SHOW_CAMS = False

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--video_path', type=str, required=False, default="")
    parser.add_argument('-d', '--display', type=str, required=False, default="True")

    args = parser.parse_args()
    display = (args.display == "True" or args.display == "true")
    
    #actor_to_display = 6 # for cams

    video_path = args.video_path
    basename = os.path.basename(video_path).split('.')[0]
    out_vid_path = "./output_videos/%s_output.mp4" % (basename if not SHOW_CAMS else basename+'_cams_actor_%.2d' % actor_to_display)
    clf_out_path = "./clf_output/{}_output.csv".format(basename if not SHOW_CAMS else basename+'_cams_actor_{}'.format(actor_to_display))
    #out_vid_path = './output_videos/testing.mp4'

    # video_path = "./tests/chase1Person1View3Point0.mp4"
    # out_vid_path = 'output.mp4'

    main_folder = './'

    # NAS

    obj_detection_model =  'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
    obj_detection_graph = os.path.join("object_detection", "weights", obj_detection_model, "frozen_inference_graph.pb")



    print("Loading object detection model at %s" % obj_detection_graph)


    obj_detector = obj.Object_Detector(obj_detection_graph)
    tracker = obj.Tracker()

    


    print("Reading video file %s" % video_path)
    reader = imageio.get_reader(video_path, 'ffmpeg')
    action_freq = 8
    # fps_divider = 1
    print('Running actions every %i frame' % action_freq)
    fps = reader.get_meta_data()['fps'] #// fps_divider
    print("FPS: {}".format(fps))
    W, H = reader.get_meta_data()['size']
    T = tracker.timesteps
    #if not display:
    writer = imageio.get_writer(out_vid_path, fps=fps)
    csv_file = open(clf_out_path, 'w', newline='')
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['Time', 'Person', 'Action', 'Probability'])
    print("Writing output to %s" % out_vid_path)

    
    # act_detector = act.Action_Detector('i3d_tail')
    # ckpt_name = 'model_ckpt_RGB_i3d_pooled_tail-4'
    act_detector = act.Action_Detector('soft_attn')
    #ckpt_name = 'model_ckpt_RGB_soft_attn-16'
    #ckpt_name = 'model_ckpt_soft_attn_ava-23'
    ckpt_name = 'model_ckpt_soft_attn_pooled_cosine_drop_ava-130'

    #input_frames, temporal_rois, temporal_roi_batch_indices, cropped_frames = act_detector.crop_tubes_in_tf([T,H,W,3])
    memory_size = act_detector.timesteps - action_freq
    updated_frames, temporal_rois, temporal_roi_batch_indices, cropped_frames = act_detector.crop_tubes_in_tf_with_memory([T,H,W,3], memory_size)
    
    rois, roi_batch_indices, pred_probs = act_detector.define_inference_with_placeholders_noinput(cropped_frames)
    

    ckpt_path = os.path.join(main_folder, 'action_detection', 'weights', ckpt_name)
    act_detector.restore_model(ckpt_path)

    prob_dict = {}
    frame_cnt = 0

    # Tewan
    min_teacher_features = 3
    teacher_identified = 0
    #missed_frame_cnt = 0
    #max_age = 120
    #frame_skips = 60
    #next_frame = 0
    teacher_ids = []
    matched_id = None
    # Tewan

    for cur_img in reader:
        frame_cnt += 1

        #if frame_cnt < next_frame:
        #    continue

        # Detect objects and make predictions every 8 frames (0.3 seconds)
        #if frame_cnt % action_freq == 0:

        # Object Detection
        expanded_img = np.expand_dims(cur_img, axis=0)
        detection_list = obj_detector.detect_objects_in_np(expanded_img) 
        detection_info = [info[0] for info in detection_list]
        # Updates active actors in tracker
        tracker.update_tracker(detection_info, cur_img)
        no_actors = len(tracker.active_actors)

        """
        if no_actors == 0:
            missed_frame_cnt += 1

            if missed_frame_cnt >= max_age:
                tracker.update_tracker(detection_info, cur_img)
                no_actors = len(tracker.active_actors)
                teacher_identified = False
                tracker.set_invalid_track()
                missed_frame_cnt = 0

                print("Reset active actors. Current number: {}".format(no_actors))

        """

        if frame_cnt % action_freq == 0 and frame_cnt > 16:
            if no_actors == 0:
                print("No actor found.")
                continue

            video_time = round(frame_cnt / fps, 1)
            valid_actor_ids = [actor["actor_id"] for actor in tracker.active_actors]

            print("frame count: {}, video time: {}s".format(frame_cnt, video_time))
            probs = []

            cur_input_sequence = np.expand_dims(np.stack(tracker.frame_history[-action_freq:], axis=0), axis=0)

            rois_np, temporal_rois_np = tracker.generate_all_rois()

            if teacher_identified < min_teacher_features:
                prompt_img = visualize_detection_results(img_np=tracker.frame_history[-16],
                                                         active_actors=tracker.active_actors,
                                                         prob_dict=None)
                cv2.imshow('prompt_img', prompt_img[:,:,::-1])
                cv2.waitKey(500)
                teacher_present = False

                teacher_id = _prompt_user_input()

                if not _check_teacher_in_frame(teacher_id=teacher_id):
                    print("Teacher not in this frame. Continuing.")
                    cv2.destroyWindow("prompt_img")
                    pass

                else:
                    if _check_valid_teacher_id(teacher_id=teacher_id, valid_actor_ids=valid_actor_ids):
                        teacher_id = int(teacher_id)
                        teacher_identified += 1
                        teacher_present = True

                    else:
                        while not teacher_present:
                            print("Invalid ID.")
                            teacher_id = _prompt_user_input()

                            if not _check_teacher_in_frame(teacher_id=teacher_id):
                                print("Teacher not in this frame. Continuing.")
                                cv2.destroyWindow("prompt_img")
                                break

                            else:
                                if _check_valid_teacher_id(teacher_id=teacher_id, valid_actor_ids=valid_actor_ids):
                                    teacher_id = int(teacher_id)
                                    teacher_identified += 1
                                    teacher_present = True
                                else:
                                    pass

                # Move on to next frame if teacher not in current frame
                if not teacher_present:
                    continue
                cv2.destroyWindow("prompt_img")

                if teacher_id not in teacher_ids:
                    teacher_ids.append(teacher_id)
                    tracker.update_teacher_candidate_ids(teacher_candidate_id=teacher_id)
            else:
                tracker.set_valid_track()

            # Identify idx of teacher for ROI selection                
            roi_idx = None
            found_id = False
            for idx, actor_info in enumerate(tracker.active_actors):
                actor_id = actor_info["actor_id"]
                for i in range(len(teacher_ids)-1, -1, -1):
                    if actor_id == teacher_ids[i]:
                        roi_idx = idx
                        matched_id = actor_info["actor_id"]
                        found_id = True
                        break
                if found_id:
                    break

            # Identify ROI and temporal ROI using ROI idx 
            if roi_idx is not None:
                rois_np = rois_np[roi_idx]
                temporal_rois_np = temporal_rois_np[roi_idx]
                rois_np = np.expand_dims(rois_np, axis=0)
                temporal_rois_np = np.expand_dims(temporal_rois_np, axis=0)
                no_actors = 1
            # If teacher not found (i.e. roi_idx is None) in current frame, move on to next frame
            else:
                continue

            #max_actors = 5
            #if no_actors > max_actors:
            #    no_actors = max_actors
            #    rois_np = rois_np[:max_actors]
            #    temporal_rois_np = temporal_rois_np[:max_actors]

            # Might have issue of not using attention map because only predict action for 1 actor (memory issue)
            feed_dict = {updated_frames:cur_input_sequence, # only update last #action_freq frames
                         temporal_rois: temporal_rois_np,
                         temporal_roi_batch_indices: np.zeros(no_actors),
                         rois:rois_np, 
                         roi_batch_indices:np.arange(no_actors)}
            run_dict = {'pred_probs': pred_probs}

            if SHOW_CAMS:
                run_dict['cropped_frames'] = cropped_frames
                run_dict['final_i3d_feats'] =  act_detector.act_graph.get_collection('final_i3d_feats')[0]
                run_dict['cls_weights'] = act_detector.act_graph.get_collection('variables')[-2] # this is the kernel

            out_dict = act_detector.session.run(run_dict, feed_dict=feed_dict)
            probs = out_dict['pred_probs']

            # associate probs with actor ids
            print_top_k = 5
            for bb in range(no_actors):
                #act_probs = probs[bb]
                #order = np.argsort(act_probs)[::-1]
                #cur_actor_id = tracker.active_actors[bb]['actor_id']
                act_probs = probs[bb]
                order = np.argsort(act_probs)[::-1]
                cur_actor_id = tracker.active_actors[roi_idx]["actor_id"]
                #print(cur_actor_id == actor_id)
                #print("Person %i" % cur_actor_id)
                #print("act_probs: {}".format(act_probs))
                #print("order: {}".format(order))
                #print("tracker.active_actors[bb]: {}".format(tracker.active_actors[bb]))
                cur_results = []
                for pp in range(print_top_k):
                    #print('\t %s: %.3f' % (act.ACTION_STRINGS[order[pp]], act_probs[order[pp]]))
                    cur_results.append((act.ACTION_STRINGS[order[pp]], act_probs[order[pp]]))
                    csv_writer.writerow([video_time, cur_actor_id, act.ACTION_STRINGS[order[pp]], act_probs[order[pp]]])

                prob_dict[cur_actor_id] = cur_results
        
        if frame_cnt > 16:
            out_img = visualize_detection_results(tracker.frame_history[-16],
                                                  tracker.active_actors,
                                                  prob_dict=prob_dict,
                                                  teacher_id=matched_id)
            if SHOW_CAMS:
                if tracker.active_actors:
                    actor_indices = [ii for ii in range(no_actors) if tracker.active_actors[ii]['actor_id'] == actor_to_display]
                    if actor_indices:
                        out_img = visualize_cams(out_img, cur_input_sequence, out_dict, actor_indices[0])
                    else:
                        continue
                else:
                    continue
            if display: 
                cv2.imshow('results', out_img[:,:,::-1])
                cv2.waitKey(10)

            writer.append_data(out_img)

    #if not display:
    writer.close()
    csv_file.close()

def _prompt_user_input():
    teacher_id = input("Enter the id of the teacher (type None if teacher is not present in this frame): ")

    return teacher_id

def _check_teacher_in_frame(teacher_id):
    if teacher_id == "None" or teacher_id == "none":
        return False
    return True

def _check_valid_teacher_id(teacher_id, valid_actor_ids):
    try:
        teacher_id = int(teacher_id)

        if teacher_id in valid_actor_ids:
            return True
        else:
            return False
    except:
        return False

np.random.seed(10)
COLORS = np.random.randint(0, 255, [1000, 3])
def visualize_detection_results(img_np, active_actors, prob_dict=None, teacher_id=None):
    score_th = 0.30
    action_th = 0.20

    # copy the original image first
    disp_img = np.copy(img_np)
    H, W, C = img_np.shape
    #for ii in range(len(active_actors)):
    for ii in range(len(active_actors)):
        cur_actor = active_actors[ii]
        actor_id = cur_actor['actor_id']

        if teacher_id is not None:
            if actor_id != teacher_id:
                continue

        if prob_dict:
            cur_act_results = prob_dict[actor_id] if actor_id in prob_dict else []

        try:
            if len(cur_actor["all_boxes"]) > 0:
                cur_box, cur_score, cur_class = cur_actor['all_boxes'][-16], cur_actor['all_scores'][0], 1
            else:
                cur_box, cur_score, cur_class = cur_actor['all_boxes'][0], cur_actor['all_scores'][0], 1
        except IndexError:
            continue
        
        if cur_score < score_th: 
            continue

        top, left, bottom, right = cur_box

        left = int(W * left)
        right = int(W * right)

        top = int(H * top)
        bottom = int(H * bottom)

        conf = cur_score

        label = obj.OBJECT_STRINGS[cur_class]['name']
        message = '%s_%i: %% %.2f' % (label, actor_id,conf)

        if prob_dict:
            action_message_list = ["%s:%.3f" % (actres[0][0:7], actres[1]) for actres in cur_act_results if actres[1]>action_th]

        color = COLORS[actor_id]
        color = (int(color[0]), int(color[1]), int(color[2]))
        cv2.rectangle(disp_img, (left,top), (right,bottom), color, 3)

        font_size =  max(0.5,(right - left)/50.0/float(len(message)))
        cv2.rectangle(disp_img, (left, top-int(font_size*40)), (right,top), color, -1)
        cv2.putText(disp_img, message, (left, top-12), 0, font_size, (255-color[0], 255-color[1], 255-color[2]), 1)

        if prob_dict:
            #action message writing
            cv2.rectangle(disp_img, (left, top), (right,top+10*len(action_message_list)), color, -1)
            for aa, action_message in enumerate(action_message_list):
                offset = aa*10
                cv2.putText(disp_img, action_message, (left, top+5+offset), 0, 0.5, (255-color[0], 255-color[1], 255-color[2]), 1)

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
