import cv2
import numpy as np
import os
from tqdm import tqdm
import imutils
import json

import multiprocessing
from multiprocessing import Pool

class EmptySegMaskError(Exception):
    pass

def preprocess_clip(args):
    cn, segments_dir, frames_dir, \
    preprocessed_segments_dir, preprocessed_frames_dir = args

    seg_clip_prefix = os.path.join(segments_dir, cn)
    frame_clip_prefix = os.path.join(frames_dir, cn)
    preprocessed_seg_clip_prefix = os.path.join(preprocessed_segments_dir, cn)
    preprocessed_frames_clip_prefix = os.path.join(preprocessed_frames_dir, cn)

    os.mkdir(preprocessed_seg_clip_prefix)
    os.mkdir(preprocessed_frames_clip_prefix)

    filenames = sorted([fn for fn in os.listdir(seg_clip_prefix) if not fn.startswith(".")])

    bounding_box_positions = []
    interp_idxes = []
    rgb_imgs = []
    seg_imgs = []
    gravity_center = []
    for idx, fn in enumerate(filenames):
        rgb_img = cv2.imread(os.path.join(frame_clip_prefix, fn) ,cv2.IMREAD_COLOR)
        seg_img = cv2.imread(os.path.join(seg_clip_prefix, fn) ,cv2.IMREAD_COLOR)

        # get the normalized gravity center
        gravity_x, gravity_y = get_gravity_centor(seg_img)
        gravity_center.append((gravity_x,gravity_y))

        supress_non_hand_pixels(seg_img)

        rgb_imgs.append(rgb_img)
        seg_imgs.append(seg_img)

        try:
            corner, width, height = get_bounding_box(seg_img)
        except EmptySegMaskError:
            # empty seg mask, should infer bounding box info from neighbours
            bounding_box_positions.append(None)
            interp_idxes.append(idx)
            continue

        bounding_box_positions.append((corner, width, height))


    # fill in the empty bounding boxes
    if interp_idxes:
        # flatten bounding_box_positions
        cXs = []
        cYs = []
        widths = []
        heights = []
        valid_idxs = []
        for idx, pos in enumerate(bounding_box_positions):
            if pos is not None:
                cXs.append(pos[0][0])
                cYs.append(pos[0][1])
                widths.append(pos[1])
                heights.append(pos[2])
                valid_idxs.append(idx)
        # interp corner location
        intp_cXs = np.interp(interp_idxes, valid_idxs, cXs)
        intp_cYs = np.interp(interp_idxes, valid_idxs, cYs)
        intp_widths = np.interp(interp_idxes, valid_idxs, widths)
        intp_heights = np.interp(interp_idxes, valid_idxs, heights)
        counter = 0
        for idx, pos in enumerate(bounding_box_positions):
            if pos is None:
                bounding_box_positions[idx] = ((intp_cXs[counter], intp_cYs[counter]),
                                                intp_widths[counter],
                                                intp_heights[counter])
                counter += 1
    
    preprocessed_clip_positions_dict = {}
    for idx, pos in enumerate(bounding_box_positions):
        (corner, width, height) = pos
        fn = filenames[idx]
        rgb_img = rgb_imgs[idx]
        seg_img = seg_imgs[idx]

        pos_gravity = gravity_center[idx]

        preprocessed_clip_positions_dict[fn] = (pos_gravity, width, height)

        rgb_img, seg_img = crop_images_by_bounding_box([rgb_img, seg_img], corner, width, height)
        cv2.imwrite(os.path.join(preprocessed_frames_clip_prefix, fn), rgb_img)
        cv2.imwrite(os.path.join(preprocessed_seg_clip_prefix, fn), seg_img)

    # change the position of bounding box corner into gravity center
    # for idx, pos_gravity in enumerate(gravity_center):
    #     (gravity_x, gravity_y) = pos_gravity
    #     preprocessed_clip_positions_dict[fn][0][0] = gravity_x
    #     preprocessed_clip_positions_dict[fn][0][1] = gravity_y

    return preprocessed_clip_positions_dict

def supress_non_hand_pixels(img):
    height, width, _ = img.shape
    for i in range(height):
        for j in range(width):
            Bv, Gv, Rv = img[i,j]
            if Rv > 200 or Gv < 200 or Bv < 200:
                img[i,j] = [0, 0 ,0]

def get_bounding_box(img, binarize_thresh=10, contour_size_filter_thresh=0.1):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binarized = cv2.threshold(img_grey, binarize_thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
    contours, _ = cv2.findContours(img_binarized, 1, 2)

    contours_areas = [cv2.contourArea(cnt) for cnt in contours]

    if not contours_areas:
        # empty segmentation mask
        raise EmptySegMaskError

    max_cnt_area = max(contours_areas)

    filtered_contours = []
    for idx, cnt in enumerate(contours):
        if contours_areas[idx] >= max_cnt_area * contour_size_filter_thresh:
            filtered_contours.append(cnt)
    concat_contours = np.concatenate(filtered_contours)

    x,y,w,h = cv2.boundingRect(concat_contours)
    cX = x + w / 2
    cY = y + h / 2
    max_wh = max(w, h) + 20

    return (int(cX - max_wh/2),int(cY - max_wh/2)), max_wh, max_wh

def crop_images_by_bounding_box(imgs, corner, width, height, output_width=224, output_height=224):
    cX, cY = corner
    cropped_images = []
    for img in imgs:
        # pad image if the bounding box extends out of bounds
        if len(img.shape) == 3:
            y_size, x_size, _ = img.shape
        else:
            y_size, x_size = img.shape
        xbef_pad_size = int(max(0, -cX))
        xaft_pad_size = int(max(0, cX + width - x_size))
        ybef_pad_size = int(max(0, -cY))
        yaft_pad_size = int(max(0, cY + height - y_size))
        
        paddings = [(ybef_pad_size, yaft_pad_size), 
                    (xbef_pad_size, xaft_pad_size)]
        if len(img.shape) == 3:
            paddings.append((0, 0))

        padded_img = np.pad(img, paddings)
        cropped_images.append(cv2.resize(
            padded_img[int(cY+ybef_pad_size) : int(cY+height+ybef_pad_size),
                int(cX+xbef_pad_size) : int(cX+width+xbef_pad_size)],
            (output_width, output_height),
            interpolation=cv2.INTER_LINEAR))
    return cropped_images

def get_gravity_centor(img, binarize_thresh=20, position_encoding_para=0.5):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binarized = cv2.threshold(img_grey, binarize_thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
    # img_binarized = cv2.bitwise_not(img_binarized)
    contours = cv2.findContours(img_binarized, 1, 2)
    contours = imutils.grab_contours(contours)
    gravity_x = 0
    gravity_y = 0
    if len(contours) == 0:
        # contours not found
        gravity_x = 0
        gravity_y = 0
        return gravity_x,gravity_y
    else:
        largest_contour_length = 0
        for c in contours:
            if len(c) <= largest_contour_length:
                continue
            largest_contour_length = len(c)
            M = cv2.moments(c)
            if M["m00"] == 0:
                gravity_x = 0
                gravity_y = 0
                continue
            gravity_x = int(M["m10"] / M["m00"])
            gravity_y = int(M["m01"] / M["m00"])
            cv2.waitKey(0)
    # Normalize the gravity
    gravity_x = position_encoding_para*gravity_x/(img.shape[0]-1)
    gravity_y = position_encoding_para*gravity_y/(img.shape[1]-1)
    return gravity_x, gravity_y

def preprocess_ipn_dataset(dataset_prefix, frames_dir="frames", segs_dir="segment"):
    preprocessed_dir = os.path.join(dataset_prefix, "preprocessed")
    preprocessed_frames = os.path.join(preprocessed_dir, "frames")
    clip_position_path = os.path.join(preprocessed_dir, "clip_positions.json")

    if os.path.exists(preprocessed_dir):
        return preprocessed_dir, preprocessed_frames, clip_position_path
    
    print("Preprocessing IPN dataset...")
    os.mkdir(preprocessed_dir)

    frames_dir = os.path.join(dataset_prefix, "frames")
    segments_dir = os.path.join(dataset_prefix, "segment")
    preprocessed_frames_dir = os.path.join(preprocessed_dir, "frames")
    preprocessed_segments_dir = os.path.join(preprocessed_dir, "segment")

    os.mkdir(preprocessed_frames_dir)
    os.mkdir(preprocessed_segments_dir)

    clip_names = [fn for fn in os.listdir(segments_dir) if not fn.startswith(".")]

    args = [(cn, segments_dir, frames_dir,
            preprocessed_segments_dir, preprocessed_frames_dir) for cn in clip_names]

    preprocessed_clip_positions = {}

    print("Launching parallel processing jobs...")
    with Pool(multiprocessing.cpu_count()) as p:
        result = list(tqdm(p.imap(preprocess_clip, args), total=len(args)))

    print("Joining clip level dicts...")
    for clip_dict in tqdm(result):
        preprocessed_clip_positions.update(clip_dict)

    with open(clip_position_path, "w") as f:
        json.dump(preprocessed_clip_positions, f)

    return preprocessed_dir, preprocessed_frames, clip_position_path