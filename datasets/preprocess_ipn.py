import cv2
import numpy as np
import os
from tqdm import tqdm

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
        cropped_images.append(cv2.resize(img[cY:cY+height, cX:cX+width], (output_width, output_height), interpolation=cv2.INTER_LINEAR))
    return cropped_images

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

    clip_names = [fn for fn in os.listdir(segments_dir) if not fn.startswith(".")]

    preprocessed_clip_positions = {}

    for cn in tqdm(clip_names):
        seg_clip_prefix = os.path.join(segments_dir, cn)
        frame_clip_prefix = os.path.join(frames_dir, cn)
        preprocessed_seg_clip_prefix = os.path.join(preprocessed_segments_dir, cn)
        preprocessed_frames_clip_prefix = os.path.join(preprocessed_segments_dir, cn)

        os.mkdir(preprocessed_seg_clip_prefix)
        os.mkdir(preprocessed_frames_clip_prefix)

        filenames = [fn for fn in os.listdir(seg_clip_prefix) if not fn.startswith(".")]

        os.mkdir()
        for fn in tqdm(filenames):
            rgb_img = cv2.imread(os.path.join(frame_clip_prefix, fn) ,cv2.IMREAD_COLOR)
            seg_img = cv2.imread(os.path.join(seg_clip_prefix, fn) ,cv2.IMREAD_COLOR)
            supress_non_hand_pixels(seg_img)
            corner, width, height = get_bounding_box(seg_img)
            preprocessed_clip_positions[fn] = (corner, width, height)
            rgb_img, seg_img = crop_images_by_bounding_box([rgb_img, seg_img], corner, width, height)
            cv2.imwrite(os.path.join(preprocessed_frames_clip_prefix, fn), rgb_img)
            cv2.imwrite(os.path.join(preprocessed_seg_clip_prefix, fn), seg_img)
    with open(clip_position_path, "w") as f:
        json.dump(preprocessed_clip_positions, f)
    return preprocessed_dir, preprocessed_frames, clip_position_path

