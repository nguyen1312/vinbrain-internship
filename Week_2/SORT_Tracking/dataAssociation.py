from lib import *
from iou import *

# Assigns detections to tracked object with Hungarian algorithm
# return match, unmatch_detection. unmatch_track
# match -> (detection_idx, track_idx)
# unmatch_detection -> ([detection_idx])
# unmatch_track -> (bbox_coordinate, id)
def hungaryAssociate(detections, trackers, iou_threshold = 0.3):
    if len(trackers) == 0:
        return  (
                    np.empty((0, 2), dtype = int),
                    np.arange(len(detections)),
                    np.empty((0, 5), dtype = int)
                )
    # row: detection, col: trackers
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype = np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    
    # apply Hungari Algo to maximize sum of iou score
    # match_indices -> [(detection_idxJ, track_idxI)] -> shape(min(len(det),len(trk)), 2)       
    # if len(det) > len(trk) -> redundancy detection -> unmatch detection
    # if len(trk) > len(det) -> redundancy track -> unmatch track
    matched_indices = linear_assignment(- iou_matrix)

    # records unmatched detection indices
    # init
    unmatched_detections = []
    for d, det in enumerate(detections):
        list_detection_idx = matched_indices[:, 0] # cot dau tien
        if d not in list_detection_idx:
            unmatched_detections.append(d)
    
    # records unmatched trackers indices
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        list_track_idx = matched_indices[:, 1] # cot thu hai
        if t not in list_track_idx:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    # match_indices -> [(detection_idxJ, track_idxI)] -> shape(min(len(det),len(trk)), 2)       
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype = int)
    else:
        matches = np.concatenate(matches, axis = 0)
    return  (
                matches,
                np.array(unmatched_detections),
                np.array(unmatched_trackers)
            )
