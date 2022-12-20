from lib import *
from kalmanFilterAlgo import KalmanBBoxTracker
from dataAssociation import hungaryAssociate

class SORT(object):
    def __init__(self, max_age = 1, min_hits = 3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
    
    def update(self, dets):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t, trk in enumerate(trks):
            # get the predict bbox
            pos = self.trackers[t].predict()[0] 
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        # row: detection, col: trackers
        # filter and delete invalid detections > appply hungarian algorithm
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_detections, unmatched_trackers = hungaryAssociate(dets, trks)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trackers:
                # matched[:, 0] -> detections, matched[:, 1] -> tracks
                # get the matched detection with related tracker
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                # Kalman Filter update function
                trk.update(dets[d, :][0])
        
        # create and initialize new trackers for unmatch detections
        for i in unmatched_detections:
            trk = KalmanBBoxTracker(dets[i, :])
            self.trackers.append(trk)
        
        num_trackers = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))

            num_trackers -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(num_trackers)
        
        # return list track
        if len(ret) > 0:
            return np.concatenate(ret)
        # return empty list track
        return np.empty((0, 5))