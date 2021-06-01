## What did I do in week 2 ?
# 4 main steps in MOT problem:
1. Get frame
2. Run detectors, obtain target detection frames.
3. Perform feature extraction
4. Calculate the similarity to calculate the matching.
5. Data association, assign target ID to each other.

# Simple Online and Realtime Tracking:
Kalman Filter algorithm divided into 2 processes: predict and update.
- Predict: when target moves, the target frame and speed of the previous frame are used to predict the target frame position and speed of the current frame.
- Update: The predicted value and observed value are linearly weighted to get the current state of the system prediction.

Hungarian algorithm is an assignment problem.

The overall process is:
- Kalman filter predicts trajectory Tracks.
- Use the Hungarian algorithm to match the predicted trajectory Tracks with the detections in the current frame (IOU matching)
- Kalman filter update.

# Deep SORT:
Plus: Matching Cascade

Matching Cascade divided into two parts:

### Part1: Cost matrix
- Calculating the similarity matrix (cost matrix) and gated matrix (limit the excessively large values in the cost matrix). 
- Cost matrix is used Cosine Similarity and motion similarity (Mahalanobis distance).

### Part2: Data association for cascade matching
- The matching process is a loop (max_age iterations, the default is 70). The trajectories from missing age = 0 to missing age = 70 are matched with Detections. 
- The trajectories that have not been lost are matched first, and the ones that are lost are more distant match back.
- Through this part of the processing, the occluded target can be retrieved again, reduce the number of ID switches.

The track of the target that temporarily disappears from the image cannot match the detection (in SORT, we say it's unmatch track)
## Objective: 
- When the occluded target reappears later, we should try to keep the ID assigned by the occluded target unchanged and reduce the number of ID Switch appearances. This requires cascading matching.


