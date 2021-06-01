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
