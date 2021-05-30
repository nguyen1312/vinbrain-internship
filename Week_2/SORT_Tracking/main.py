from lib import *
from detectionYoloV5 import detectImage
from sort import SORT
from inference import inferenceTrackFromFrame
if __name__ == "__main__":
    # path_to_image = osp.relpath("./sample_image/sample.jpg")
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    # tracker = SORT()

    # image = cv2.imread(path_to_image)
    # track_img = inferenceTrackFromFrame(tracker, image, model)
    # track_img = cv2.resize(track_img, dsize=(700, 700))
    # cv2.imshow("Result", track_img)
    # cv2.waitKey(0) 


    # detections = model(path_to_image)
    # tensorBBox = detections.xyxy[0].detach().numpy()
    # bboxPerson = tensorBBox[np.where(tensorBBox[:, -1] == 0)]
    # print(tracker.update(bboxPerson))

    # tracker = SORT()
    # cap = cv2.VideoCapture('pedestrian-1.mp4') # read video
    # while cap.isOpened():
    #     ret, image = cap.read()
    #     if not ret:
    #     break
    #     track_img = inferenceTrackFromFrame(tracker, image, model)
    #     track_img = cv2.resize(track_img, dsize=(50, 50))
    #     # track_img = Image.fromarray(track_img).resize((300,300))
    #     clear_output()
    #     cv2_imshow(track_img)
        
    # cv2.destroyAllWindows()
    # cap.release()

    path_to_video = osp.relpath("./sample_video/pedestrian.mp4")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    tracker = SORT()
    cap = cv2.VideoCapture(path_to_video) # read video
    res_arr = []
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        track_img = inferenceTrackFromFrame(tracker, image, model)
        track_img = cv2.resize(track_img, dsize=(700, 700))
        res_arr.append(track_img)
    cv2.destroyAllWindows()
    cap.release()

    video = cv2.VideoWriter('res.mp4',-1, 1, (700, 700))
    for img in res_arr:
        video.write(img)

    # url = upload_public('res.mp4')   
    # HTML(f"""<video src={url} width=500 controls/>""")


    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture('res.mp4')
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    # Read until video is completed
    while(cap.isOpened()):
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            cv2.imshow('Frame',frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else: 
            break

    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()