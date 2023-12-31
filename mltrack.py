from ultralytics import YOLO
import cv2
from numpy import append as npappend
from numpy import ndarray as nd
import numpy as np

model=YOLO('yolov8m.pt')
model.to(0)
video_path = ".\dataset\\6p-c2_test.mp4"

cap = cv2.VideoCapture(video_path)

xyxys=[]
class_ids=[]
imgdata=[]
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Run YOLOv8 inference on the frame
        result = model.track(frame,persist=True,tracker='bytetrack.yaml',classes=0,device=0)
        # Visualize the results on the frame
        annotated_frame = result[0].plot()
        
        xyxys.append(result[0].boxes.xyxy.cpu().numpy())
        class_ids.append(result[0].boxes.cls.cpu().numpy())
        imgdata.append(result[0].orig_img)
        # Display the annotated frame
        #cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break
    
# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
np.savez("savedData",imgdata=imgdata,xyxys=xyxys,class_ids=class_ids)
print(xyxys[200],imgdata[200],class_ids[200])
# with open('tracking_results.json', 'w') as pkl:
#     pickle.dump(results, pkl)
