import cv2
import numpy as np

video = cv2.VideoCapture("./video.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
print("frames per second =", fps)

seconds = np.linspace(0,12,20)
for second in seconds:
    frame_id = int(fps * second)
    print("frame id =", frame_id)

    video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = video.read()

# cv2.imshow("frame", frame)
# cv2.waitKey(0)
    cv2.imwrite(f"frame_{frame_id}.png", frame)
