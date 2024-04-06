import cv2

video = cv2.VideoCapture("./short_video.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
print("frames per second =", fps)

seconds = 6
frame_id = int(fps * seconds)
print("frame id =", frame_id)

video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
ret, frame = video.read()

cv2.imshow("frame", frame)
cv2.waitKey(0)
cv2.imwrite(f"frame_{frame_id}.png", frame)
