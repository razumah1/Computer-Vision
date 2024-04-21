import cv2
import depthai as dai
from datetime import datetime


# Start defining a pipeline
pipeline = dai.Pipeline()


# Define a source - color camera
cam = pipeline.create(dai.node.ColorCamera)


# Script node
script = pipeline.create(dai.node.Script)
script.setScript("""
    import time
    ctrl = CameraControl()
    ctrl.setCaptureStill(True)
    while True:
        time.sleep(0.033)
        node.io['out'].send(ctrl)
""")


# XLinkOut
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName('camera')


# Connections
script.outputs['out'].link(cam.inputControl)
cam.still.link(xout.input)


# Connect to device with pipeline
with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:
   while True:
    img = device.getOutputQueue("camera").get()
    cv2.imshow('camera',img.getCvFrame())
    key = cv2.waitKey(1)
        
    if key == ord('q'):
      break
        
    elif key == ord('c'):
      filename = str(datetime.now()).replace(':','-')
      cv2.imwrite(f'./{filename}.jpg',img.getCvFrame())
