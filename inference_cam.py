# import the opencv library
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch
from model import MattingNetwork
import numpy as np
import time

model = MattingNetwork('mobilenetv3').eval().cuda()  # or "resnet50"
model.load_state_dict(torch.load('rvm_mobilenetv3.pth'))
# define a video capture object
width = 1280
height = 720
vid = cv2.VideoCapture(0)
vid.set(3, width)
vid.set(4, height)
# Green background.
# background = torch.tensor([.47, 1, .6]).view(3, 1, 1).cuda()
img = cv2.imread('test.jpg', cv2.IMREAD_ANYCOLOR)
background = cv2.resize(img, (width, height))
background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
background = np.transpose(background, [2, 0, 1])
background = background.astype(np.float32)
background = background / 255.0
background = torch.from_numpy(background).cuda()
# Initial recurrent states.
rec = [None] * 4
# Adjust based on your video.
downsample_ratio = 0.25

while(True):
    # Capture the video frame
    # by frame
    t0 = time.time()
    ret, frame = vid.read()
    rgb_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_data = np.transpose(rgb_data, [2, 0, 1])
    rgb_data = rgb_data.astype(np.float32)
    rgb_data = rgb_data/255.0
    rgb_data = np.expand_dims(rgb_data, axis=0)
    rgb_frame = torch.from_numpy(rgb_data)
    with torch.no_grad():
        fgr, pha, *rec = model(rgb_frame.cuda(), *rec, downsample_ratio)
        # Composite to green background.
        com = fgr * pha + background * (1 - pha)

    output = com.squeeze().cpu().numpy()
    output = np.transpose(output, [1, 2, 0])
    bgr_frame = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    # Display the resulting frame
    cv2.imshow('frame', bgr_frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    t1 = time.time()
    print("FPS is ", 1.0 / (t1 - t0))

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
