# import the opencv library
import cv2
import numpy as np
import time
import onnxruntime as ort

print(ort.get_device())
sess = ort.InferenceSession('rvm_mobilenetv3_fp32.onnx', providers=[
                            'CUDAExecutionProvider', 'CPUExecutionProvider'])
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

downsample_ratio = np.array([0.25], dtype=np.float32)
rec = [np.zeros([1, 1, 1, 1], dtype=np.float32)] * 4
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
    # rgb_frame = torch.from_numpy(rgb_data)
    # fgr, pha, *rec = model(rgb_frame.cuda(), *rec, downsample_ratio)
    fgr, pha, *rec = sess.run([], {
        'src': rgb_data,
        'r1i': rec[0],
        'r2i': rec[1],
        'r3i': rec[2],
        'r4i': rec[3],
        'downsample_ratio': downsample_ratio
    })
    com = fgr * pha + background * (1 - pha)
    # print(com.shape)
    squeezed_com = np.squeeze(com, axis=0)
    # print(squeezed_com.shape)
    output = np.transpose(squeezed_com, [1, 2, 0])
    # print(output.shape)
    bgr_frame = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imshow('frame', bgr_frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
    t1 = time.time()
    print("FPS is ", 1.0 / (t1 - t0))

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
