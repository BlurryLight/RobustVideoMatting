# import the opencv library
import cv2
import numpy as np
import time
import onnxruntime as ort


sess = ort.InferenceSession('rvm_mobilenetv3_fp16.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

width = 1280
height = 720
vid = cv2.VideoCapture(0)
vid.set(3, width)
vid.set(4, height)
img = cv2.imread('test.jpg', cv2.IMREAD_ANYCOLOR)
background = cv2.resize(img, (width, height))
background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
background = np.transpose(background, [2, 0, 1])
background = background.astype(np.float16)
background = background / 255.0

io = sess.io_binding()
rec = [ ort.OrtValue.ortvalue_from_numpy(np.zeros([1, 1, 1, 1], dtype=np.float16), 'cuda') ] * 4
downsample_ratio = ort.OrtValue.ortvalue_from_numpy(np.asarray([0.25], dtype=np.float32), 'cuda')
for name in ['fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o']:
    io.bind_output(name, 'cuda')

while(True):
    # Capture the video frame
    # by frame
    t0 = time.time()
    ret, frame = vid.read()
    rgb_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_data = np.transpose(rgb_data, [2, 0, 1])
    rgb_data = rgb_data.astype(np.float16)
    rgb_data = rgb_data/255.0
    rgb_data = np.expand_dims(rgb_data, axis=0)
    io.bind_cpu_input('src', rgb_data)
    io.bind_ortvalue_input('r1i', rec[0])
    io.bind_ortvalue_input('r2i', rec[1])
    io.bind_ortvalue_input('r3i', rec[2])
    io.bind_ortvalue_input('r4i', rec[3])
    io.bind_ortvalue_input('downsample_ratio', downsample_ratio)
    sess.run_with_iobinding(io)
    fgr, pha, *rec = io.get_outputs()
    fgr = fgr.numpy()
    pha = pha.numpy()

    com = fgr * pha + background * (1 - pha)
    squeezed_com = np.squeeze(com, axis=0)
    output = np.transpose(squeezed_com, [1, 2, 0])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    t1 = time.time()
    print("FPS is ", 1.0 / (t1 - t0))

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
