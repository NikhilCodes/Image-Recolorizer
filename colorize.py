import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('model_dir/my_model.h5')

if len(sys.argv) == 2:
    img_file_name = sys.argv[1]
else:
    sys.stderr.write("\n\nInvalid Arguments!\nRequired Syntax:\n\tpython colorize.py file_name.jpg\n\n")
    sys.exit(-1)

img = cv2.resize(cv2.imread(img_file_name), (300, 250))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255
img = np.expand_dims(img, axis=-1)


out = model.predict(np.array([img]))[0]
out = cv2.resize(out, (512, 410))
while True:
    cv2.imshow("Output", out)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
