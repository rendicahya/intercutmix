from pathlib import Path

import cv2
import numpy as np

mask_path = Path(
    "/nas.dbms/randy/projects/intercutmix/data/ucf101/REPP/actorcutmix/mask/all-mpnet-base-v2/0.5/Archery/v_Archery_g01_c01.npz"
)
kernel_size = 65
mask_bundle = np.load(mask_path)["arr_0"]
blurred = cv2.GaussianBlur(mask_bundle[0], (kernel_size, kernel_size), 0)

cv2.imwrite("smooth-mask.png", blurred)
