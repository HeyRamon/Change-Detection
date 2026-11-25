## Course code
  - get_gaussian_kernel(), compute_gradient() (Edges / Corners tutorials) -> preprocess.py
  - non_maxima_suppression(), Harris cornerness (Corners tutorial) -> align.py
  - match_features() (Features tutorial) -> align.py

## Mini projects
  - edge_linking() within MP2 -> connected_components() in segment.py
  - warp() within MP4 -> warp() in align.py
  - cv2.filter2D() within MP1 -> convolution() in preprocess.py

## Libraries
  - OpenCV: imread(), imwrite(), imshow(), waitKey(), cvtColor(), filter2D(), Canny(), SIFT_create(), detectAndCompute(), KeyPoint(), DMatch(), drawMatches(), line(), putText(), VideoCapture(), namedWindow(), createTrackbar(), getTrackbarPos(), setTrackbarPos(), selectROIs(), getTickCount()
  - NumPy: linalg.svd(), linalg.norm(), argsort(), random.choice()
  - Matplotlib: imshow(), show()
  - PyTorch / torchvision: vgg16() (cnn_features.py)

## Data
  - Before/after photos and ground-truth masks were made by our team