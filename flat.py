from UndistortFishEye import fisheye_module
import cv2 as cv
import glob
import os

input_dir = "input/"
calibration_file_path = "calibration.txt"
output_dir = "output/"


def main():
    if not os.path.isdir(output_dir):
        print("Creating output directory...")
        os.mkdir(output_dir)

    k, d, dims = fisheye_module.load_calibration(calibration_file_path)
    number = 0

    for image_path in glob.glob(input_dir + "*.jpg"):
        img = cv.imread(image_path)
        undistorted_img = fisheye_module.undistort(img, k, d, dims)
        cv.imwrite(output_dir + image_path.split("\\")[-1], undistorted_img)
        number += 1
        print("Processed images: " + str(number))

    print("Done.")


if __name__ == '__main__':
    main()
