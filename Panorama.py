import numpy as np
import cv2

class Stitcher:
    def __init__(self):
        # Initialize the feature extractor, using SIFT for detecting keypoints and descriptors
        self.feature_extractor = cv2.SIFT_create()

    def stitch(self, img_left, img_right):
        # Enhance image contrast by equalizing the histogram of both images
        img_left = self.equalize_histogram_color(img_left)
        img_right = self.equalize_histogram_color(img_right)

        # STEP - 1 Detect keypoints and compute descriptors for both images
        keypoints_l, descriptors_l = self.compute_descriptors(img_left)
        keypoints_r, descriptors_r = self.compute_descriptors(img_right)

        # STEP - 2 Match keypoints between the two images
        matches = self.matching(descriptors_l, descriptors_r)

        # STEP - 3 Draw lines between matching keypoints for visualization
        self.draw_matches(img_left, img_right, matches, keypoints_l, keypoints_r)

        # STEP - 4 Find the transformation matrix (homography) to align the images
        M = self.find_homography(keypoints_l, keypoints_r, matches)

        # STEP - 5 Warp images to the same perspective and stitch them together
        result_image = self.warping(img_right, img_left, M)

        # STEP - 6 Show the resulting stitched image
        cv2.imshow('Result', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def compute_descriptors(self, img):
        # Compute keypoints and descriptors for an image using the SIFT feature extractor
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.feature_extractor.detectAndCompute(grey, None)
        return keypoints, descriptors
    def matching(self, descriptors_l, descriptors_r):
        matches = []
        ratio_threshold = 0.75  # Lowe's ratio test threshold
        distance_threshold = 100  # Optional distance threshold to filter out distant matches

        # Iterate through each descriptor in the left image
        for i, desc_l in enumerate(descriptors_l):
            # Calculate the distance between desc_l and all descriptors in the right image
            distances = np.linalg.norm(descriptors_r - desc_l, axis=1)

            # Sort the distances and get the indices of the two smallest distances
            sorted_indices = np.argsort(distances)
            closest_idx = sorted_indices[0]
            second_closest_idx = sorted_indices[1]

            # Apply Lowe's ratio test
            if distances[closest_idx] < ratio_threshold * distances[second_closest_idx] and distances[
                closest_idx] < distance_threshold:
                match = cv2.DMatch(i, closest_idx, distances[closest_idx])
                matches.append(match)

        return matches
    def draw_matches(self, img_left, img_right, matches, keypoints_l, keypoints_r):
        img_with_correspondences = cv2.drawMatches(img_left, keypoints_l, img_right, keypoints_r, matches, None)
        cv2.imshow('correspondences', img_with_correspondences)
        cv2.waitKey(0)

    def find_homography(self, keypoints_l, keypoints_r, matches):
        '''
        Manually find the homography matrix between two images using DLT.
        '''
        if len(matches) < 4:
            raise ValueError("At least 4 matches are required to compute a homography.")

        # Arrays to store the corresponding points
        src1_pts = np.float32([keypoints_l[m.queryIdx].pt for m in matches])
        src2_pts = np.float32([keypoints_r[m.trainIdx].pt for m in matches])

        # Construct the matrix A for DLT
        A = []
        for i in range(len(matches)):
            x1, y1 = src1_pts[i]
            x2, y2 = src2_pts[i]
            A.append([-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2])
            A.append([0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2])

        A = np.array(A)

        # Compute SVD of A
        _, _, Vh = np.linalg.svd(A)

        # The homography is the last row of V (or the last column of V transposed)
        H = Vh[-1].reshape(3, 3)

        # Normalize and return the homography matrix
        return H / H[2, 2]

if __name__ == "__main__":
    img_left = cv2.imread('img_left.jpg')
    img_right = cv2.imread('img_right.jpg')

    stitcher = Stitcher()
    result = stitcher.stitch(img_left, img_right)