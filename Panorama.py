import numpy as np
import cv2

class Stitcher:
    def __init__(self):
        # Initialize the feature extractor
        self.feature_extractor = cv2.AKAZE_create()

    def stitch(self, img_left, img_right):

        print("panorama underway this will take a couple of minutes")

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

        # STEP - 6 Return the stitched image
        return result_image

    def compute_descriptors(self, img):
        # Compute keypoints and descriptors for an image
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.feature_extractor.detectAndCompute(grey, None)
        return keypoints, descriptors

    def matching(self, descriptors_l, descriptors_r):
        matches = []
        ratio_threshold = 0.75  # ratio test threshold
        distance_threshold = 100  # distance threshold

        # Iterate through each descriptor in the left image
        for i, desc_l in enumerate(descriptors_l):
            # Calculate the distance between desc_l and all descriptors in the right image
            distances = np.linalg.norm(descriptors_r - desc_l, axis=1)

            # Sort the distances and get the indices of the two smallest distances
            sorted_indices = np.argsort(distances)
            closest_idx = sorted_indices[0]
            second_closest_idx = sorted_indices[1]

            # Apply thresholds
            if distances[closest_idx] < ratio_threshold * distances[second_closest_idx] and distances[
                closest_idx] < distance_threshold:
                match = cv2.DMatch(i, closest_idx, distances[closest_idx])
                matches.append(match)

        return matches

    def draw_matches(self, img_left, img_right, matches, keypoints_l, keypoints_r):
        img_with_correspondences = cv2.drawMatches(img_left, keypoints_l, img_right, keypoints_r, matches, None)
        cv2.imshow('Correspondences', img_with_correspondences)
        cv2.waitKey(0)

    def find_homography(self, keypoints_l, keypoints_r, matches):
        if len(matches) < 4:
            print("Not enough matches to compute homography.")
            return None

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

    def warping(self, img_left, img_right, H):
        h1, w1 = img_left.shape[:2]
        h2, w2 = img_right.shape[:2]

        img1_dims = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        img2_dims_temp = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

        img2_dims = cv2.perspectiveTransform(img2_dims_temp, H)
        result_dims = np.concatenate((img1_dims, img2_dims), axis=0)

        [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

        transform_dist = [-x_min, -y_min]
        transform_array = np.array([[1, 0, transform_dist[0]], [0, 1, transform_dist[1]], [0, 0, 1]])

        result = cv2.warpPerspective(img_right, transform_array.dot(H), (x_max - x_min, y_max - y_min))
        result[transform_dist[1]:h1 + transform_dist[1], transform_dist[0]:w1 + transform_dist[0]] = img_left

        return result

    def equalize_histogram_color(self, img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img

if __name__ == "__main__":
    img_left = cv2.imread('img_left.jpg')
    img_right = cv2.imread('img_right.jpg')

    stitcher = Stitcher()
    result = stitcher.stitch(img_left, img_right)

    cv2.imshow('Stitched Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img_left = cv2.imread('IL.jpg')
    img_right = cv2.imread('IR.jpg')

    stitcher = Stitcher()
    result = stitcher.stitch(img_left, img_right)

    cv2.imshow('Stitched Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()