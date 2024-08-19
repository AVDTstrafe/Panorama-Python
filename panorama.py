import numpy as np
import cv2

class Stitcher:
    def __init__(self):
        self.feature_extractor = cv2.SIFT_create()
        
    def stitch(self, img_left, img_right):
        
        # STEP 1 - collects keypoints from both images
        keypoints_l, descriptors_l = self.compute_descriptors(img_left)
        keypoints_r, descriptors_r = self.compute_descriptors(img_right)

        # STEP 2 - finds keypoints that match and stores them
        matches = self.matching(descriptors_l, descriptors_r)
        print("matches found!")
        print("Number of matching correspondences selected:", len(matches))

        # STEP 3 - draws lines between matches to make them easier to compare
        self.draw_matches(img_left, img_right, matches, keypoints_l, keypoints_r)

    def compute_descriptors(self, img):
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.feature_extractor.detectAndCompute(grey, None) #using AKAZE to find features on grey img
        return keypoints, descriptors #

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
        '''
            Connect correspondences between images with lines and draw these
            lines
        '''
        img_with_correspondences = cv2.drawMatches(img_left, keypoints_l, img_right, keypoints_r, matches, None)
        cv2.imshow('correspondences', img_with_correspondences)
        cv2.waitKey(0)

if __name__ == "__main__":
    # Load the image files
    img_left = cv2.imread('img_left.jpg')
    img_right = cv2.imread('img_right.jpg')

    # Initialize stitcher
    stitcher = Stitcher()

    # Perform stitching
    result = stitcher.stitch(img_left, img_right)

    # Display the result
    #cv2.imshow('Result', result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()