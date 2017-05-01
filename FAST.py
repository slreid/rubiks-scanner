# Steven Stetzler
# Project 2
# panorama_stitching.py

import cv2
from matplotlib import pyplot as plt
import numpy as np
import skimage, skimage.io, scipy.ndimage.filters, math, scipy.signal, skimage.color, skimage.feature, random


# This function applies a homography to the coordinate (x, y), producing (x_h, y_h)
def apply_homography(h, x, y):
    x_h = h[0, 0] * x + h[0, 1] * y + h[0, 2]
    y_h = h[1, 0] * x + h[1, 1] * y + h[1, 2]
    d_h = h[2, 0] * x + h[2, 1] * y + 1
    x_h /= d_h
    y_h /= d_h
    return x_h, y_h


# This function takes in a list of 4 points (x_a, y_a) from image A and 4 points (x_b, y_,) from image B
# and fits a homography exactly to those point
def fit_homography(points_a, points_b):
    A = np.zeros([8, 8])
    b = np.zeros(8)
    for i in range(0, 4):
        A[i, 0] = points_b[i][0]
        A[i, 1] = points_b[i][1]
        A[i, 2] = 1
        A[i, 6] = -points_a[i][0] * points_b[i][0]
        A[i, 7] = -points_a[i][0] * points_b[i][1]
        A[i + 4, 3] = points_b[i][0]
        A[i + 4, 4] = points_b[i][1]
        A[i + 4, 5] = 1
        A[i + 4, 6] = -points_a[i][1] * points_b[i][0]
        A[i + 4, 7] = -points_a[i][1] * points_b[i][1]
        b[i] = points_a[i][0]
        b[i + 4] = points_a[i][1]
    # print "Constructed matrix:", A
    params = np.linalg.lstsq(A, b)
    #     print "Params: ", params[0]
    H = np.zeros([3, 3])
    for i in range(0, 3):
        for j in range(0, 3):
            if i == 2 and j == 2:
                H[i, j] = 1
            else:
                H[i, j] = params[0][i * 3 + j]
    return H


# This function returns the images a and a stitched and blended together based on the homography H
def composite_warped(a, b, H):
    # "Warp images a and b to a's coordinate system using the homography H which maps b coordinates to a coordinates."
    out_shape = (a.shape[0], 2 * a.shape[1])  # Output image (height, width)
    p = skimage.transform.ProjectiveTransform(np.linalg.inv(H))  # Inverse of homography (used for inverse warping)
    bwarp = skimage.transform.warp(b, p, output_shape=out_shape)  # Inverse warp b to a coords
    plt.imshow(bwarp)
    bvalid = np.zeros(b.shape, 'uint8')  # Establish a region of interior pixels in b
    bvalid[1:-1, 1:-1, :] = 255
    bmask = skimage.transform.warp(bvalid, p, output_shape=out_shape)  # Inverse warp interior pixel region to a coords
    avalid = np.zeros(a.shape, 'uint8')
    avalid[1:-1, 1:-1, :] = 255
    apad = np.hstack((skimage.img_as_float(a), np.zeros(a.shape)))  # Pad a with black pixels on the right
    # Compute the distance transform of the padded A and B images
    dist_a = scipy.ndimage.morphology.distance_transform_edt(skimage.color.rgb2gray(apad))
    dist_b = scipy.ndimage.morphology.distance_transform_edt(skimage.color.rgb2gray(bmask))
    out_image = np.where(bmask == 1.0, bwarp, apad)
    for y in range(0, out_shape[0]):
        for x in range(0, out_shape[1]):
            if (dist_a[y, x] != 0) and (dist_b[y, x] != 0):
                # Compute the alpha value at each area in the overlapping region based on the distance
                # values of each pixel in A and B
                # The alpha determines how much of each image (the background or the foreground) should contribute
                # to the final image
                alpha = dist_a[y, x] / (dist_a[y, x] + dist_b[y, x])
                # Blend the background and the foreground together using the alpha computed
                out_image[y, x] = alpha * apad[y, x] + (1 - alpha) * out_image[y, x]
    return skimage.img_as_ubyte(out_image)


# Use the RANSAC algorithm to find the best homography to stitch the two images images together
def get_best_homography(good, keypoints_a, keypoints_b):
    best_match_inlier_count = 0
    best_match = None
    # Pick a large amount of iterations to ensure convergence on the best match
    for _ in range(0, 1400):
        # Get four random corresponding features
        random_four_matches = random.sample(good, 4)
        a = []
        b = []
        for i in range(0, 4):
            a.append([keypoints_a[random_four_matches[i][0].queryIdx].pt[0],
                      keypoints_a[random_four_matches[i][0].queryIdx].pt[1]])
            b.append([keypoints_b[random_four_matches[i][0].trainIdx].pt[0],
                      keypoints_b[random_four_matches[i][0].trainIdx].pt[1]])
        # Fit a homography to the four (x, y) points in image a and image b
        homography = fit_homography(a, b)
        # Loop through all good matches and determine if this homography is good by computing the number of
        # corresponding features which are close to where they should be once the homography is applied
        # counting the number of inliers
        inliers = 0
        for match in good:
            x_a, y_a = keypoints_a[match[0].queryIdx].pt
            x_b, y_b = keypoints_b[match[0].trainIdx].pt
            x_b_hom, y_b_hom = apply_homography(homography, x_b, y_b)
            # Compute the distance from the transformed coordinate (H applied to feature in B) to the coordinate of
            # that feature in A
            dist = math.sqrt((x_a - x_b_hom) ** 2 + (y_a - y_b_hom) ** 2)
            if dist < 0.01:
                inliers += 1
        if inliers > best_match_inlier_count:
            best_match_inlier_count = inliers
            best_match = homography
    print(('Found', best_match_inlier_count, 'inliers.'))
    print('Best Homography:')
    print(best_match)
    return best_match


print('Reading images.')
image_a = cv2.imread("test_images\\close_center.jpg")
image_b = cv2.imread("test_images\\template_inside.png")

# Use the SIFT descriptor to find keypoint features in the left and right images
print('Finding features in both images.')
sift = cv2.xfeatures2d.SIFT_create()
# Initiate STAR detector
orb = cv2.ORB_create()
keypoints_a = orb.detect(image_a, None)
keypoints_b = orb.detect(image_b, None)
keypoints_a, descriptor_a = orb.compute(image_a, keypoints_a)
keypoints_b, descriptor_b = orb.compute(image_b, keypoints_b)

img2 = cv2.drawKeypoints(image_a, keypoints_a, color=(0, 255, 0), flags=0, None)
img3 = cv2.drawKeypoints(image_b, keypoints_b, color=(255, 0, 0), flags=0, None)

plt.imsave("orb_center.png", img2)
plt.imsave('orb_template.png', img3)

# #keypoints_a, descriptor_a = sift.detectAndCompute(image_a, None)
# #keypoints_b, descriptor_b = sift.detectAndCompute(image_b, None)
#
# # Compare the features in both images. For each feature in the first image,
# # find the closest matching feature in the other image
# print('Computing matching features.')
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(descriptor_a, descriptor_b, k=2)
#
# # Apply the ratio test to remove matches which are too similar to one another
# # Keeping only the unique matches
# print('Applying ratio test to matches.')
# good = []
# for m, n in matches:
#     if m.distance < 0.7 * n.distance:
#         good.append([m])
#
# # Find the best homography to stitch the two images together
# print('Computing best homography.')
# best_homography = get_best_homography(good, keypoints_a, keypoints_b)
#
# p = skimage.transform.ProjectiveTransform(best_homography)  # Inverse of homography (used for inverse warping)
# bwarp = skimage.transform.warp(image_a, p, output_shape=(500, 500))  # Inverse warp b to a coords
# plt.imsave("warped.png", bwarp)
#
#
# # Stitch the images
# #print 'Stitching images.'
# #image_out = composite_warped(image_a, image_b, best_homography)
#
# #print 'Saving stitched image to: stitched.png'
# #plt.imsave("stitched.png", image_out[:, :, ::-1])
