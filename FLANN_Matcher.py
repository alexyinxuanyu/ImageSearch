import numpy as np
import cv2
from matplotlib import pyplot as plt
#img1 = cv2.imread('box.png',0)          # queryImage
img1 = cv2.imread('/Users/yinxuanyu/Desktop/imagefeaturesearch/image_search/data_example/query/query.png',0)
img2 = cv2.imread('/Users/yinxuanyu/Desktop/imagefeaturesearch/image_search/1.jpg',0)
#img2 = cv2.imread('box_in_scene.png',0) # trainImage
# Initiate SIFT detector
#orb = cv2.ORB_create()
def featureextractandmatching(img1,img2,z):

    orb = cv2.KAZE_create()
    print("running")
    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1 , None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    '''if(des1.type()!=CV_32F):
        des1.convertTo(des1, CV_32F)
        des1.convertTo(CV_32F)
    
    
    if(des2.type()!=CV_32F):
        des2.convertTo(des2, CV_32F)'''
    res = np.float32(des1)
    res2 = np.float32(des2)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flannss = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flannss.knnMatch(res, res2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    plt.imshow(img3)
    plt.savefig('/Users/yinxuanyu/Desktop/imagefeaturesearch/image_search/KAZE_Result/'+str(z)+'.jpg')
    plt.show()



if __name__ == '__main__':
    for i in range(7,9):
        img1 = cv2.imread('/Users/yinxuanyu/Desktop/imagefeaturesearch/image_search/data_example/query/query.png', 0)
        img2 = cv2.imread('/Users/yinxuanyu/Desktop/imagefeaturesearch/image_search/'+str(i+1)+'.jpg', 0)
        featureextractandmatching(img1, img2, i)
