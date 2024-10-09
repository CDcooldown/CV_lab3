#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    // 1. 读取图像
    Mat img1 = imread("images/0.png", IMREAD_GRAYSCALE);
    Mat img2 = imread("images/1.png", IMREAD_GRAYSCALE);
    if (img1.empty() || img2.empty()) {
        cerr << "Could not open or find the images!" << endl;
        return -1;
    }

    // 2. SIFT 特征点检测与描述符计算
    Ptr<SIFT> sift = SIFT::create();
    std::vector<KeyPoint> keyImg1, keyImg2;
    Mat descImg1, descImg2;
    sift->detectAndCompute(img1, Mat(), keyImg1, descImg1);
    sift->detectAndCompute(img2, Mat(), keyImg2, descImg2);

    // 3. 使用 KNN 匹配器匹配描述符
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<DMatch>> knn_matches;
    matcher->knnMatch(descImg1, descImg2, knn_matches, 2);

    // 4. 通过 Lowe's ratio test 筛选匹配
    const float ratio_thresh = 0.75f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    cv::Mat result;
    cv::drawMatches(img1, keyImg1, img2, keyImg2, good_matches, result);
    cout<<"Matching..."<<endl;
    cv::imshow("Matches",result);

    // 5. 提取匹配点的位置
    std::vector<Point2f> points1, points2;
    for (const auto& match : good_matches) {
        points1.push_back(keyImg1[match.queryIdx].pt);
        points2.push_back(keyImg2[match.trainIdx].pt);
    }

    // 6. 相机内参矩阵
    Mat cameraMatrix = (Mat_<double>(3, 3) << 718.8560, 0, 607.1928, 0, 718.8560, 185.2157, 0, 0, 1);

    // 7. 计算本质矩阵并使用 RANSAC 筛选内点
    Mat mask;
    Mat E = findEssentialMat(points1, points2, cameraMatrix, RANSAC, 0.999, 1.0, mask);

    // 8. 恢复相对位姿（旋转矩阵和平移向量）
    Mat R, t;
    int inliers = recoverPose(E, points1, points2, cameraMatrix, R, t, mask);

    cout << "Inliers: " << inliers << endl;
    cout << "Rotation Matrix (R): " << R << endl;
    cout << "Translation Vector (t): " << t << endl;

    // 9. 筛选出 RANSAC 后的内点匹配
    std::vector<DMatch> inlier_matches;
    for (size_t i = 0; i < good_matches.size(); i++) {
        if (mask.at<uchar>(i)) {
            inlier_matches.push_back(good_matches[i]);
        }
    }

    // 10. 绘制内点匹配
    Mat result_inliers;
    drawMatches(img1, keyImg1, img2, keyImg2, inlier_matches, result_inliers);

    // 11. 显示筛选后的匹配结果
    imshow("Inlier Matches", result_inliers);
    waitKey(0);

    return 0;
}
