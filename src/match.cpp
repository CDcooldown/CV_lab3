#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;

// 定义一个函数，判断两点的连线是否通过光心
bool isLineThroughPrincipalPoint(const Point2f& pt1, const Point2f& pt2, const Point2f& principalPoint, double threshold = 1.0) {
    // 计算两点与光心连线的斜率
    double slope1 = (pt1.y - principalPoint.y) / (pt1.x - principalPoint.x);
    double slope2 = (pt2.y - principalPoint.y) / (pt2.x - principalPoint.x);

    // 判断两条斜率是否相等，允许一定误差
    return fabs(slope1 - slope2) < threshold;
}

// 筛选匹配点，返回通过光心的匹配点
std::vector<cv::DMatch> filterMatchesByPrincipalPoint(
    const std::vector<cv::KeyPoint>& keyImg1, 
    const std::vector<cv::KeyPoint>& keyImg2, 
    const std::vector<cv::DMatch>& matches, 
    const cv::Point2f& principalPoint, 
    double threshold = 1.0) 
{
    std::vector<cv::DMatch> filteredMatches;

    for (const auto& match : matches) {
        Point2f pt1 = keyImg1[match.queryIdx].pt;
        Point2f pt2 = keyImg2[match.trainIdx].pt;

        // 检查连线是否通过光心
        if (isLineThroughPrincipalPoint(pt1, pt2, principalPoint, threshold)) {
            filteredMatches.push_back(match);
        }
    }

    return filteredMatches;
}

int main(int argc, char** argv) {
    // 加载图片和相机内参
    Mat img1 = imread("images/0.png", IMREAD_GRAYSCALE);
    Mat img2 = imread("images/1.png", IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        cout << "Could not open or find the images!" << endl;
        return -1;
    }

    // 相机内参矩阵
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 718.8560, 0, 607.1928, 0, 718.8560, 185.2157, 0, 0, 1);
    Point2f principalPoint(cameraMatrix.at<double>(0, 2), cameraMatrix.at<double>(1, 2));  // 光心坐标

    Ptr<Feature2D> b = cv::ORB::create();
    Mat descImg1, descImg2;
    std::vector<cv::KeyPoint> keyImg1, keyImg2;

    b->detectAndCompute(img1, Mat(), keyImg1, descImg1);
    b->detectAndCompute(img2, Mat(), keyImg2, descImg2);

    Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("BruteForce");
    std::vector<cv::DMatch> matches;
    descriptorMatcher->match(descImg1, descImg2, matches, Mat());

    // 筛选匹配点，利用光心连线条件
    std::vector<cv::DMatch> filteredMatches = filterMatchesByPrincipalPoint(keyImg1, keyImg2, matches, principalPoint);

    cout << "Original matches: " << matches.size() << endl;
    cout << "Filtered matches (passing through principal point): " << filteredMatches.size() << endl;

    // 提取筛选后的匹配点的图像坐标
    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : filteredMatches) {
        points1.push_back(keyImg1[match.queryIdx].pt);
        points2.push_back(keyImg2[match.trainIdx].pt);
    }

    // RANSAC 计算本质矩阵
    cv::Mat mask;
    cv::Mat E = findEssentialMat(points1, points2, cameraMatrix, RANSAC, 0.999, 1.0, mask);

    // 恢复旋转矩阵和平移向量
    cv::Mat R, t;
    int inliers = recoverPose(E, points1, points2, cameraMatrix, R, t, mask);

    cout << "Inliers after RANSAC: " << inliers << endl;
    cout << "Rotation Matrix (R):\n" << R << endl;
    cout << "Translation Vector (t):\n" << t << endl;

    // 过滤出 RANSAC 内点对应的匹配
    std::vector<cv::DMatch> ransacMatches;
    for (size_t i = 0; i < filteredMatches.size(); ++i) {
        if (mask.at<uchar>(i)) {
            ransacMatches.push_back(filteredMatches[i]);
        }
    }

    // 绘制 RANSAC 后的匹配结果
    Mat imgMatches;
    drawMatches(img1, keyImg1, img2, keyImg2, ransacMatches, imgMatches);
    imshow("RANSAC Filtered Matches", imgMatches);
    waitKey(0);

    return 0;
}
