#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;

// 计算旋转矩阵和单位矩阵的差距
double evaluateR(const cv::Mat& R) {
    cv::Mat identity = cv::Mat::eye(3, 3, CV_64F);  // 单位矩阵
    return norm(R - identity);  // 计算差距的范数
}

// 计算平移向量和目标向量 [0, 0, 1] 的差距
double evaluateT(const cv::Mat& t) {
    cv::Mat targetT = (cv::Mat_<double>(3, 1) << 0, 0, 1);  // 目标平移向量
    return norm(t - targetT);  // 计算差距的范数
}

int main(int argc, char** argv) {
    Mat img1 = imread("images/1.png", IMREAD_GRAYSCALE);
    Mat img2 = imread("images/0.png", IMREAD_GRAYSCALE);

    Ptr<Feature2D> b = cv::ORB::create();
    Mat descImg1, descImg2;
    std::vector<cv::KeyPoint> keyImg1, keyImg2;

    b->detectAndCompute(img1, Mat(), keyImg1, descImg1);
    b->detectAndCompute(img2, Mat(), keyImg2, descImg2);

    cv::Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("BruteForce");
    std::vector<cv::DMatch> matches;
    descriptorMatcher->match(descImg1, descImg2, matches, Mat());
 
    std::vector<cv::Point2f> points1, points2;
    for (const cv::KeyPoint& kp : keyImg1) {
        points1.push_back(kp.pt);
    }
    for (const cv::KeyPoint& kp : keyImg2) {
        points2.push_back(kp.pt);
    }
 
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 718.8560, 0, 607.1928, 0, 718.8560, 185.2157, 0, 0, 1);

    // 初始化最佳参数和最小误差
    double bestConfidence = 0.999;
    double bestThreshold = 5.0;
    double minError = std::numeric_limits<double>::max();
    cv::Mat bestR, bestT;

    for (double confidence = 0.99; confidence <= 0.99; confidence += 0.001) {
        for (double threshold = 1.0; threshold <= 10.0; threshold += 0.5) {
 
            cv::Mat mask;
            cv::Mat E = findEssentialMat(points1, points2, cameraMatrix, RANSAC, confidence, threshold, mask);
 
            cv::Mat R, t;
            int inliers = recoverPose(E, points1, points2, cameraMatrix, R, t, mask);
 
            double errorR = evaluateR(R);
            double errorT = evaluateT(t);

            double totalError = errorR + errorT;

            // 如果找到更好的参数组合，更新最优值
            if (totalError < minError) {
                minError = totalError;
                bestConfidence = confidence;
                bestThreshold = threshold;
                bestR = R.clone();
                bestT = t.clone();
                cout << confidence << "   " << threshold << endl;

            }
        }
    }

    // 输出最优结果
    cout << "Best Confidence: " << bestConfidence << endl;
    cout << "Best Threshold: " << bestThreshold << endl;
    cout << "Best Rotation Matrix (R): " << bestR << endl;
    cout << "Best Translation Vector (t): " << bestT << endl;

    // 使用内点匹配绘制匹配图
    std::vector<cv::DMatch> matches_inliers;
    cv::Mat mask;
    cv::Mat E = findEssentialMat(points1, points2, cameraMatrix, RANSAC, bestConfidence, bestThreshold, mask);
    recoverPose(E, points1, points2, cameraMatrix, bestR, bestT, mask);

    for (int i = 0; i < matches.size(); ++i) {
        if (mask.at<uchar>(i)) {
            matches_inliers.push_back(matches[i]);
        }
    }

    cv::Mat result_inliers;
    cv::drawMatches(img1, keyImg1, img2, keyImg2, matches_inliers, result_inliers);
    cv::imshow("Inlier Matches", result_inliers);
    cv::waitKey(0);

    return 0;
}
