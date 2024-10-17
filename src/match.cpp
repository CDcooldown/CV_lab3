#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <random>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    std::vector<cv::String, std::allocator<cv::String>> imagePath;

    Mat img1 = imread("images/0.png", IMREAD_GRAYSCALE);
    Mat img2 = imread("images/1.png", IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        cout << "Could not open or find the images!" << endl;
        return -1;
    }

    Ptr<Feature2D> b;
    b = cv::ORB::create();
    Mat descImg1, descImg2;
    std::vector<cv::KeyPoint> keyImg1, keyImg2;

    b->detectAndCompute(img1, Mat(), keyImg1, descImg1);
    b->detectAndCompute(img2, Mat(), keyImg2, descImg2);

    // 1) 选取特征匹配算法
    cv::Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("BruteForce");
    // 2) 特征匹配
    std::vector<cv::DMatch> matches;
    descriptorMatcher->match(descImg1, descImg2, matches, Mat());

    // 3) 手动创建上下排列的输出图像
    int height = img1.rows + img2.rows;
    int width = std::max(img1.cols, img2.cols);
    cv::Mat result = cv::Mat::zeros(height, width, CV_8UC3);

    // 将 img1 和 img2 拷贝到 result 的上下位置
    cv::Mat top(result, Rect(0, 0, img1.cols, img1.rows));  // img1 位于上半部分
    cv::cvtColor(img1, top, COLOR_GRAY2BGR);  // 转换为彩色图像以便显示彩色匹配点

    cv::Mat bottom(result, Rect(0, img1.rows, img2.cols, img2.rows));  // img2 位于下半部分
    cv::cvtColor(img2, bottom, COLOR_GRAY2BGR);  // 转换为彩色图像

    // 随机数生成器，用于生成随机颜色
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> dist(0, 255);

    // 4) 绘制匹配（使用随机颜色）
    for (const auto& match : matches) {
        cv::Point2f pt1 = keyImg1[match.queryIdx].pt;
        cv::Point2f pt2 = keyImg2[match.trainIdx].pt;

        pt2.y += img1.rows;  // 调整 pt2 的 y 坐标，使其对应 img2 的位置

        // 生成随机颜色
        cv::Scalar randomColor(dist(rng), dist(rng), dist(rng));

        // 画出匹配线和关键点，使用随机颜色
        cv::line(result, pt1, pt2, randomColor, 1);
        cv::circle(result, pt1, 4, randomColor, 1);
        cv::circle(result, pt2, 4, randomColor, 1);
    }

    cout << "Matching..." << endl;
    cv::imshow("Matches", result);

    // 相机内参矩阵
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 718.8560, 0, 607.1928, 0, 718.8560, 185.2157, 0, 0, 1);
    std::vector<cv::Point2f> points1, points2;

    // 遍历关键点并提取位置
    for (const cv::KeyPoint& kp : keyImg1) {
        points1.push_back(kp.pt);  // kp.pt 是关键点的位置
    }
    for (const cv::KeyPoint& kp : keyImg2) {
        points2.push_back(kp.pt);  // kp.pt 是关键点的位置
    }

    cv::Mat mask;
    // 计算本质矩阵
    cv::Mat E = findEssentialMat(points1, points2, cameraMatrix, RANSAC, 0.999, 3.0, mask);

    // 存储恢复出的旋转矩阵和平移向量
    cv::Mat R, t;

    // 恢复相对位姿
    int inliers = recoverPose(E, points1, points2, cameraMatrix, R, t, mask);

    cout << "Inliers: " << inliers << endl;
    cout << "Rotation Matrix (R): " << R << endl;
    cout << "Translation Vector (t): " << t << endl;

    // 1. 定义一个新的 matches_inliers 向量来存储筛选出的内点匹配
    std::vector<cv::DMatch> matches_inliers;

    for (int i = 0; i < matches.size(); ++i) {
        if (mask.at<uchar>(i)) {  // 只选择内点 (mask[i] == 1)
            matches_inliers.push_back(matches[i]);
        }
    }

    // 2. 手动创建上下排列的内点匹配图
    cv::Mat result_inliers = cv::Mat::zeros(height, width, CV_8UC3);

    cv::Mat top_inliers(result_inliers, Rect(0, 0, img1.cols, img1.rows));
    cv::cvtColor(img1, top_inliers, COLOR_GRAY2BGR);

    cv::Mat bottom_inliers(result_inliers, Rect(0, img1.rows, img2.cols, img2.rows));
    cv::cvtColor(img2, bottom_inliers, COLOR_GRAY2BGR);

    // 3. 绘制内点匹配（使用随机颜色）
    for (const auto& match : matches_inliers) {
        cv::Point2f pt1 = keyImg1[match.queryIdx].pt;
        cv::Point2f pt2 = keyImg2[match.trainIdx].pt;
        pt2.y += img1.rows;  // 调整 pt2 的 y 坐标

        // 生成随机颜色
        cv::Scalar randomColor(dist(rng), dist(rng), dist(rng));

        cv::line(result_inliers, pt1, pt2, randomColor, 1);
        cv::circle(result_inliers, pt1, 4, randomColor, 1);
        cv::circle(result_inliers, pt2, 4, randomColor, 1);
    }

    // 4. 显示筛选后的匹配结果
    cv::imshow("Inlier Matches", result_inliers);
    cv::waitKey(0);

    return 0;
}
