#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <random>

using namespace std;
using namespace cv;
// 判断连线是否经过光心且两个点在光心的同侧
bool isValidMatch(const Point2f& pt1, const Point2f& pt2, const Point2f& principalPoint, double threshold = 1.0) {
    // 计算两点与光心连线的斜率
    double slope1 = (pt1.y - principalPoint.y) / (pt1.x - principalPoint.x);
    double slope2 = (pt2.y - principalPoint.y) / (pt2.x - principalPoint.x);

    // 判断连线是否经过光心（斜率相近）
    bool lineThroughPrincipalPoint = fabs(slope1 - slope2) < threshold;

    // 检查X方向是否在同侧
    bool sameSideX = (pt1.x - principalPoint.x) * (pt2.x - principalPoint.x) > 0;
    // 检查Y方向是否在同侧
    bool sameSideY = (pt1.y - principalPoint.y) * (pt2.y - principalPoint.y) > 0;

    // 两个点要在X方向和Y方向都在同侧
    bool sameSide = sameSideX && sameSideY;

    // 返回连线经过光心且在光心同侧的结果
    return lineThroughPrincipalPoint && sameSide;
}

// 筛选匹配点，返回连线经过光心且在光心同侧的匹配点
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

        // 检查匹配点是否满足条件
        if (isValidMatch(pt1, pt2, principalPoint, threshold)) {
            filteredMatches.push_back(match);
        }
    }

    return filteredMatches;
}

// 绘制匹配点（上下排列方式），并使用随机颜色
void drawMatchesWithRandomColors(const Mat& img1, const Mat& img2, 
                                 const std::vector<KeyPoint>& keyImg1, const std::vector<KeyPoint>& keyImg2, 
                                 const std::vector<DMatch>& matches, const std::string& windowName) 
{
    // 随机数生成器，用于生成随机颜色
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> dist(0, 255);

    // 创建结果图像，大小为两张图像上下拼接
    int height = img1.rows + img2.rows;
    int width = std::max(img1.cols, img2.cols);
    Mat result = Mat::zeros(height, width, CV_8UC3);

    // 将 img1 和 img2 分别拷贝到 result 的上下部分
    Mat top(result, Rect(0, 0, img1.cols, img1.rows));  // img1 位于上半部分
    cvtColor(img1, top, COLOR_GRAY2BGR);  // 转换为彩色

    Mat bottom(result, Rect(0, img1.rows, img2.cols, img2.rows));  // img2 位于下半部分
    cvtColor(img2, bottom, COLOR_GRAY2BGR);  // 转换为彩色

    // 绘制每个匹配点和连线
    for (const auto& match : matches) {
        Point2f pt1 = keyImg1[match.queryIdx].pt;
        Point2f pt2 = keyImg2[match.trainIdx].pt;

        pt2.y += img1.rows;  // 调整 pt2 的 y 坐标，使其对应 img2 的位置

        // 生成随机颜色
        Scalar randomColor(dist(rng), dist(rng), dist(rng));

        // 画出匹配线和关键点
        line(result, pt1, pt2, randomColor, 1);
        circle(result, pt1, 4, randomColor, 1);
        circle(result, pt2, 4, randomColor, 1);
    }

    // 显示结果图像
    imshow(windowName, result);
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
    Mat cameraMatrix = (Mat_<double>(3, 3) << 718.8560, 0, 607.1928, 0, 718.8560, 185.2157, 0, 0, 1);
    Point2f principalPoint(cameraMatrix.at<double>(0, 2), cameraMatrix.at<double>(1, 2));  // 光心坐标

    Ptr<Feature2D> b = ORB::create();
    Mat descImg1, descImg2;
    std::vector<KeyPoint> keyImg1, keyImg2;

    b->detectAndCompute(img1, Mat(), keyImg1, descImg1);
    b->detectAndCompute(img2, Mat(), keyImg2, descImg2);

    Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("BruteForce");
    std::vector<DMatch> matches;
    descriptorMatcher->match(descImg1, descImg2, matches, Mat());

    // 绘制初始匹配点
    drawMatchesWithRandomColors(img1, img2, keyImg1, keyImg2, matches, "Initial Matches");

    // 筛选匹配点，利用光心连线条件
    std::vector<DMatch> filteredMatches = filterMatchesByPrincipalPoint(keyImg1, keyImg2, matches, principalPoint);

    // 绘制筛选后的匹配点
    drawMatchesWithRandomColors(img1, img2, keyImg1, keyImg2, filteredMatches, "Filtered Matches (Through Principal Point)");

    // 提取筛选后的匹配点的图像坐标
    std::vector<Point2f> points1, points2;
    for (const auto& match : filteredMatches) {
        points1.push_back(keyImg1[match.queryIdx].pt);
        points2.push_back(keyImg2[match.trainIdx].pt);
    }
    // 输出初始匹配点数和筛选后匹配点数
    cout << "Initial Matches: " << matches.size() << endl;
    cout << "Filtered Matches (Through Principal Point): " << filteredMatches.size() << endl;

    // RANSAC 计算本质矩阵
    Mat mask;
    Mat E = findEssentialMat(points1, points2, cameraMatrix, RANSAC, 0.999, 3.0, mask);

    // 恢复旋转矩阵和平移向量
    Mat R, t;
    int inliers = recoverPose(E, points1, points2, cameraMatrix, R, t, mask);

    cout << "Inliers after RANSAC: " << inliers << endl;
    cout << "Rotation Matrix (R):\n" << R << endl;
    cout << "Translation Vector (t):\n" << t << endl;

    // 过滤出 RANSAC 内点对应的匹配
    std::vector<DMatch> ransacMatches;
    for (size_t i = 0; i < filteredMatches.size(); ++i) {
        if (mask.at<uchar>(i)) {
            ransacMatches.push_back(filteredMatches[i]);
        }
    }

    // 绘制 RANSAC 后的匹配结果
    drawMatchesWithRandomColors(img1, img2, keyImg1, keyImg2, ransacMatches, "RANSAC Filtered Matches");

    waitKey(0);

    return 0;
}
