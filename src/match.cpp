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

    // 5. 手动创建上下拼接的输出图像
    int height = img1.rows + img2.rows;
    int width = std::max(img1.cols, img2.cols);
    cv::Mat result = cv::Mat::zeros(height, width, CV_8UC3);

    // 将 img1 和 img2 拷贝到 result 的上下位置
    cv::Mat top(result, Rect(0, 0, img1.cols, img1.rows));  // img1 位于上半部分
    cv::cvtColor(img1, top, COLOR_GRAY2BGR);  // 转换为彩色图像以便显示彩色匹配点

    cv::Mat bottom(result, Rect(0, img1.rows, img2.cols, img2.rows));  // img2 位于下半部分
    cv::cvtColor(img2, bottom, COLOR_GRAY2BGR);  // 转换为彩色图像

    // 6. 绘制匹配点和连线
    for (const auto& match : good_matches) {
        Point2f pt1 = keyImg1[match.queryIdx].pt;
        Point2f pt2 = keyImg2[match.trainIdx].pt;

        pt2.y += img1.rows;  // 调整 pt2 的 y 坐标，使其位于 img2 的正确位置

        // 生成随机颜色
        Scalar randomColor(rand() % 256, rand() % 256, rand() % 256);

        // 绘制匹配点和连线
        line(result, pt1, pt2, randomColor, 1);
        circle(result, pt1, 4, randomColor, 1);
        circle(result, pt2, 4, randomColor, 1);
    }

    // 7. 显示匹配结果
    cout << "Matching..." << endl;
    imshow("Matches", result);

    // 8. 提取匹配点的位置
    std::vector<Point2f> points1, points2;
    for (const auto& match : good_matches) {
        points1.push_back(keyImg1[match.queryIdx].pt);
        points2.push_back(keyImg2[match.trainIdx].pt);
    }

    // 9. 相机内参矩阵
    Mat cameraMatrix = (Mat_<double>(3, 3) << 718.8560, 0, 607.1928, 0, 718.8560, 185.2157, 0, 0, 1);

    // 10. 计算本质矩阵并使用 RANSAC 筛选内点
    Mat mask;
    Mat E = findEssentialMat(points1, points2, cameraMatrix, RANSAC, 0.999, 1.0, mask);

    // 11. 恢复相对位姿（旋转矩阵和平移向量）
    Mat R, t;
    int inliers = recoverPose(E, points1, points2, cameraMatrix, R, t, mask);

    cout << "Inliers: " << inliers << endl;
    cout << "Rotation Matrix (R): " << R << endl;
    cout << "Translation Vector (t): " << t << endl;

    // 12. 筛选出 RANSAC 后的内点匹配
    std::vector<DMatch> inlier_matches;
    for (size_t i = 0; i < good_matches.size(); i++) {
        if (mask.at<uchar>(i)) {
            inlier_matches.push_back(good_matches[i]);
        }
    }

    // 13. 手动创建上下拼接的内点匹配图
    cv::Mat result_inliers = cv::Mat::zeros(height, width, CV_8UC3);
    cv::Mat top_inliers(result_inliers, Rect(0, 0, img1.cols, img1.rows));
    cv::cvtColor(img1, top_inliers, COLOR_GRAY2BGR);
    cv::Mat bottom_inliers(result_inliers, Rect(0, img1.rows, img2.cols, img2.rows));
    cv::cvtColor(img2, bottom_inliers, COLOR_GRAY2BGR);

    // 14. 绘制内点匹配
    for (const auto& match : inlier_matches) {
        Point2f pt1 = keyImg1[match.queryIdx].pt;
        Point2f pt2 = keyImg2[match.trainIdx].pt;

        pt2.y += img1.rows;  // 调整 pt2 的 y 坐标，使其位于 img2 的正确位置

        // 生成随机颜色
        Scalar randomColor(rand() % 256, rand() % 256, rand() % 256);

        // 绘制内点匹配
        line(result_inliers, pt1, pt2, randomColor, 1);
        circle(result_inliers, pt1, 4, randomColor, 1);
        circle(result_inliers, pt2, 4, randomColor, 1);
    }

    // 15. 显示筛选后的内点匹配结果
    imshow("Inlier Matches", result_inliers);
    waitKey(0);

    return 0;
}
