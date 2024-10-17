#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <random>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    std::vector<cv::String, std::allocator<cv::String>> imagePath;

    // 加载两张图片
    Mat img1 = imread("images/0.png", IMREAD_GRAYSCALE);
    Mat img2 = imread("images/1.png", IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        cout << "Could not open or find the images!" << endl;
        return -1;
    }

    // ORB 特征检测
    Ptr<Feature2D> b = cv::ORB::create();
    Mat descImg1, descImg2;
    std::vector<cv::KeyPoint> keyImg1, keyImg2;

    // 检测特征点并计算描述符
    b->detectAndCompute(img1, Mat(), keyImg1, descImg1);
    b->detectAndCompute(img2, Mat(), keyImg2, descImg2);

    // 转换为彩色图像以便绘制彩色的特征点
    Mat img1_with_keypoints, img2_with_keypoints;
    cvtColor(img1, img1_with_keypoints, COLOR_GRAY2BGR);
    cvtColor(img2, img2_with_keypoints, COLOR_GRAY2BGR);

    // 随机数生成器，用于生成随机颜色
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> dist(0, 255);

    // 在图像1中标记特征点（使用随机颜色）
    for (const auto& keypoint : keyImg1) {
        // 生成随机颜色
        cv::Scalar randomColor(dist(rng), dist(rng), dist(rng));

        // 绘制特征点
        cv::circle(img1_with_keypoints, keypoint.pt, 4, randomColor, 1);
    }

    // 在图像2中标记特征点（使用随机颜色）
    for (const auto& keypoint : keyImg2) {
        // 生成随机颜色
        cv::Scalar randomColor(dist(rng), dist(rng), dist(rng));

        // 绘制特征点
        cv::circle(img2_with_keypoints, keypoint.pt, 4, randomColor, 1);
    }

    // 显示两张标注了特征点的图片
    cv::imshow("Image 1 with Keypoints", img1_with_keypoints);
    cv::imshow("Image 2 with Keypoints", img2_with_keypoints);

    cv::waitKey(0);

    return 0;
}
