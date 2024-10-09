
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    std::vector<cv::String, std::allocator<cv::String>> imagePath;

    Mat img1 = imread(/*fileName[0]*/"images/0.png", IMREAD_GRAYSCALE);
    Mat img2 = imread(/*fileName[1]*/"images/1.png", IMREAD_GRAYSCALE);

    Ptr<Feature2D> b;
    b = cv::ORB::create();
    Mat descImg1,descImg2;
    std::vector<cv::KeyPoint> keyImg1,keyImg2;
    // b->detect(img1, keyImg1, Mat()); 
    // b->compute(img1, keyImg1, descImg1);
    b->detectAndCompute(img1, Mat(), keyImg1, descImg1, false);
    b->detectAndCompute(img2, Mat(), keyImg2, descImg2, false);
    //1）选取特征匹配算法
    cv::Ptr<DescriptorMatcher>  descriptorMatcher = DescriptorMatcher::create("BruteForce");
    //2）特征匹配
    std::vector<cv::DMatch> matches;
    descriptorMatcher->match(descImg1, descImg2, matches, Mat());
    //3）可视化
    cv::Mat result;
    cv::drawMatches(img1, keyImg1, img2, keyImg2, matches, result);
    cout<<"Matching..."<<endl;
    cv::imshow("Matches",result);
    // waitKey(0);
    cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) << 718.8560, 0, 607.1928, 0, 718.8560, 185.2157, 0, 0, 1);
    std::vector<cv::Point2f> points1, points2;
    // 遍历关键点并提取位置
    for (const cv::KeyPoint& kp : keyImg1) {
        points1.push_back(kp.pt); // kp.pt 是关键点的位置
    }
    for (const cv::KeyPoint& kp : keyImg2) {
        points2.push_back(kp.pt); // kp.pt 是关键点的位置
    }
    cout<<"E testing..."<<endl;

    cv::Mat mask;
    // cv::Mat E = findEssentialMat(points1, points2, cameraMatrix, RANSAC);
    cv::Mat E = findEssentialMat(points1, points2, cameraMatrix, RANSAC, 0.9, 3.0, mask);

    cout<<"E good!"<<endl;

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

    // 2. 使用内点匹配绘制匹配图
    cv::Mat result_inliers;
    cv::drawMatches(img1, keyImg1, img2, keyImg2, matches_inliers, result_inliers);

    // 3. 显示筛选后的匹配结果
    cv::imshow("Inlier Matches", result_inliers);
    cv::waitKey(0);

}








