#include<bits/stdc++.h>
#include<opencv2/opencv.hpp>
#include<opencv2/dnn/dnn.hpp>

using cv::Mat;
using std::vector;
using std::string;

//ONNX推理模型路径
std::string ONNX_Path ="/home/valmorx/PycharmProjects/TransONNX/yolo11n.onnx";
int ONNX_Width = 640;
int ONNX_Height = 480;

static const vector<string> class_name = {};


void print_result(const Mat& result,float conf=0.5,int len_data=84) {
    //std::cout<<result.total()<<std::endl;
    float *pdata = (float*)result.data;

    for (int i=0;i<result.total()/len_data;i++) {
        if (pdata[4*result.total()/len_data+i]>conf) {
            for (int j=0;j<len_data;j++) {
                std::cout<<pdata[j*result.total()/len_data+i]<<" ";
            }
            std::cout<<std::endl;
        }
    }
}

vector<vector<float>> get_info (const Mat& result,float conf=0.5,int len_data=84) {
    //std::cout<<result.total()<<std::endl;
    float *pdata = (float*)result.data;

    vector<vector<float>> info;
    for (int i=0;i<result.total()/len_data;i++) {

        if (pdata[4*result.total()/len_data+i]>conf) {
            vector<float> info_line;
            for (int j=0;j<len_data;j++) {
                //std::cout<<pdata[j*result.total()/len_data+i]<<" ";
                info_line.push_back(pdata[j*result.total()/len_data+i]);
            }
            info.push_back(info_line);
        }
    }
    return info;
}

void info_simplify(vector<vector<float>> &info) { //坐标转化与

    if (info.empty()) {
        return;
    }

    for (auto i=0;i<info.size(); i++) {
        info[i][5] = std::max_element(info[i].cbegin()+5,info[i].cend())-(info[i].cbegin()+5); //指向最大元素的迭代器
        info[i].resize(6); //缩小 去除无用数据

        float x=info[i][0];
        float y=info[i][1];
        float w=info[i][2];
        float h=info[i][3];

        //四角坐标
        info[i][0]=x-w/2.0 < 0 ? 0 : x-w/2.0;//左上
        info[i][1]=y-h/2.0 < 0 ? 0 : y-h/2.0;
        info[i][2]=x+w/2.0 > ONNX_Width ? ONNX_Width : x+w/2.0;//右下
        info[i][3]=y+h/2.0 > ONNX_Height ? ONNX_Height : y+h/2.0;

    }

}

vector<vector<vector<float>>> split_info(vector<vector<float>>& info) { //分类

    vector<vector<vector<float>>> info_split;

    if (info.empty()) {
        return info_split;
    }

    vector<int> class_id;

    for (auto i=0;i<info.size(); i++) {
        if (std::find(class_id.begin(),class_id.end(),(int)info[i][5])==class_id.end()) {
            class_id.push_back((int)info[i][5]);
            vector<vector<float>> info_;
            info_split.push_back(info_);
        }
        info_split[std::find(class_id.begin(),class_id.end(),(int)info[i][5])-class_id.begin()].push_back(info[i]);

    }
    return info_split;
}

bool cmp(vector<float> a,vector<float> b) {return a[4]>b[4];}
void nms(vector<vector<float>> &info,float IOU = 0.8) { //NMS去重函数

    if (info.empty()) {
        return;
    }

    int counter = 0;
    vector<vector<float>> return_info;

    while (counter < info.size()) {
        return_info.clear();
        float x1=0,x2=0,y1=0,y2=0;

        std::sort(info.begin(),info.end(),cmp);

        for (auto i=0;i<info.size(); i++) {
            if (i<counter) {
                return_info.push_back(info[i]);
                continue;
            }
            if (i==counter) {
                x1 = info[i][0];
                y1 = info[i][1];
                x2 = info[i][2];
                y2 = info[i][3];
                return_info.push_back(info[i]);
                continue;
            }
            if (i>counter) {
                if (info[i][0] > x2 || info[i][2] < x1 ||info[i][1] > y2 || info[i][3] < y1) {
                    return_info.push_back(info[i]);
                }else {
                    float over_x1 = std::max(x1,info[i][0]);
                    float over_y1 = std::max(y1,info[i][1]);
                    float over_x2 = std::min(x2,info[i][2]);
                    float over_y2 = std::min(y2,info[i][3]);

                    float s_over = (over_x2-over_x1)*(over_y2-over_y1);
                    float s_total = (x2-x1)*(y2-y1)+(info[i][0]-info[i][2])*(info[i][1]-info[i][3])-s_over;
                    float ratio = 1.0*s_over/s_total;

                    //std::cout << ratio << std::endl;

                    if (ratio < IOU) {
                        return_info.push_back(info[i]);
                    }
                }
            }
        }
        info = return_info;
        counter++;
    }
}

void print_info(const vector<vector<float>> &info) {

    if (info.empty()) {
        return;
    }

    for (auto i=0;i<info.size(); i++) {
        for (auto j=0;j<info[i].size(); j++) {
            std::cout<<info[i][j]<<" ";
        }
        std::cout<<std::endl;
    }
}

void drawBox(Mat& img,const vector<vector<float>> &info) { //画框框

    if (info.empty()) {
        return;
    }

    for (auto i=0;i<info.size(); i++) {
        cv::rectangle(img,cv::Point(info[i][0],info[i][1]),cv::Point(info[i][2],info[i][3]),cv::Scalar(0,0,255),4);

        string label;
        std::stringstream oss;
        oss<<info[i][4];

        label += oss.str();
        cv::putText(img,label,cv::Point(info[i][0],info[i][1]),1,1,cv::Scalar(0,255,0),2);
    }

}

void PhotoProc(string path_Target) {

    cv::dnn::Net net = cv::dnn::readNetFromONNX(ONNX_Path);
    Mat img = cv::imread(path_Target);
    int img_width=img.cols,img_height=img.rows;
    std::cout << img_width << " " << img_height << std::endl;

    cv::resize(img,img,cv::Size(ONNX_Width,ONNX_Height));
    Mat blob = cv::dnn::blobFromImage(img,1.0/255.0,cv::Size(ONNX_Width,ONNX_Height),cv::Scalar(),true);

    net.setInput(blob);

    Mat processed=net.forward();

    //print_result(processed);

    vector<vector<float>> info = get_info(processed);

    info_simplify(info);
    //print_info(info);

    vector<vector<vector<float>>> info_split=split_info(info);
    //print_info(info_split[0]);

    //std::cout << info.size() << " " << info[0].size() << std::endl;

    nms(info_split[0]);
    print_info(info_split[0]);
    drawBox(img,info_split[0]);
    resize(img,img,cv::Size(img_width,img_height));

    cv::namedWindow("image",cv::WINDOW_AUTOSIZE);
    cv::imshow("image",img);
    cv::waitKey(0);
    cv::destroyAllWindows();

}

void VideoProc(int index) {
    cv::dnn::Net mynet=cv::dnn::readNetFromONNX(ONNX_Path);

    if (mynet.empty()) {
        std::cout << "Empty net." << std::endl;
        return;
    }

    Mat img;
    cv::VideoCapture cap(index,cv::CAP_V4L2);

    while (true) {
        cap.read(img);
        int img_width=img.cols,img_height=img.rows;

        if (img.empty()) {
            std::cout << "Empty / Done." << std::endl;
            break;
        }

        cv::namedWindow("image",cv::WINDOW_AUTOSIZE);
        imshow("image",img);

        Mat blob = cv::dnn::blobFromImage(img,1.0/255.0,cv::Size(ONNX_Width,ONNX_Height),cv::Scalar(),true);
        mynet.setInput(blob);

        Mat processed=mynet.forward();


        //print_result(processed);

        vector<vector<float>> info = get_info(processed);

        if (info.empty()) {
            imshow("image",img);

            char c=cv::waitKey(30);
            if (c==27) break;
            continue;
        }

        info_simplify(info);
        //print_info(info);

        vector<vector<vector<float>>> info_split=split_info(info);
        //print_info(info_split[0]);

        //std::cout << info.size() << " " << info[0].size() << std::endl;

        nms(info_split[0]);
        print_info(info_split[0]);
        drawBox(img,info_split[0]);
        resize(img,img,cv::Size(img_width,img_height));
        imshow("image",img);

        char c=cv::waitKey(30);
        if (c==27) break;
    }
    cv::destroyAllWindows();
}

void VideoProc(string VideoPath) {
    cv::dnn::Net mynet=cv::dnn::readNetFromONNX(ONNX_Path);

    if (mynet.empty()) {
        std::cout << "Empty net." << std::endl;
        return;
    }

    Mat img;
    cv::VideoCapture cap(VideoPath);

    while (true) {
        cap.read(img);
        int img_width=img.cols,img_height=img.rows;
        //std::cout << img_width << " " << img_height << std::endl;

        if (img.empty()) {
            std::cout << "Empty / Done." << std::endl;
            break;
        }

        cv::namedWindow("image",cv::WINDOW_AUTOSIZE);

        Mat blob = cv::dnn::blobFromImage(img,1.0/255.0,cv::Size(ONNX_Width,ONNX_Height),cv::Scalar(),true);
        mynet.setInput(blob);

        Mat processed=mynet.forward();


        //print_result(processed);

        vector<vector<float>> info = get_info(processed);

        if (info.empty()) {
            imshow("image",img);

            char c=cv::waitKey(30);
            if (c==27) break;
            continue;
        }

        info_simplify(info);
        //print_info(info);

        vector<vector<vector<float>>> info_split=split_info(info);
        //print_info(info_split[0]);

        //std::cout << info.size() << " " << info[0].size() << std::endl;

        nms(info_split[0]);
        print_info(info_split[0]);
        drawBox(img,info_split[0]);
        resize(img,img,cv::Size(img_width,img_height));
        imshow("image",img);

        char c=cv::waitKey(30);
        if (c==27) break;
    }
    cv::destroyAllWindows();
}

int main() {
    string path_Target = "/home/valmorx/CLionProjects/OpenCV_Yolov11/Lib_Photo/06.jpg";
    //PhotoProc(path_Target);
    VideoProc(0);
    string path_Video = "/home/valmorx/CLionProjects/OpenCV_Yolov11/Lib_Photo/01.mp4";
    //VideoProc(path_Video);
}