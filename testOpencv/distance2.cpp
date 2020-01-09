#include <opencv2/opencv.hpp>  
#include "iostream"
#include <stdio.h>
using namespace std;
using namespace cv;

Mat depth;
//depth=Mat::zeros(480,640,CV_8UC1);

static void print_help()
{
	printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");
	printf("\nUsage: stereo_match <left_image> <right_image> [--algorithm=bm|sgbm|hh|var] [--blocksize=<block_size>]\n"
		"[--max-disparity=<max_disparity>] [--scale=scale_factor>] [-i <intrinsic_filename>] [-e <extrinsic_filename>]\n"
		"[--no-display] [-o <disparity_image>] [-p <point_cloud_file>]\n");
}
int pic_info[4];

static void onMouse(int event, int x, int y, int flags, void* param)
{
	//Mat mouse_show;
	//image.copyTo(mouse_show);
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		Mat& image = *(cv::Mat*) param;
		/* cout<<"x  "<<x<<endl;
		cout<<"y  "<<y<<endl;
		cout<<image.cols<<endl;
		cout<<image.rows<<endl;
		image.at<uchar>(y,x);*/
		/*cout<<image<<endl<<endl;*/
		double distance1 = (image.at<float>(y, x));

		cout << "distance:" << distance1 << "米" << endl;

		//  cout << "x:" << pic_info[0]<<'\n'<<"y:" << pic_info[1] << endl;
	   //left_mouse = true;
	}
}
static void saveXYZ(const char* filename, const Mat& mat)
{
	const double max_z = 1.0e4;
	FILE* fp = fopen(filename, "wt");
	for (int y = 0; y < mat.rows; y++)
	{
		for (int x = 0; x < mat.cols; x++)
		{
			Vec3f point = mat.at<Vec3f>(y, x);
			if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
			fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
		}
	}
	fclose(fp);
}


int main()
{

	//读取.xml文件中的相机矩阵
	FileStorage M1_fs("Intrinsics_Camera_Left.yaml", cv::FileStorage::READ);
	Mat M1;
	M1_fs["Intrinsics_Camera_Left"] >> M1;//指的是.xml的id
	cout << "M1" << M1 << endl << endl;
	FileStorage M2_fs("Intrinsics_Camera_Right.xml", cv::FileStorage::READ);
	Mat M2;
	M2_fs["Intrinsics_Camera_Right"] >> M2;//指的是.xml的id
	cout << "M2" << M2 << endl << endl;
	FileStorage D1_fs("Distortion_Camera_Left.xml", cv::FileStorage::READ);
	Mat D1;
	D1_fs["Distortion"] >> D1;//指的是.xml的id
	cout << "D1" << D1 << endl << endl;
	FileStorage D2_fs("Distortion_Camera_Right.xml", cv::FileStorage::READ);
	Mat D2;
	D2_fs["Distortion"] >> D2;//指的是.xml的id
	cout << "D2" << D2 << endl << endl;
	FileStorage R_fs("RotRodrigues.xml", cv::FileStorage::READ);
	Mat R;
	R_fs["RotRodrigues"] >> R;//指的是.xml的id
	cout << "R" << R << endl << endl;
	FileStorage T_fs("Translation.xml", cv::FileStorage::READ);
	Mat T;
	T_fs["Translation"] >> T;//指的是.xml的id
	cout << "T" << T << endl << endl;
	//StereoBM bm;
	cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create();
	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create();

	//StereoSGBM sgbm;
	Mat left_img, right_img;
	left_img=imread("E:\\test\\testOpencv\\test\\1.png", CV_LOAD_IMAGE_COLOR);
	right_img=imread("E:\\test\\testOpencv\\test\\2.png", CV_LOAD_IMAGE_COLOR);
	VideoCapture cap(0);
	if (cap.isOpened() == 0)
	{
		cout << "open_erro" << endl;
	}
	Size img_size = Size(640, 480);//图片大小
	Mat frame;
	while (1)
	{
		cap >> frame;
		left_img = frame(Rect(0, 0, 320, 240));
		resize(left_img, left_img, img_size);
		cvtColor(left_img, left_img, CV_BGR2GRAY);
		right_img = frame(Rect(320, 0, 320, 240));
		resize(right_img, right_img, img_size);
		cvtColor(right_img, right_img, CV_BGR2GRAY);
		int cn = left_img.channels();
		//输出参数
		Mat R1, P1, R2, P2;
		Rect roi1, roi2;
		Mat Q;
		stereoRectify
		(
			//通过标定参数来得到校准参数,输出校准的矩阵P1，P2为下面的initUndistortRectifyMap提供参数
			M1,//第一个相机矩阵
			D1, //第一个相机畸变参数
			M2,// 第二个相机矩阵
			D2,//第二个相机畸变参数
			img_size,// 用于校正的图像大小
			R,// 第一和第二相机坐标系之间的旋转矩阵。
			T,//第一和第二相机坐标系之间的平移矩阵.
			R1,//输出第一个相机的3x3矫正变换(旋转矩阵) 
			R2, //输出第二个相机的3x3矫正变换(旋转矩阵)
			P1,//在第一台相机的新的坐标系统(矫正过的)输出 3x3 的投影矩阵
			P2,//在第二台相机的新的坐标系统(矫正过的)输出 3x3 的投影矩阵
			Q, //输出深度视差映射矩阵
			CALIB_ZERO_DISPARITY,//CALIB_ZERO_DISPARITY,如果设置了CV_CALIB_ZERO_DISPARITY,函数的作用是使每个相机的主点在校正后的图像上有相同的像素坐标。如果未设置标志，功能还可以改变图像在水平或垂直方向 
			0,//alpha=0，校正后的图像进行缩放和偏移，只有有效像素是可见的（校正后没有黑色区域）alpha= 1意味着校正图像的抽取和转移，所有相机原始图像素像保留在校正后的图像（源图像像素没有丢失） 
			img_size,//校正后新的图像分辨率 
			&roi1, &roi2 //校正后的图像可选的输出矩形，里面所有像素都是有效的。如果alpha= 0，ROIs覆盖整个图像。否则，他们可能会比较小。
		);


		Mat map11, map12, map21, map22;
		initUndistortRectifyMap//initUndistortRecti01Map用来计算畸变映射
		(M1,//cameraMatrix为之前求得的相机的内参矩阵
			D1, //distCoeffs为之前求得的相机畸变矩阵；
			R1,//可选的输入，是第一和第二相机坐标之间的旋转矩阵；
			P1,//输入的校正后的3X3摄像机矩阵
			img_size, //摄像机采集的无失真的图像尺寸
			CV_16SC2,
			map11, map12);//输出的X/Y坐标重映射参数；
		initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);


		Mat img1r, img2r;
		remap//remap把求得的映射应用到图像上
		(left_img,//代表畸变的原始图像；
			img1r,//矫正后的输出图像，跟输入图像具有相同的类型和大小；
			map11, map12,//X坐标和Y坐标的映射；
			INTER_LINEAR);//定义图像的插值方式；
		remap(right_img, img2r, map21, map22, INTER_LINEAR);


		left_img = img1r;
		right_img = img2r;
		imshow("left", left_img);
		imshow("right", right_img);
		int numberOfDisparities = 0;
		numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width / 8) + 15) & -16;//numberOfDisparities==0,取后者,获得最大视差
		Mat disp, disp8;
		bm->setROI1(roi1);
		bm->setROI2(roi2);//roi1 roi2 是stereoRectify函数产生的，意义是标定后对图像校正的有效像素范围(矩阵)
		bm->setPreFilterCap(31);//preFilterCap预处理滤波器的截断值，预处理的输出值仅保留[-preFilterCap, preFilterCap]范围内的值，参数范围：1 - 31（文档中是31，但代码中是 63）, intpreFilterType：预处理滤波器的类型，主要是用于降低亮度失真（photometric distortions）、消除噪声和增强纹理等
		bm->setBlockSize(9);//由于==0,取9，SAD窗口大小，容许范围是[5,255]，一般应该在 5x5 至 21x21 之间，参数必须是奇数，int 型
		bm->setMinDisparity(0);//最小视差
		bm->setNumDisparities(numberOfDisparities);//最大的视差，这个需要自己去定，这个数值比0大，而且要被16整除， 比如32 64
		bm->setTextureThreshold(10);//低纹理区域的判断阈值。如果当前SAD窗口内所有邻居像素点的x导数绝对值之和小于指定阈值，则该窗口对应的像素点的视差值为 0
		bm->setUniquenessRatio(15);//视差唯一性百分比， 视差窗口范围内最低代价是次低代价的(1 + uniquenessRatio/100)倍时，最低代价对应的视差值才是该像素点的视差，否则该像素点的视差为 0该参数不能为负值，一般5-15左右的值比较合适，int 型
		bm->setSpeckleWindowSize(100);//检查视差连通区域变化度的窗口大小, 值为 0 时取消 speckle 检查，int 型
		bm->setSpeckleRange(32);//视差变化阈值，当的视差清零，int 型
		bm->setDisp12MaxDiff(1);//左视差图（直接计算得出）和右视差图（通过cvValidateDisparity计算得出）之间的最大容许差异。超过该阈值的视差值将被清零。该参数默认为 -1，即不执行左右视差检查。int 型
		bm->compute(left_img, right_img, disp);
		//生成三维云点
		Mat xyz;
		reprojectImageTo3D(disp, xyz, Q, true);
		xyz *= 1.6;
		/*const char* point_cloud_filename = 0;*/
		//point_cloud_filename ="point_cloud.txt";//保存云点  
		//saveXYZ(point_cloud_filename, xyz);
		vector<Mat> xyzSet;
		split(xyz, xyzSet);
		xyzSet[2].copyTo(depth);//第三个数组存的是Z值，及深度
		disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities*16.));// 将原始视差数据的位深转换为 8 位
		Mat temp = disp8.clone();
		Mat disp8u = Mat::zeros(disp8.rows, disp8.cols, CV_8UC3);
		//把8位单通道图像转化为3通道彩色图像
		for (int y = 0; y < disp8.rows; y++)
		{
			for (int x = 0; x < disp8.cols; x++)
			{
				uchar val = disp8.at<uchar>(y, x);
				uchar r, g, b;
				if (val == 0)
					r = g = b = 0;
				else
				{
					r = 255 - val;
					g = val < 128 ? val * 2 : (uchar)((255 - val) * 2);
					b = val;
				}


				disp8u.at<cv::Vec3b>(y, x) = cv::Vec3b(r, g, b);

			}
		}

		imshow("disp", disp8u);
		threshold(temp, temp, 100, 255, THRESH_BINARY);
		imshow("temp", temp);
		/*cout<<depth<<endl<<endl;*/
		setMouseCallback("disp", onMouse, (void*)&depth);

		waitKey(10);
	}
}