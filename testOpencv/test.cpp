#include "highgui.h"
#include <math.h>
#include <cv.h>
#include "cxcore.h"
#define cvCvtPlaneToPix cvMerge 
double PSNR_B = 0;
double PSNR_G = 0;
double PSNR_R = 0;
double PSNR;

int main(int argc, char* argv[])
{
	const char* imagename = "E:\\test\\testOpencv\\1.jpg";
	IplImage *src;
	CvScalar SrcPixel;
	CvScalar DstPixel;
	double sumB = 0;
	double sumG = 0;
	double sumR = 0;
	double mseB;
	double mseG;
	double mseR;

	src = cvLoadImage(imagename, 1);
	if (!src)
	{
		printf("can't open the image...\n");
		return -1;
	}
	// YUV颜色空间   
	IplImage* YUVImage = cvCreateImage(cvSize(src->width, src->height), src->depth, 3);
	IplImage* dst = cvCreateImage(cvSize(src->width, src->height), src->depth, 3);
	// YUV颜色空间各通道   
	IplImage* Y = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, 1);
	IplImage* U = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, 1);
	IplImage* V = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, 1);

	//cvNamedWindow( "Origin Image", CV_WINDOW_AUTOSIZE );   
	cvCvtColor(src, YUVImage, CV_BGR2YUV); //BGR→YUV   
	cvSplit(YUVImage, Y, U, V, NULL);//分割通道   

	CvMat* MatY = cvCreateMat(Y->height, Y->width, CV_64FC1);
	CvMat* MatU = cvCreateMat(V->height, U->width, CV_64FC1);
	CvMat* MatV = cvCreateMat(V->height, V->width, CV_64FC1);

	CvMat* DCTY = cvCreateMat(Y->height, Y->width, CV_64FC1);
	CvMat* DCTU = cvCreateMat(U->height, U->width, CV_64FC1);
	CvMat* DCTV = cvCreateMat(V->height, V->width, CV_64FC1);

	cvScale(Y, MatY);
	cvScale(U, MatU);
	cvScale(V, MatV);

	cvDCT(MatY, DCTY, CV_DXT_FORWARD); //余弦变换   
	cvDCT(MatU, DCTU, CV_DXT_FORWARD); //余弦变换   
	cvDCT(MatV, DCTV, CV_DXT_FORWARD); //余弦变换   

									   //Y 通道压缩   
	for (int i = 0; i < Y->height; i++)
	{
		for (int j = 0; j < Y->width; j++)
		{
			double  element = CV_MAT_ELEM(*DCTY, double, i, j);
			if (abs(element) < 10)
				CV_MAT_ELEM(*DCTY, double, i, j) = 0;
		}
	}

	// U 通道压缩   
	for (int i = 0; i < U->height; i++)
	{
		for (int j = 0; j < U->width; j++)
		{
			double  element = CV_MAT_ELEM(*DCTU, double, i, j);
			if (abs(element) < 20)
				CV_MAT_ELEM(*DCTU, double, i, j) = 0;
		}
	}

	// V 通道压缩   
	for (int i = 0; i < V->height; i++)
	{
		for (int j = 0; j < V->width; j++)
		{
			double  element = CV_MAT_ELEM(*DCTV, double, i, j);
			if (abs(element) < 20)
				CV_MAT_ELEM(*DCTV, double, i, j) = 0;
		}
	}
	cvDCT(DCTY, MatY, CV_DXT_INVERSE); //余弦反变换   
	cvDCT(DCTU, MatU, CV_DXT_INVERSE);
	cvDCT(DCTV, MatV, CV_DXT_INVERSE);

	cvScale(MatY, Y);
	cvScale(MatU, U);
	cvScale(MatV, V);

	cvMerge(Y, U, V, NULL, YUVImage);
	cvCvtColor(YUVImage, dst, CV_YUV2BGR); //YUV→BGR   

										   //  计算前后两幅图像的PSNR值   
	for (int i = 0; i < src->height; i++)
	{
		for (int j = 0; j < src->width; j++)
		{
			SrcPixel = cvGet2D(src, i, j);
			DstPixel = cvGet2D(dst, i, j);
			sumB += (SrcPixel.val[0] - DstPixel.val[0]) * (SrcPixel.val[0] - DstPixel.val[0]);
			sumG += (SrcPixel.val[1] - DstPixel.val[1]) * (SrcPixel.val[1] - DstPixel.val[1]);
			sumR += (SrcPixel.val[2] - DstPixel.val[2]) * (SrcPixel.val[2] - DstPixel.val[2]);

		}
	}
	mseB = sumB / ((src->width) * (src->height)); //计算均方差   
	mseG = sumG / ((src->width) * (src->height));
	mseR = sumR / ((src->width) * (src->height));

	PSNR_B = 10.0 * (log10(255.0 * 255.0 / mseB));
	PSNR_G = 10.0 * (log10(255.0 * 255.0 / mseG));
	PSNR_R = 10.0 * (log10(255.0 * 255.0 / mseR));
	PSNR = (PSNR_B + PSNR_G + PSNR_R) / 3;
	printf("PSNR:%d ", PSNR_B);
	cvShowImage("YImage", Y);
	cvShowImage("UImage", U);
	cvShowImage("VImage", V);
	cvShowImage("DstImage", dst);

	cvSaveImage("E:\test\testOpencv\3.jpg", dst);

	while (1)
	{
		if (cvWaitKey(0) == 27) break;
	}

	cvDestroyWindow("YImage");
	cvDestroyWindow("UImage");
	cvDestroyWindow("VImage");
	cvDestroyWindow("DstImage");


	cvReleaseImage(&Y);
	cvReleaseImage(&U);
	cvReleaseImage(&V);
	cvReleaseImage(&src);
	cvReleaseImage(&dst);
	cvReleaseImage(&YUVImage);
	system("pause");
	return 0;

}