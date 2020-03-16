
/*********************************************/
/*小结：一开始理解有误，以为需要傅里叶变换，
频率域滤波等（然而并不能看懂qvq）。
只好照着论文公式敲了一遍，但是效果不太好.
又根据论文提示加了一道双边滤波预处理。*/
/*********************************************/


#include <opencv2/core/core.hpp>                    
#include <opencv2/highgui/highgui.hpp>        
#include <opencv2/imgproc/imgproc.hpp>    
#include <opencv2/highgui/highgui.hpp>
#include <iostream>       

using namespace std;
using namespace cv;

/* 高斯滤波 (待处理单通道图片, 高斯分布数组， 高斯数组大小(核大小) ) */
void gaussian(cv::Mat *_src,cv::Mat *keep, double **_array, int _size)
{
	cv::Mat keep1((*_src).rows, (*_src).cols, CV_64FC1);
	/*double[][]*/
	/*keep.convertTo(keep, CV_32FC1);*/
	cv::Mat temp = (*_src).clone();
	/*keep1 = (*_src).clone();*/
	// [1] 扫描
	for (int i = 0; i < (*_src).rows; i++) {
		for (int j = 0; j < (*_src).cols; j++) {
			// [2] 忽略边缘
			if (i > (_size / 2) - 1 && j > (_size / 2) - 1 &&
				i < (*_src).rows - (_size / 2) && j < (*_src).cols - (_size / 2))
			{
				// [3] 找到图像输入点f(i,j),以输入点为中心与核中心对齐
				//     核心为中心参考点 卷积算子=>高斯矩阵180度转向计算
				//     x y 代表卷积核的权值坐标   i j 代表图像输入点坐标
				//     卷积算子     (f*g)(i,j) = f(i-k,j-l)g(k,l)          f代表图像输入 g代表核
				//     带入核参考点 (f*g)(i,j) = f(i-(k-ai), j-(l-aj))g(k,l)   ai,aj 核参考点
				//     加权求和  注意：核的坐标以左上0,0起点
				double sum = 0.0;
				for (int k = 0; k < _size; k++)
				{
					for (int l = 0; l < _size; l++) 
					{
						sum += (*_src).ptr<uchar>(i - k + (_size / 2))[j - l + (_size / 2)] * _array[k][l];
						//mat.ptr<type>(row)[col]
						//卷积算子和相关算子在核上是180度翻转的矩阵
					}
				}
				// 放入中间结果,计算所得的值与没有计算的值不能混用
				/*cout << sum << endl;*/
				temp.ptr<uchar>(i)[j] = sum;
				keep1.at<double>(i,j) = sum;
				/*cout << keep1.at<double>(i, j)<<endl;*/
				/*keep.ptr<float>(i)[j] = sum;
				cout << keep.ptr<float>(i)[j] << endl;*/
			}
			else keep1.at<double>(i, j) = temp.ptr<uchar>(i)[j];
		}
	}
	// 放入原图
	(*_src)= temp.clone();
	(*keep) = keep1.clone();
}

/* 获取高斯分布数组               (核大小， sigma值) */
double **getGaussianArray(int arr_size, double sigma)
{
	int i, j;
	// [1] 初始化权值数组
	double **array = new double*[arr_size];
	for (i = 0; i < arr_size; i++) {
		array[i] = new double[arr_size];
	}

	// [2] 高斯分布计算
	int center_i, center_j;
	center_i = center_j = arr_size / 2;
	double pi = 3.141592653589793;
	double sum = 0.0f;
	// [2-1] 高斯函数
	for (i = 0; i < arr_size; i++) {
		for (j = 0; j < arr_size; j++) {
			array[i][j] =
				//后面进行归一化，这部分可以不用
				//0.5f *pi*(sigma*sigma) * 
				exp(-(1.0f)* (((i - center_i)*(i - center_i) + (j - center_j)*(j - center_j)) /
(2.0f*sigma*sigma)));
			sum += array[i][j];
		}
	}
	// [2-2] 归一化求权值
	for (i = 0; i < arr_size; i++) {
		for (j = 0; j < arr_size; j++) {
			array[i][j] /= sum;
			printf(" [%.15f] ", array[i][j]);
		}
		printf("\n");
	}
	return array;
}



uchar Median(uchar n1, uchar n2, uchar n3, uchar n4, uchar n5, uchar n6, uchar n7, uchar n8, uchar n9)
{
	uchar *ptr = new uchar[9];	
	uchar temp;	*ptr = n1;  *(ptr + 1) = n2;	*(ptr + 2) = n3;	
	*(ptr + 3) = n4;	*(ptr + 4) = n5;	*(ptr + 5) = n6;	
	*(ptr + 6) = n7;	*(ptr + 7) = n8;	*(ptr + 8) = n9; 	
	for (int i = 0; i < 9; i++) 
	{ 
		for (int j = 0; j < 9; j++) 
		{ 
			if (ptr[i] > ptr[j]) 
			{
				temp = ptr[i];		
				ptr[i] = ptr[j];	
				ptr[j] = temp;
			}
		} 
	}	
	temp = ptr[4];
	delete[] ptr;
	return temp;

	//返回中值
}
/*窗口感知的引导高斯频域滤波 (待处理单通道图片, 高斯空间域滤波信息， 高斯数组大小(核大小)，窗口阈值，带宽 ) */
void gaussianpro(cv::Mat *_src,cv::Mat *keep, int w_size,double threshold,double width)
{
	int N = w_size;
	//窗口大小

	cv::Mat temp = (*_src).clone();
	// [1] 扫描

	for (int i = 0; i < (*_src).rows; i++)
	{
		for (int j = 0; j < (*_src).cols; j++)
		{
			// [2] 忽略边缘
			if (i > (N / 2) - 1 && j > (N / 2) - 1 &&
				i < (*_src).rows - (N / 2) && j < (*_src).cols - (N / 2))
			{
				//判断是否为异常点
				double sum = 0.0;
				double sum_all = 0.0;
				int num = 0;
				for (int x = 0; x < w_size; x++)
				{
					for (int y = 0; y < w_size; y++)
					{
						if(abs((*keep).at<double>(i - w_size / 2 + x, j - w_size / 2 + y) - (*keep).at<double>(i, j) )<=threshold)
							num++;
					}

				}
				/*cout << num << endl;*/
				if (num == 1)
					//是异常点，采用中值滤波
				{
					temp.ptr<uchar>(i)[j]= Median(temp.ptr<uchar>(i-1)[j-1], temp.ptr<uchar>(i - 1)[j], temp.ptr<uchar>(i - 1)[j +1],
						temp.ptr<uchar>(i)[j - 1], temp.ptr<uchar>(i)[j], temp.ptr<uchar>(i)[j+1],
						temp.ptr<uchar>(i + 1)[j - 1], temp.ptr<uchar>(i + 1)[j], temp.ptr<uchar>(i + 1)[j + 1]);
					
				}
				else
					//非异常点，采用非box型窗口引导滤波
				{
					for (int x = 0; x < w_size; x++)
					{
						for (int y = 0; y < w_size; y++)
						{
							int record_x = i - w_size / 2 + x;
							int record_y = j - w_size / 2 + y;
							double q = (*keep).at<double>(record_x, record_y);
							double p = (*keep).at<double>(i, j);
							if ((abs(q - p)) <= threshold)
							{
								
								sum +=
									exp(-(1.0f)* pow((q - p), 2) /
									(2.0f*width*width));
							/*	cout << sum<<endl;*/
								sum_all += (exp(-(1.0f)* pow((q - p), 2) /
									(2.0f*width*width)))*p;
							}
						}

					}

					sum_all = sum_all / sum;
					temp.ptr<uchar>(i)[j] = sum_all;
				}

			}
			/*cout <<  temp.ptr<double>(i)[j]<<endl;*/
			

		}
		/*cout << endl;*/
	}

	(*_src) = temp.clone();


}
//n 窗口大小，sigma
void myFilter(cv::Mat *src, cv::Mat *dst, int n, double sigma,double threshold,double width)
{
	// [1] 初始化
	*dst = (*src).clone();
	// [2] 彩色图片通道分离
	std::vector<cv::Mat> channels;
	cv::split(*dst, channels);
	// [3] 滤波

	// [3-1] 确定高斯正态矩阵
	double **array = getGaussianArray(n, sigma);

	// [3-2] 高斯滤波处理
	for (int i = 0; i < 3; i++) 
	{
		cv::Mat keep1((*src).rows, (*src).cols, CV_64FC1);
		gaussian(&channels[i],&keep1, array, n);
		/*for (int i = 0; i < (*src).rows; i++) {
			for (int j = 0; j < (*src).cols; j++) {
				cout<< keep1.at<double>(i, j) << endl;
			}
		}*/

		/*imshow("test", channels[1]);*/
		gaussianpro(&channels[i], &keep1, n, threshold,width);
		//gaussianpro(cv::Mat *_src,cv::Mat *keep, int w_size,double threshold,double width)

	}
	/*imshow("test", channels[0]);*/
	// [4] 合并返回
	cv::merge(channels, *dst);
	return;
}






int main(void)
{
	// [1] src读入图片
	cv::Mat src = cv::imread("why.jpg");
	// [2] dst目标图片
	cv::Mat dst;
	cv::Mat dst_2;
	//用BF预处理
	bilateralFilter(src,dst, 30,35,15);
	//public static void bilateralFilter(Mat src, Mat dst, int d, double sigmaColor, double sigmaSpace)


	imshow("pre", dst);
	// [3] 高斯滤波  sigma越大越平越模糊
	myFilter(&dst, &dst_2,9,0.3f,25.0f,15.0f);
	//myFilter(cv::Mat *src, cv::Mat *dst, int n, double sigma,double threshold,double width)
	// [4] 窗体显示
	cv::imshow("src", src);
	cv::imshow("dst_2", dst_2);
	cv::waitKey(0);
	cv::destroyAllWindows();

	cv::imwrite("test0.jpg", dst_2);
	return 0;
}

////添加滑动条
//
//// 全局变量的声明与初始化
//const float sigma_max = 300;
//int sigma_slider;
//double sgima_pro;
//
//void on_trackbar(int, void*)
//{
//	sgima_pro = (double)sigma_slider / 100;
//	// [1] src读入图片
//	cv::Mat src = cv::imread("fig5.jpg");
//	// [2] dst目标图片
//	cv::Mat dst;
//	// [3] 高斯滤波  sigma越大越平越模糊
//	myFilter(&src, &dst, 9, sgima_pro, 0.12f, 1.0f);
//	//myFilter(cv::Mat *src, cv::Mat *dst, int n, double sigma,double threshold,double width)
//	// [4] 窗体显示
//	cv::imshow("src", src);
//}
//
//
//int main(void)
//{
//	sigma_slider = 50;
//	namedWindow("test", WINDOW_AUTOSIZE);
//	char TrackbarName[50] = "miao";
//	createTrackbar(TrackbarName, "test", &sigma_slider, sigma_max, on_trackbar);
//
//	on_trackbar(sigma_slider, 0);
//
//	cv::waitKey(0);
//	cv::destroyAllWindows();
//	return 0;
//}


