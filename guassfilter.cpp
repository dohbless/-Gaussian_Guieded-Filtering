
/*********************************************/
/*С�᣺һ��ʼ���������Ϊ��Ҫ����Ҷ�任��
Ƶ�����˲��ȣ�Ȼ�������ܿ���qvq����
ֻ���������Ĺ�ʽ����һ�飬����Ч����̫��.
�ָ���������ʾ����һ��˫���˲�Ԥ����*/
/*********************************************/


#include <opencv2/core/core.hpp>                    
#include <opencv2/highgui/highgui.hpp>        
#include <opencv2/imgproc/imgproc.hpp>    
#include <opencv2/highgui/highgui.hpp>
#include <iostream>       

using namespace std;
using namespace cv;

/* ��˹�˲� (������ͨ��ͼƬ, ��˹�ֲ����飬 ��˹�����С(�˴�С) ) */
void gaussian(cv::Mat *_src,cv::Mat *keep, double **_array, int _size)
{
	cv::Mat keep1((*_src).rows, (*_src).cols, CV_64FC1);
	/*double[][]*/
	/*keep.convertTo(keep, CV_32FC1);*/
	cv::Mat temp = (*_src).clone();
	/*keep1 = (*_src).clone();*/
	// [1] ɨ��
	for (int i = 0; i < (*_src).rows; i++) {
		for (int j = 0; j < (*_src).cols; j++) {
			// [2] ���Ա�Ե
			if (i > (_size / 2) - 1 && j > (_size / 2) - 1 &&
				i < (*_src).rows - (_size / 2) && j < (*_src).cols - (_size / 2))
			{
				// [3] �ҵ�ͼ�������f(i,j),�������Ϊ����������Ķ���
				//     ����Ϊ���Ĳο��� �������=>��˹����180��ת�����
				//     x y �������˵�Ȩֵ����   i j ����ͼ�����������
				//     �������     (f*g)(i,j) = f(i-k,j-l)g(k,l)          f����ͼ������ g�����
				//     ����˲ο��� (f*g)(i,j) = f(i-(k-ai), j-(l-aj))g(k,l)   ai,aj �˲ο���
				//     ��Ȩ���  ע�⣺�˵�����������0,0���
				double sum = 0.0;
				for (int k = 0; k < _size; k++)
				{
					for (int l = 0; l < _size; l++) 
					{
						sum += (*_src).ptr<uchar>(i - k + (_size / 2))[j - l + (_size / 2)] * _array[k][l];
						//mat.ptr<type>(row)[col]
						//������Ӻ���������ں�����180�ȷ�ת�ľ���
					}
				}
				// �����м���,�������õ�ֵ��û�м����ֵ���ܻ���
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
	// ����ԭͼ
	(*_src)= temp.clone();
	(*keep) = keep1.clone();
}

/* ��ȡ��˹�ֲ�����               (�˴�С�� sigmaֵ) */
double **getGaussianArray(int arr_size, double sigma)
{
	int i, j;
	// [1] ��ʼ��Ȩֵ����
	double **array = new double*[arr_size];
	for (i = 0; i < arr_size; i++) {
		array[i] = new double[arr_size];
	}

	// [2] ��˹�ֲ�����
	int center_i, center_j;
	center_i = center_j = arr_size / 2;
	double pi = 3.141592653589793;
	double sum = 0.0f;
	// [2-1] ��˹����
	for (i = 0; i < arr_size; i++) {
		for (j = 0; j < arr_size; j++) {
			array[i][j] =
				//������й�һ�����ⲿ�ֿ��Բ���
				//0.5f *pi*(sigma*sigma) * 
				exp(-(1.0f)* (((i - center_i)*(i - center_i) + (j - center_j)*(j - center_j)) /
(2.0f*sigma*sigma)));
			sum += array[i][j];
		}
	}
	// [2-2] ��һ����Ȩֵ
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

	//������ֵ
}
/*���ڸ�֪��������˹Ƶ���˲� (������ͨ��ͼƬ, ��˹�ռ����˲���Ϣ�� ��˹�����С(�˴�С)��������ֵ������ ) */
void gaussianpro(cv::Mat *_src,cv::Mat *keep, int w_size,double threshold,double width)
{
	int N = w_size;
	//���ڴ�С

	cv::Mat temp = (*_src).clone();
	// [1] ɨ��

	for (int i = 0; i < (*_src).rows; i++)
	{
		for (int j = 0; j < (*_src).cols; j++)
		{
			// [2] ���Ա�Ե
			if (i > (N / 2) - 1 && j > (N / 2) - 1 &&
				i < (*_src).rows - (N / 2) && j < (*_src).cols - (N / 2))
			{
				//�ж��Ƿ�Ϊ�쳣��
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
					//���쳣�㣬������ֵ�˲�
				{
					temp.ptr<uchar>(i)[j]= Median(temp.ptr<uchar>(i-1)[j-1], temp.ptr<uchar>(i - 1)[j], temp.ptr<uchar>(i - 1)[j +1],
						temp.ptr<uchar>(i)[j - 1], temp.ptr<uchar>(i)[j], temp.ptr<uchar>(i)[j+1],
						temp.ptr<uchar>(i + 1)[j - 1], temp.ptr<uchar>(i + 1)[j], temp.ptr<uchar>(i + 1)[j + 1]);
					
				}
				else
					//���쳣�㣬���÷�box�ʹ��������˲�
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
//n ���ڴ�С��sigma
void myFilter(cv::Mat *src, cv::Mat *dst, int n, double sigma,double threshold,double width)
{
	// [1] ��ʼ��
	*dst = (*src).clone();
	// [2] ��ɫͼƬͨ������
	std::vector<cv::Mat> channels;
	cv::split(*dst, channels);
	// [3] �˲�

	// [3-1] ȷ����˹��̬����
	double **array = getGaussianArray(n, sigma);

	// [3-2] ��˹�˲�����
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
	// [4] �ϲ�����
	cv::merge(channels, *dst);
	return;
}






int main(void)
{
	// [1] src����ͼƬ
	cv::Mat src = cv::imread("why.jpg");
	// [2] dstĿ��ͼƬ
	cv::Mat dst;
	cv::Mat dst_2;
	//��BFԤ����
	bilateralFilter(src,dst, 30,35,15);
	//public static void bilateralFilter(Mat src, Mat dst, int d, double sigmaColor, double sigmaSpace)


	imshow("pre", dst);
	// [3] ��˹�˲�  sigmaԽ��ԽƽԽģ��
	myFilter(&dst, &dst_2,9,0.3f,25.0f,15.0f);
	//myFilter(cv::Mat *src, cv::Mat *dst, int n, double sigma,double threshold,double width)
	// [4] ������ʾ
	cv::imshow("src", src);
	cv::imshow("dst_2", dst_2);
	cv::waitKey(0);
	cv::destroyAllWindows();

	cv::imwrite("test0.jpg", dst_2);
	return 0;
}

////��ӻ�����
//
//// ȫ�ֱ������������ʼ��
//const float sigma_max = 300;
//int sigma_slider;
//double sgima_pro;
//
//void on_trackbar(int, void*)
//{
//	sgima_pro = (double)sigma_slider / 100;
//	// [1] src����ͼƬ
//	cv::Mat src = cv::imread("fig5.jpg");
//	// [2] dstĿ��ͼƬ
//	cv::Mat dst;
//	// [3] ��˹�˲�  sigmaԽ��ԽƽԽģ��
//	myFilter(&src, &dst, 9, sgima_pro, 0.12f, 1.0f);
//	//myFilter(cv::Mat *src, cv::Mat *dst, int n, double sigma,double threshold,double width)
//	// [4] ������ʾ
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


