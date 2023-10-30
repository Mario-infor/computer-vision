#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <iostream>
#include <math.h>
#include <list>

using namespace std;
using namespace cv;

class Class
{
private:
	int tag;
	int amount;
	Mat Mean;
	Mat Cov;

public:
	Class(int _tag, Mat _Mean, Mat _Cov)
	{
		tag = _tag;
		amount = 0;
		Mean = _Mean;
		Cov = _Cov;
	}

	~Class()
	{
		//std::cout << "Class was destroyed " << this->tag << endl;
	}

	void SetTag(int _tag)
	{
		this->tag = _tag;
	}

	int GetTag()
	{
		return this->tag;
	}

	void SetMean(Mat _Mean)
	{
		this->Mean = _Mean;
	}

	Mat GetMean()
	{
		return this->Mean;
	}

	void SetCov(Mat _Cov)
	{
		this->Cov = _Cov;
	}

	Mat GetCov()
	{
		return this->Cov;
	}

	void SetAmount(int _amount)
	{
		this->amount = _amount;
	}

	int GetAmount()
	{
		return this->amount;
	}
};

double dMahalanobisN(double a, double b, double am, double bm, double s1,
	double s2, double s3)
{
	/*
	   (a,b) vector to calculate its distance.
	   (am, bm)  Class average vector;
	   [s1,s3
	   s3,s2] Class covariance matrix.
	 */
	double det, idet, da, db, D;

	det = s1 * s2 - s3 * s3;
	idet = 1 / det;
	da = a - am;
	db = b - bm;
	D = log(det) + idet * (da * (da * s2 - db * s3) +
		db * (db * s1 - da * s3));
	return D;
}

double dEuclidean(double a, double b, double am, double bm, double s1,
	double s2, double s3)
{
	/*
	   (a,b) vector to calculate its distance.
	   (am, bm) Class average vector;
	   [s1,s3
	   s3,s2] Class covariance matrix.
	 */
	double da, db, D;

	da = a - am;
	db = b - bm;
	D = da * da + db * db;
	return D;
}

int MeanCov(Mat& image, Mat& Mask, Mat& mean, Mat& cov)
{
	float m[2], pm[2], Cv[3], icont;
	int cont;

	mean = Mat::zeros(2, 1, CV_32F);
	cov = Mat::zeros(2, 2, CV_32F);

	Mat_ < Vec3f >::iterator it, itEnd;
	Mat_ < uchar >::iterator itM;

	it = image.begin < Vec3f >();
	itM = Mask.begin < uchar >();
	itEnd = image.end < Vec3f >();
	m[0] = m[1] = 0;
	memset(m, 0, 2 * sizeof(float));
	for (cont = 0; it != itEnd; ++it, ++itM)
	{
		if ((*itM))
		{
			m[0] += (*it)[1];      // We accumulate component a of CIE-Lab
			m[1] += (*it)[2];      // We accumulate component b of CIE-Lab
			cont++;
		}
	}

	if (!cont)
		return -1;
	m[0] /= cont;
	m[1] /= cont;
	mean = Mat(2, 1, CV_32F, m).clone();

	if (cont < 2)
	{
		cov.at < float >(0, 0) = cov.at < float >(1, 1) = 1.;
		return -2;
	}
	it = image.begin < Vec3f >();
	itM = Mask.begin < uchar >();
	memset(Cv, 0, 3 * sizeof(float));
	for (; it != itEnd; ++it, ++itM)
	{
		if ((*itM))
		{
			pm[0] = (*it)[1] - m[0];
			pm[1] = (*it)[2] - m[1];
			Cv[0] += pm[0] * pm[0];
			Cv[1] += pm[1] * pm[1];
			Cv[2] += pm[0] * pm[1];
		}
	}
	icont = 1. / (cont - 1);
	Cv[0] *= icont;
	Cv[1] *= icont;
	Cv[2] *= icont;

	cov.at < float >(0, 0) = Cv[0];
	cov.at < float >(1, 1) = Cv[1];
	cov.at < float >(1, 0) = cov.at < float >(0, 1) = Cv[2];

	return cont;
}

bool classEndures(Mat classes, int findTag)
{
	for (int i = 0; i < classes.rows; i++)
	{
		for (int j = 0; j < classes.cols; j++)
		{
			if (classes.at<uchar>(i, j) == findTag)
			{
				return true;
			}
		}
	}
	return false;
}

void countClasses(vector<Class>& classesList, Mat classesMatrix)
{
	for (size_t i = 0; i < classesMatrix.rows; i++)
	{
		for (size_t j = 0; j < classesMatrix.cols; j++)
		{
			int tag = classesMatrix.at<uchar>(i, j);

			for (size_t k = 0; k < classesList.size(); k++)
			{
				if (classesList.at(k).GetTag() == tag)
				{
					classesList.at(k).SetAmount(classesList.at(k).GetAmount() + 1);
				}
			}
		}
	}

	for (size_t i = 0; i < classesList.size(); i++)
	{
		std::cout << "Class " << to_string(classesList.at(i).GetTag()) << ": " << to_string(classesList.at(i).GetAmount()) << endl;
	}
}



int main()
{
	Mat frame, fFrame, mMask, labFrame;
	Mat Mean, lastFrame, Cov, M, classes;
	Mat MeanOriginal, CovOriginal;
	int tagId = 0;
	int amountTagsCreate = 2;
	bool finish = false;
	bool covClass1AllCero = false;
	bool covClasss2AllCero = false;
	vector<Class> listClasses;
	vector<Class> listClassesEndures;
	vector<int> currentTags;
	vector<int> newTags;
	vector<int> tempNewTags;
	vector<Mat> imgList;
	vector<Mat> listMatClasses;
	frame = imread("Img/Jaguar2.png", 1);

	frame.convertTo(fFrame, CV_32FC3);
	fFrame /= 255;
	cv::cvtColor(fFrame, labFrame, COLOR_BGR2Lab);
	classes = Mat::ones(fFrame.size(), CV_8UC1);

	for (int i = 0; i < classes.rows; i++)
	{
		for (int j = 0; j < classes.cols; j++)
		{
			classes.at<uchar>(i, j) = 1 + rand() % 2;
		}
	}

	tagId += 1;
	newTags.push_back(tagId);
	tagId += 1;
	newTags.push_back(tagId);

	/*******************************************************************************************/
	while (newTags.size() > 0 && !finish)
	{
		currentTags.clear();
		currentTags.push_back(newTags.front());
		newTags.erase(newTags.begin());
		currentTags.push_back(newTags.front());
		newTags.erase(newTags.begin());

		mMask = Mat::zeros(fFrame.size(), CV_8UC1);
		for (int i = 0; i < classes.rows; i++)
		{
			for (int j = 0; j < classes.cols; j++)
			{
				if (classes.at<uchar>(i, j) == currentTags[0])
				{
					mMask.at<uchar>(i, j) = 1;
				}
			}
		}
		MeanCov(labFrame, mMask, Mean, Cov);
		if (countNonZero(Cov) < 1)
		{
			covClass1AllCero = true;
			finish = true;
		}
		std::cout << Mean << endl << endl;
		std::cout << Cov << endl << endl;
		Class clase1(currentTags[0], Mean.clone(), Cov.clone());
		listClasses.push_back(clase1);
		listClassesEndures.push_back(clase1);


		mMask = Mat::zeros(fFrame.size(), CV_8UC1);
		for (int i = 0; i < classes.rows; i++)
		{
			for (int j = 0; j < classes.cols; j++)
			{
				if (classes.at<uchar>(i, j) == currentTags[1])
				{
					mMask.at<uchar>(i, j) = 1;
				}
			}
		}
		MeanCov(labFrame, mMask, Mean, Cov);
		if (countNonZero(Cov) < 1)
		{
			covClasss2AllCero = true;
			finish = true;
		}
		std::cout << Mean << endl << endl;
		std::cout << Cov << endl << endl;
		Class clase2(currentTags[1], Mean.clone(), Cov.clone());
		listClasses.push_back(clase2);
		listClassesEndures.push_back(clase2);

		bool changes = false;
		while (!changes && !covClass1AllCero && !covClasss2AllCero)
		{
			for (int i = 0; i < labFrame.rows; i++)
			{
				for (int j = 0; j < labFrame.cols; j++)
				{
					if (classes.at<uchar>(i, j) == clase1.GetTag() || classes.at<uchar>(i, j) == clase2.GetTag())
					{
						double d1 = dEuclidean(labFrame.at<Vec3f>(i, j)[1], labFrame.at<Vec3f>(i, j)[2],
							clase1.GetMean().at<float>(0, 0), clase1.GetMean().at<float>(1, 0),
							clase1.GetCov().at<float>(0, 0), clase1.GetCov().at<float>(1, 1), clase1.GetCov().at<float>(0, 1));

						double d2 = dEuclidean(labFrame.at<Vec3f>(i, j)[1], labFrame.at<Vec3f>(i, j)[2],
							clase2.GetMean().at<float>(0, 0), clase2.GetMean().at<float>(1, 0),
							clase2.GetCov().at<float>(0, 0), clase2.GetCov().at<float>(1, 1), clase2.GetCov().at<float>(0, 1));

						if (d1 <= d2)
						{
							if (classes.at<uchar>(i, j) != clase1.GetTag())
							{
								changes = true;
								classes.at<uchar>(i, j) = clase1.GetTag();
							}
						}
						else
						{
							if (classes.at<uchar>(i, j) != clase2.GetTag())
							{
								changes = true;
								classes.at<uchar>(i, j) = clase2.GetTag();
							}
						}
					}

				}
			}

			if (changes)
			{
				mMask = Mat::zeros(fFrame.size(), CV_8UC1);
				for (int i = 0; i < classes.rows; i++)
				{
					for (int j = 0; j < classes.cols; j++)
					{
						if (classes.at<uchar>(i, j) == clase1.GetTag())
						{
							mMask.at<uchar>(i, j) = 1;
						}
					}
				}
				MeanCov(labFrame, mMask, Mean, Cov);
				if (countNonZero(Cov) < 1)
				{
					covClass1AllCero = true;
					if (countNonZero(Mean) < 1)
					{
						std::cout << "detener" << endl << endl;
					}
				}
				std::cout << Mean << endl << endl;
				std::cout << Cov << endl << endl;
				clase1.SetMean(Mean.clone());
				clase1.SetCov(Cov.clone());

				mMask = Mat::zeros(fFrame.size(), CV_8UC1);
				for (int i = 0; i < classes.rows; i++)
				{
					for (int j = 0; j < classes.cols; j++)
					{
						if (classes.at<uchar>(i, j) == clase2.GetTag())
						{
							mMask.at<uchar>(i, j) = 1;
						}
					}
				}
				MeanCov(labFrame, mMask, Mean, Cov);
				if (countNonZero(Cov) < 1)
				{
					covClasss2AllCero = true;
					if (countNonZero(Mean) < 1)
					{
						std::cout << "detener" << endl << endl;
					}
				}
				std::cout << Mean << endl << endl;
				std::cout << Cov << endl << endl;
				clase2.SetMean(Mean.clone());
				clase2.SetCov(Cov.clone());
				changes = false;
			}
			else
			{
				changes = true;
			}
		}

		changes = false;
		while (!changes && !covClass1AllCero && !covClasss2AllCero)
		{
			for (int i = 0; i < labFrame.rows; i++)
			{
				for (int j = 0; j < labFrame.cols; j++)
				{
					if (classes.at<uchar>(i, j) == clase1.GetTag() || classes.at<uchar>(i, j) == clase2.GetTag())
					{
						double d1 = dMahalanobisN(labFrame.at<Vec3f>(i, j)[1], labFrame.at<Vec3f>(i, j)[2],
							clase1.GetMean().at<float>(0, 0), clase1.GetMean().at<float>(1, 0),
							clase1.GetCov().at<float>(0, 0), clase1.GetCov().at<float>(1, 1), clase1.GetCov().at<float>(0, 1));

						double d2 = dMahalanobisN(labFrame.at<Vec3f>(i, j)[1], labFrame.at<Vec3f>(i, j)[2],
							clase2.GetMean().at<float>(0, 0), clase2.GetMean().at<float>(1, 0),
							clase2.GetCov().at<float>(0, 0), clase2.GetCov().at<float>(1, 1), clase2.GetCov().at<float>(0, 1));

						if (d1 <= d2)
						{
							if (classes.at<uchar>(i, j) != clase1.GetTag())
							{
								changes = true;
								classes.at<uchar>(i, j) = clase1.GetTag();
							}
						}
						else
						{
							if (classes.at<uchar>(i, j) != clase2.GetTag())
							{
								changes = true;
								classes.at<uchar>(i, j) = clase2.GetTag();
							}
						}
					}
				}
			}

			if (changes)
			{
				mMask = Mat::zeros(fFrame.size(), CV_8UC1);
				for (int i = 0; i < classes.rows; i++)
				{
					for (int j = 0; j < classes.cols; j++)
					{
						if (classes.at<uchar>(i, j) == clase1.GetTag())
						{
							mMask.at<uchar>(i, j) = 1;
						}
					}
				}
				MeanCov(labFrame, mMask, Mean, Cov);
				if (countNonZero(Cov) < 1)
				{
					covClass1AllCero = true;
					if (countNonZero(Mean) < 1)
					{
						std::cout << "detener" << endl << endl;
					}
				}
				std::cout << Mean << endl << endl;
				std::cout << Cov << endl << endl;
				clase1.SetMean(Mean.clone());
				clase1.SetCov(Cov.clone());

				mMask = Mat::zeros(fFrame.size(), CV_8UC1);
				for (int i = 0; i < classes.rows; i++)
				{
					for (int j = 0; j < classes.cols; j++)
					{
						if (classes.at<uchar>(i, j) == clase2.GetTag())
						{
							mMask.at<uchar>(i, j) = 1;
						}
					}
				}
				MeanCov(labFrame, mMask, Mean, Cov);
				if (countNonZero(Cov) < 1)
				{
					covClasss2AllCero = true;
					if (countNonZero(Mean) < 1)
					{
						std::cout << "detener" << endl << endl;
					}
				}
				std::cout << Mean << endl << endl;
				std::cout << Cov << endl << endl;
				clase2.SetMean(Mean.clone());
				clase2.SetCov(Cov.clone());
				changes = false;
			}
			else
			{
				changes = true;
			}
		}

		if (!finish)
		{
			if (covClass1AllCero && covClasss2AllCero)
			{
				for (size_t i = 0; i < listClassesEndures.size(); i++)
				{
					if (listClassesEndures.at(i).GetTag() == clase1.GetTag() || listClassesEndures.at(i).GetTag() == clase2.GetTag())
					{
						listClassesEndures.erase(listClassesEndures.begin() + i);
						i--;
					}
				}
				covClass1AllCero = false;
				covClasss2AllCero = false;
			}

			if (!classEndures(classes, clase1.GetTag()) || !classEndures(classes, clase2.GetTag()))
			{
				int etiquetaPadre = -1;

				for (int i = 0; i < classes.rows; i++)
				{
					for (int j = 0; j < classes.cols; j++)
					{
						if (classes.at<uchar>(i, j) == clase1.GetTag() || classes.at<uchar>(i, j) == clase2.GetTag())
						{
							if (etiquetaPadre == -1)
							{
								if (listMatClasses.size() == 0)
								{
									etiquetaPadre = clase1.GetTag();
								}
								else
								{
									etiquetaPadre = listMatClasses.at(listMatClasses.size() - 1).at<uchar>(i, j);
								}
							}
							classes.at<uchar>(i, j) = etiquetaPadre;
						}
					}
				}

				for (size_t i = 0; i < listClassesEndures.size(); i++)
				{
					if (listClassesEndures.at(i).GetTag() == clase1.GetTag() || listClassesEndures.at(i).GetTag() == clase2.GetTag() || listClassesEndures.at(i).GetTag() == etiquetaPadre)
					{
						listClassesEndures.erase(listClassesEndures.begin() + i);
						i--;
					}
				}

				if (listClassesEndures.size() == 0)
				{
					finish = true;
				}
			}
		}

		/*********************************************************************************************************************************/
		if (newTags.size() == 0 && !finish)
		{
			for (int i = 0; i < listClassesEndures.size(); i++)
			{
				if (!classEndures(classes, listClassesEndures.at(i).GetTag()))
				{
					listClassesEndures.erase(listClassesEndures.begin() + i);
					i--;
				}

				if (listClassesEndures.size() == 0)
				{
					finish = true;
				}
			}

			listMatClasses.push_back(classes.clone());
			Mat labFrameCopy = labFrame.clone();

			for (int i = 0; i < classes.rows; i++)
			{
				for (int j = 0; j < classes.cols; j++)
				{
					int tempEtiqueta = classes.at<uchar>(i, j);
					for (int k = 0; k < listClasses.size(); k++)
					{
						if (listClasses.at(k).GetTag() == tempEtiqueta)
						{
							labFrameCopy.at<Vec3f>(i, j)[1] = listClasses.at(k).GetMean().at<float>(0, 0);
							labFrameCopy.at<Vec3f>(i, j)[2] = listClasses.at(k).GetMean().at<float>(1, 0);
						}
					}
				}
			}

			Mat BGRFrame;
			Mat exitFrame;
			cv::cvtColor(labFrameCopy, BGRFrame, COLOR_Lab2BGR);
			BGRFrame *= 255;
			BGRFrame.convertTo(exitFrame, CV_8UC3);
			imgList.push_back(exitFrame);

			if (!finish)
			{
				amountTagsCreate = listClassesEndures.size() * 2;

				for (int i = 0; i < amountTagsCreate; i++)
				{
					tagId += 1;
					newTags.push_back(tagId);
				}

				tempNewTags = newTags;

				for (int i = 0; i < listClassesEndures.size(); i++)
				{
					int primerHijo = tempNewTags.front();
					tempNewTags.erase(tempNewTags.begin());

					int segundoHijo = tempNewTags.front();
					tempNewTags.erase(tempNewTags.begin());

					bool alternate = true;

					for (int j = 0; j < classes.rows; j++)
					{
						for (int k = 0; k < classes.cols; k++)
						{
							if (classes.at<uchar>(j, k) == listClassesEndures.at(i).GetTag())
							{
								if (alternate)
								{
									classes.at<uchar>(j, k) = primerHijo;
									alternate = !alternate;
								}
								else
								{
									classes.at<uchar>(j, k) = segundoHijo;
								}
							}
						}
					}
				}
			}
		}
		covClass1AllCero = false;
		covClasss2AllCero = false;
	}

	countClasses(listClasses, classes);

	for (int i = 0; i < imgList.size(); i++)
	{
		ostringstream cadena;
		cadena << "Ite " << i << " :";
		imshow(cadena.str(), imgList.at(i));
		waitKey(0);
	}

	return 0;
}


