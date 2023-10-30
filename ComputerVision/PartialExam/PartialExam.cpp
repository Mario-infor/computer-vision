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
	for (int i = 0; i < clases.rows; i++)
	{
		for (int j = 0; j < clases.cols; j++)
		{
			if (clases.at<uchar>(i, j) == etiquetaBuscar)
			{
				return true;
			}
		}
	}
	return false;
}

void contarClases(vector<Class>& listaClases, Mat matrizClases)
{
	for (size_t i = 0; i < matrizClases.rows; i++)
	{
		for (size_t j = 0; j < matrizClases.cols; j++)
		{
			int etiqueta = matrizClases.at<uchar>(i, j);

			for (size_t k = 0; k < listaClases.size(); k++)
			{
				if (listaClases.at(k).GetTag() == etiqueta)
				{
					listaClases.at(k).SetAmount(listaClases.at(k).GetAmount() + 1);
				}
			}
		}
	}

	for (size_t i = 0; i < listaClases.size(); i++)
	{
		std::cout << "Clase " << to_string(listaClases.at(i).GetTag()) << ": " << to_string(listaClases.at(i).GetAmount()) << endl;
	}
}



int main()
{
	Mat frame, fFrame, mMask, labFrame;
	Mat Mean, frameAnterior, Cov, M, clases;
	Mat MeanOriginal, CovOriginal;
	int identificadorEtiqueta = 0;
	int cantEtiquetasCrear = 2;
	bool terminar = false;
	bool covClase1TodoCero = false;
	bool covClase2TodoCero = false;
	vector<Class> listaClases;
	vector<Class> listaClasesPerduran;
	vector<int> etiquetasActuales;
	vector<int> etiquetasNuevas;
	vector<int> tempEtiquetasNuevas;
	vector<Mat> listaImagenes;
	vector<Mat> listaMatClases;
	frame = imread("Img/Jaguar2.png", 1);

	frame.convertTo(fFrame, CV_32FC3);
	fFrame /= 255;
	cvtColor(fFrame, labFrame, COLOR_BGR2Lab);
	clases = Mat::ones(fFrame.size(), CV_8UC1);

	for (int i = 0; i < clases.rows; i++)
	{
		for (int j = 0; j < clases.cols; j++)
		{
			clases.at<uchar>(i, j) = 1 + rand() % 2;
		}
	}

	identificadorEtiqueta += 1;
	etiquetasNuevas.push_back(identificadorEtiqueta);
	identificadorEtiqueta += 1;
	etiquetasNuevas.push_back(identificadorEtiqueta);

	/*******************************************************************************************/
	while (etiquetasNuevas.size() > 0 && !terminar)
	{
		etiquetasActuales.clear();
		etiquetasActuales.push_back(etiquetasNuevas.front());
		etiquetasNuevas.erase(etiquetasNuevas.begin());
		etiquetasActuales.push_back(etiquetasNuevas.front());
		etiquetasNuevas.erase(etiquetasNuevas.begin());

		mMask = Mat::zeros(fFrame.size(), CV_8UC1);
		for (int i = 0; i < clases.rows; i++)
		{
			for (int j = 0; j < clases.cols; j++)
			{
				if (clases.at<uchar>(i, j) == etiquetasActuales[0])
				{
					mMask.at<uchar>(i, j) = 1;
				}
			}
		}
		MeanCov(labFrame, mMask, Mean, Cov);
		if (countNonZero(Cov) < 1)
		{
			covClase1TodoCero = true;
			terminar = true;
		}
		std::cout << Mean << endl << endl;
		std::cout << Cov << endl << endl;
		Class clase1(etiquetasActuales[0], Mean.clone(), Cov.clone());
		listaClases.push_back(clase1);
		listaClasesPerduran.push_back(clase1);


		mMask = Mat::zeros(fFrame.size(), CV_8UC1);
		for (int i = 0; i < clases.rows; i++)
		{
			for (int j = 0; j < clases.cols; j++)
			{
				if (clases.at<uchar>(i, j) == etiquetasActuales[1])
				{
					mMask.at<uchar>(i, j) = 1;
				}
			}
		}
		MeanCov(labFrame, mMask, Mean, Cov);
		if (countNonZero(Cov) < 1)
		{
			covClase2TodoCero = true;
			terminar = true;
		}
		std::cout << Mean << endl << endl;
		std::cout << Cov << endl << endl;
		Class clase2(etiquetasActuales[1], Mean.clone(), Cov.clone());
		listaClases.push_back(clase2);
		listaClasesPerduran.push_back(clase2);

		bool huboCambio = false;
		while (!huboCambio && !covClase1TodoCero && !covClase2TodoCero)
		{
			for (int i = 0; i < labFrame.rows; i++)
			{
				for (int j = 0; j < labFrame.cols; j++)
				{
					if (clases.at<uchar>(i, j) == clase1.GetTag() || clases.at<uchar>(i, j) == clase2.GetTag())
					{
						double d1 = dEuclidean(labFrame.at<Vec3f>(i, j)[1], labFrame.at<Vec3f>(i, j)[2],
							clase1.GetMean().at<float>(0, 0), clase1.GetMean().at<float>(1, 0),
							clase1.GetCov().at<float>(0, 0), clase1.GetCov().at<float>(1, 1), clase1.GetCov().at<float>(0, 1));

						double d2 = dEuclidean(labFrame.at<Vec3f>(i, j)[1], labFrame.at<Vec3f>(i, j)[2],
							clase2.GetMean().at<float>(0, 0), clase2.GetMean().at<float>(1, 0),
							clase2.GetCov().at<float>(0, 0), clase2.GetCov().at<float>(1, 1), clase2.GetCov().at<float>(0, 1));

						if (d1 <= d2)
						{
							if (clases.at<uchar>(i, j) != clase1.GetTag())
							{
								huboCambio = true;
								clases.at<uchar>(i, j) = clase1.GetTag();
							}
						}
						else
						{
							if (clases.at<uchar>(i, j) != clase2.GetTag())
							{
								huboCambio = true;
								clases.at<uchar>(i, j) = clase2.GetTag();
							}
						}
					}

				}
			}

			if (huboCambio)
			{
				mMask = Mat::zeros(fFrame.size(), CV_8UC1);
				for (int i = 0; i < clases.rows; i++)
				{
					for (int j = 0; j < clases.cols; j++)
					{
						if (clases.at<uchar>(i, j) == clase1.GetTag())
						{
							mMask.at<uchar>(i, j) = 1;
						}
					}
				}
				MeanCov(labFrame, mMask, Mean, Cov);
				if (countNonZero(Cov) < 1)
				{
					covClase1TodoCero = true;
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
				for (int i = 0; i < clases.rows; i++)
				{
					for (int j = 0; j < clases.cols; j++)
					{
						if (clases.at<uchar>(i, j) == clase2.GetTag())
						{
							mMask.at<uchar>(i, j) = 1;
						}
					}
				}
				MeanCov(labFrame, mMask, Mean, Cov);
				if (countNonZero(Cov) < 1)
				{
					covClase2TodoCero = true;
					if (countNonZero(Mean) < 1)
					{
						std::cout << "detener" << endl << endl;
					}
				}
				std::cout << Mean << endl << endl;
				std::cout << Cov << endl << endl;
				clase2.SetMean(Mean.clone());
				clase2.SetCov(Cov.clone());
				huboCambio = false;
			}
			else
			{
				huboCambio = true;
			}
		}

		huboCambio = false;
		while (!huboCambio && !covClase1TodoCero && !covClase2TodoCero)
		{
			for (int i = 0; i < labFrame.rows; i++)
			{
				for (int j = 0; j < labFrame.cols; j++)
				{
					if (clases.at<uchar>(i, j) == clase1.GetTag() || clases.at<uchar>(i, j) == clase2.GetTag())
					{
						double d1 = dMahalanobisN(labFrame.at<Vec3f>(i, j)[1], labFrame.at<Vec3f>(i, j)[2],
							clase1.GetMean().at<float>(0, 0), clase1.GetMean().at<float>(1, 0),
							clase1.GetCov().at<float>(0, 0), clase1.GetCov().at<float>(1, 1), clase1.GetCov().at<float>(0, 1));

						double d2 = dMahalanobisN(labFrame.at<Vec3f>(i, j)[1], labFrame.at<Vec3f>(i, j)[2],
							clase2.GetMean().at<float>(0, 0), clase2.GetMean().at<float>(1, 0),
							clase2.GetCov().at<float>(0, 0), clase2.GetCov().at<float>(1, 1), clase2.GetCov().at<float>(0, 1));

						if (d1 <= d2)
						{
							if (clases.at<uchar>(i, j) != clase1.GetTag())
							{
								huboCambio = true;
								clases.at<uchar>(i, j) = clase1.GetTag();
							}
						}
						else
						{
							if (clases.at<uchar>(i, j) != clase2.GetTag())
							{
								huboCambio = true;
								clases.at<uchar>(i, j) = clase2.GetTag();
							}
						}
					}
				}
			}

			if (huboCambio)
			{
				mMask = Mat::zeros(fFrame.size(), CV_8UC1);
				for (int i = 0; i < clases.rows; i++)
				{
					for (int j = 0; j < clases.cols; j++)
					{
						if (clases.at<uchar>(i, j) == clase1.GetTag())
						{
							mMask.at<uchar>(i, j) = 1;
						}
					}
				}
				MeanCov(labFrame, mMask, Mean, Cov);
				if (countNonZero(Cov) < 1)
				{
					covClase1TodoCero = true;
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
				for (int i = 0; i < clases.rows; i++)
				{
					for (int j = 0; j < clases.cols; j++)
					{
						if (clases.at<uchar>(i, j) == clase2.GetTag())
						{
							mMask.at<uchar>(i, j) = 1;
						}
					}
				}
				MeanCov(labFrame, mMask, Mean, Cov);
				if (countNonZero(Cov) < 1)
				{
					covClase2TodoCero = true;
					if (countNonZero(Mean) < 1)
					{
						std::cout << "detener" << endl << endl;
					}
				}
				std::cout << Mean << endl << endl;
				std::cout << Cov << endl << endl;
				clase2.SetMean(Mean.clone());
				clase2.SetCov(Cov.clone());
				huboCambio = false;
			}
			else
			{
				huboCambio = true;
			}
		}

		if (!terminar)
		{
			if (covClase1TodoCero && covClase2TodoCero)
			{
				for (size_t i = 0; i < listaClasesPerduran.size(); i++)
				{
					if (listaClasesPerduran.at(i).GetTag() == clase1.GetTag() || listaClasesPerduran.at(i).GetTag() == clase2.GetTag())
					{
						listaClasesPerduran.erase(listaClasesPerduran.begin() + i);
						i--;
					}
				}
				covClase1TodoCero = false;
				covClase2TodoCero = false;
			}

			if (!clasePerdura(clases, clase1.GetTag()) || !clasePerdura(clases, clase2.GetTag()))
			{
				int etiquetaPadre = -1;

				for (int i = 0; i < clases.rows; i++)
				{
					for (int j = 0; j < clases.cols; j++)
					{
						if (clases.at<uchar>(i, j) == clase1.GetTag() || clases.at<uchar>(i, j) == clase2.GetTag())
						{
							if (etiquetaPadre == -1)
							{
								if (listaMatClases.size() == 0)
								{
									etiquetaPadre = clase1.GetTag();
								}
								else
								{
									etiquetaPadre = listaMatClases.at(listaMatClases.size() - 1).at<uchar>(i, j);
								}
							}
							clases.at<uchar>(i, j) = etiquetaPadre; 
						}
					}
				}

				for (size_t i = 0; i < listaClasesPerduran.size(); i++)
				{
					if (listaClasesPerduran.at(i).GetTag() == clase1.GetTag() || listaClasesPerduran.at(i).GetTag() == clase2.GetTag() || listaClasesPerduran.at(i).GetTag() == etiquetaPadre)
					{
						listaClasesPerduran.erase(listaClasesPerduran.begin() + i);
						i--;
					}
				}

				if (listaClasesPerduran.size() == 0)
				{
					terminar = true;
				}
			}
		}

		/*********************************************************************************************************************************/
		if (etiquetasNuevas.size() == 0 && !terminar)
		{
			for (int i = 0; i < listaClasesPerduran.size(); i++)
			{
				if (!clasePerdura(clases, listaClasesPerduran.at(i).GetTag()))
				{
					listaClasesPerduran.erase(listaClasesPerduran.begin() + i);
					i--;
				}

				if (listaClasesPerduran.size() == 0)
				{
					terminar = true;
				}
			}

			listaMatClases.push_back(clases.clone());
			Mat labFrameCopy = labFrame.clone();

			for (int i = 0; i < clases.rows; i++)
			{
				for (int j = 0; j < clases.cols; j++)
				{
					int tempEtiqueta = clases.at<uchar>(i, j);
					for (int k = 0; k < listaClases.size(); k++)
					{
						if (listaClases.at(k).GetTag() == tempEtiqueta)
						{
							labFrameCopy.at<Vec3f>(i, j)[1] = listaClases.at(k).GetMean().at<float>(0, 0);
							labFrameCopy.at<Vec3f>(i, j)[2] = listaClases.at(k).GetMean().at<float>(1, 0);
						}
					}
				}
			}

			Mat BGRFrame;
			Mat exitFrame;
			cvtColor(labFrameCopy, BGRFrame, COLOR_Lab2BGR);
			BGRFrame *= 255;
			BGRFrame.convertTo(exitFrame, CV_8UC3);
			listaImagenes.push_back(exitFrame);

			if (!terminar)
			{
				cantEtiquetasCrear = listaClasesPerduran.size() * 2;

				for (int i = 0; i < cantEtiquetasCrear; i++)
				{
					identificadorEtiqueta += 1;
					etiquetasNuevas.push_back(identificadorEtiqueta);
				}

				tempEtiquetasNuevas = etiquetasNuevas;

				for (int i = 0; i < listaClasesPerduran.size(); i++)
				{
					int primerHijo = tempEtiquetasNuevas.front();
					tempEtiquetasNuevas.erase(tempEtiquetasNuevas.begin());

					int segundoHijo = tempEtiquetasNuevas.front();
					tempEtiquetasNuevas.erase(tempEtiquetasNuevas.begin());

					bool alternar = true;

					for (int j = 0; j < clases.rows; j++)
					{
						for (int k = 0; k < clases.cols; k++)
						{
							if (clases.at<uchar>(j, k) == listaClasesPerduran.at(i).GetTag())
							{
								if (alternar)
								{
									clases.at<uchar>(j, k) = primerHijo;
									alternar = !alternar;
								}
								else
								{
									clases.at<uchar>(j, k) = segundoHijo;
								}
							}
						}
					}
				}
			}
		}
		covClase1TodoCero = false;
		covClase2TodoCero = false;
	}

	contarClases(listaClases, clases);

	for (int i = 0; i < listaImagenes.size(); i++)
	{
		ostringstream cadena;
		cadena << "Iteracion " << i << " :";
		imshow(cadena.str(), listaImagenes.at(i));
		waitKey(0);
	}

	return 0;
}


