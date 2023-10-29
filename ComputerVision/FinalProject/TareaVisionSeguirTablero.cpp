#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <string>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

void printMat(const cv::Mat & mat) {
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++)
            std::cout << mat.at<double>(i, j) << "\t";
        std::cout << std::endl;
    }
	cout << std::endl;
}

Mat draw(Mat img, vector<Point2f> corners, vector<Point2f> imgpts) {
	
	Point2f corner(corners[0].x, corners[0].y);

	line(img, corner, imgpts[0], Scalar(0, 0, 255), 10);
	line(img, corner, imgpts[1], Scalar(0, 255, 0), 10);
	line(img, corner, imgpts[2], Scalar(255, 0, 0), 10);

	return img;
}


Mat drawCube(Mat img, vector<Point2f> corners, vector<Point2f> imgpts) {
	// Convert imgpts to a vector of type cv::Point with integers numbers
	std::vector<cv::Point> intImgPts;
	for (const auto& pt : imgpts) {
		intImgPts.push_back(cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
	}

	// Draw ground floor in green
	std::vector<std::vector<cv::Point>> groundContours = { std::vector<cv::Point>(intImgPts.begin(), intImgPts.begin() + 4) };
	cv::drawContours(img, groundContours, -1, cv::Scalar(0, 255, 0), -3);

	// Dibujar pilares en color azul
	for (int i = 0; i < 4; ++i) {
		cv::line(img, intImgPts[i], intImgPts[i + 4], cv::Scalar(255), 3);
	}

	// Dibujar top layer en color rojo
	std::vector<std::vector<cv::Point>> topContours = { std::vector<cv::Point>(intImgPts.begin() + 4, intImgPts.end()) };
	cv::drawContours(img, topContours, -1, cv::Scalar(0, 0, 255), 3);

	return img;
}

void proj(cv::Mat& u, cv::Mat& v, cv::Mat& pv)
{
	pv = u.dot(v) * u / u.dot(u);
}

void gramSchmidtOrthogonalizationMethod(cv::Mat& H)
{
	cv::Mat v1, v2, v3, u[3];

	v1 = H(cv::Rect(0, 0, 1, 3));
	v2 = H(cv::Rect(1, 0, 1, 3));
	v3 = H(cv::Rect(2, 0, 1, 3));
	u[0] = v1;
	proj(u[0], v2, u[1]);
	u[1] = v2 - u[1];
	proj(u[0], v3, v1);
	proj(u[1], v3, v2);
	u[2] = v3 - v1 - v2;

	u[0] = u[0] / sqrt(u[0].dot(u[0]));
	u[1] = u[1] / sqrt(u[1].dot(u[1]));
	u[2] = u[2] / sqrt(u[2].dot(u[2]));
	cv::hconcat(u, 3, H);
}

void convertVecPointToMat(const std::vector<cv::Point3f>& vecPoint, cv::Mat& matPoint) {
	int count = int(vecPoint.size());
	matPoint = cv::Mat(4, count, CV_64FC1);
	for (int i = 0; i < count; ++i) {
		matPoint.at<double>(0, i) = vecPoint[i].x;
		matPoint.at<double>(1, i) = vecPoint[i].y;
		matPoint.at<double>(2, i) = vecPoint[i].z;
		matPoint.at<double>(3, i) = 1.;
	}
}

void convertVecPointToMatFloat(const std::vector<cv::Point3f>& vecPoint, cv::Mat& matPoint) {
	int count = int(vecPoint.size());
	matPoint = cv::Mat(4, count, CV_32F);
	for (int i = 0; i < count; ++i) {
		matPoint.at<float>(0, i) = vecPoint[i].x;
		matPoint.at<float>(1, i) = vecPoint[i].y;
		matPoint.at<float>(2, i) = vecPoint[i].z;
		matPoint.at<float>(3, i) = 1.;
	}
}

void convertVecPointToMat(const vector<Point2f>& vecPoint, Mat& matPoint)
{
	int count = int(vecPoint.size());

	matPoint = cv::Mat(3, count, CV_64FC1);
	for (int i = 0; i < count; ++i)
	{
		matPoint.at<double>(0, i) = vecPoint[i].x;
		matPoint.at<double>(1, i) = vecPoint[i].y;
		matPoint.at<double>(2, i) = 1.;
	}
}

void convertMatToVecPoint(const cv::Mat& matPoint, std::vector<cv::Point2f>& vecPoint)
{
	int i;

	vecPoint.clear();
	for (i = 0; i < matPoint.cols; ++i)
	{
		double w = matPoint.at<double>(2, i);
		vecPoint.emplace_back(
			matPoint.at<double>(0, i) / w,
			matPoint.at<double>(1, i) / w
		);

	}
}

void convertMatToVecPoint(const cv::Mat& matPoint, std::vector<cv::Point3f>& vecPoint)
{
	int i;
	vecPoint.clear();
	for (i = 0; i < matPoint.cols; ++i)
		vecPoint.push_back(cv::Point3f(matPoint.at<double>(0, i), matPoint.at<double>(1, i), matPoint.at<double>(2, i)));
}

void homogenous2Cartesian(cv::Mat& matPoint)
{
	int i;
	double val;
	for (i = 0; i < matPoint.cols; ++i)
	{
		val = matPoint.at<double>(2, i);
		matPoint.at<double>(0, i) /= val;
		matPoint.at<double>(1, i) /= val;
	}
}

void printXYPoints(vector<Point2f> corners) 
{
	vector<float> allX;
	vector<float> allY;

	for (size_t i = 0; i < corners.size(); i++)
	{
		allX.push_back(corners[i].x);
		allY.push_back(corners[i].y);
	}

	cout << "[";
	for (size_t i = 0; i < allX.size(); i++)
	{
		cout << allX[i] << ", ";
	}
	cout << "]" << endl;

	cout << "[";
	for (size_t i = 0; i < allY.size(); i++)
	{
		cout << allY[i] << ", ";
	}
	cout << "]" << endl;
}

void printXYPoints(vector<Point3f> corners)
{
	vector<float> allX;
	vector<float> allY;

	for (size_t i = 0; i < corners.size(); i++)
	{
		allX.push_back(corners[i].x);
		allY.push_back(corners[i].y);
	}

	cout << "[";
	for (size_t i = 0; i < allX.size(); i++)
	{
		cout << allX[i] << ", ";
	}
	cout << "]" << endl;

	cout << "[";
	for (size_t i = 0; i < allY.size(); i++)
	{
		cout << allY[i] << ", ";
	}
	cout << "]" << endl;
}

void invertRotationAroundZ(Mat& vector)
{
	double x = vector.at<double>(0, 0);
	double y = vector.at<double>(1, 0);
	double z = vector.at<double>(2, 0);

	// Calcular el ángulo de rotación original alrededor del eje Z
	double angle = std::atan2(y, x);

	// Invertir el signo del ángulo de rotación
	double inverted_angle = -angle;

	// Aplicar la rotación inversa utilizando el nuevo ángulo de rotación invertido
	double inverted_x = std::cos(inverted_angle) * x - std::sin(inverted_angle) * y;
	double inverted_y = std::sin(inverted_angle) * x + std::cos(inverted_angle) * y;

	vector.at<double>(0, 0) = inverted_x;
	vector.at<double>(0, 0) = inverted_y;
	vector.at<double>(0, 0) = z;

}


int main() {

	// dimensions of the chessboard and size of the little squares
	int rowsChessBoard = 6;
	int colsChessBoard = 8;
	float squareSize = 0.02845;

	// K matrix for de camera calibration
	Mat matrixK = (Mat_<double>(3, 3) <<
		7.6808743880079578e+02, 0, 4.2089026071365419e+02,
		0, 7.6808743880079578e+02, 3.0855623821214846e+02,
		0, 0, 1);

	Mat iK;
	invert(matrixK, iK);

	float lineLenght = 2 * squareSize;

	// matrix for the axis we will draw on the picture
	Mat axis = (Mat_<float>(3, 3) <<
		lineLenght, 0, 0,
		0, lineLenght, 0,
		0, 0, -lineLenght);

	Mat axisCube = (Mat_<float>(8, 3) <<
		0, 0, 0,
		0, lineLenght, 0,
		lineLenght, lineLenght, 0,
		lineLenght, 0 , 0,
		0, 0, -lineLenght,
		0, lineLenght, -lineLenght,
		lineLenght, lineLenght, -lineLenght,
		lineLenght, 0, -lineLenght);

	vector<Point3f> point3D;

	// for loop to fill up de 3D point of the chessboard
	for (size_t i = 0; i < rowsChessBoard; i++)
	{
		for (int j = 0; j < colsChessBoard; j++)
		{
			point3D.push_back(Point3f(j * squareSize, i * squareSize, 0));
		}
	}


	printXYPoints(point3D);

	//printXYPoints(point3D);

	// Capture the video file and verify that it was load correctly
	VideoCapture camera = VideoCapture("C:/Mario/Maestría/Clases Maestría/2do Semestre/Vision Computacional/Tarea Seguir Tablero/TareasVisionComputacional/TareaVisionSeguirTablero/Chessboard.mp4");
	if (!camera.isOpened()) {
		cout << "No se puede abrir el dispositivo de video" << endl;
		return 0;
	}

	vector<Point2f> corners;			// chessboard points vector, output of findChessboardCorners() method
	vector<Point2f> imgPoints;
	vector<Point2f> Points3DPlanar;
	vector<float> rotationVector;		// rotation vector, output of solvePnP() method
	vector<float> traslationVector;		// traslation vector, output of solvePnP() method
	Mat distCoeffs;					// empty vector to ignore the distorsion coefficients of the camera

	namedWindow("Chessboard");
	Mat frame;
	Mat gray;

	Mat gFrom;
	Mat RChessBoardFrom;
	Mat TChessBoardFrom;

	Mat gTo;
	Mat RChessBoardTo;
	Mat TChessBoardTo;

	//double deltaT = 1. / 30;
	double deltaT = 1;

	int contador = 0;

	while (true) {

		bool retval = camera.read(frame);

		//Mat resizedImage;
		//resize(frame, resizedImage, cv::Size(854, 480));

		//cvtColor(frame, gray, cv::COLOR_RGB2GRAY);

		if (retval) {
			// openCV method that finds the corners inside the ChessBoard
			bool retval = findChessboardCorners(frame, cv::Size(8, 6), corners);

			if (contador < 5)
			{
				cout << "ChessBoardPoints " << contador << endl;
				cout << corners << endl << endl;
			}

			if (retval) {

				//drawChessboardCorners(frame, cv::Size(8, 6), corners, retval);

				// openCV method that outputs the rotation and traslation vectors of the camera using the points from the ChessBoard
				solvePnP(point3D, corners, matrixK, distCoeffs, rotationVector, traslationVector);
				
				Points3DPlanar.clear();
				for (size_t i = 0; i < point3D.size(); i++)
				{
					Points3DPlanar.push_back(Point2f(point3D[i].x, point3D[i].y));
				}

				//vector<Point2f> imagePoints;
				//undistortPoints(corners, imagePoints, matrixK, distCoeffs);
				
				Mat cornersMat = Mat::zeros(2, corners.size(), CV_32F);
				
				convertVecPointToMat(corners, cornersMat);
				cornersMat = iK * cornersMat;

				vector<Point2f> convertedCorners(corners.size());

				convertMatToVecPoint(cornersMat, convertedCorners);

				//printXYPoints(convertedCorners);

				Mat H = findHomography(Points3DPlanar, convertedCorners);
				
				if (contador < 5)
				{
					cout << "Homografy " << contador << endl;
					cout << H << endl << endl;
				}

				contador++;
				//printMat(H);

				//Mat H = findHomography(Points3DPlanar, imagePoints);

				H /= cv::norm(H.col(0));

				//printMat(H);

				//cout << cv::norm(H.col(0)) << endl;

				Mat tvec = H.col(2).clone();
				
				//printMat(tvec);

				//cout << cv::norm(H.col(1)) << endl;
				//cout << cv::norm(H.col(2)) << endl;

				//printMat(H);

				//gramSchmidtOrthogonalizationMethod(H);

				//printMat(H);

				//cout << cv::norm(H.col(1)) << endl;

				H.col(1) /= cv::norm(H.col(1));

				//printMat(H);

				//printMat(tvec);

				/*cout << "Punto original: " << convertedCorners[0] << endl;
				//cout << "Punto 3DPlanar: " << Points3DPlanar[0] << endl;

				Mat point = Mat::zeros(3, 1, CV_64F);
				
				point.at<double>(0, 0) = Points3DPlanar[0].x;
				point.at<double>(1, 0) = Points3DPlanar[0].y;
				point.at<double>(2, 0) = 1;

				printMat(point);

				Mat temp = matrixK * H * point;

				cout << "Punto multiplicado por H: " << endl;

				printMat(temp);*/

				//circle(frame, cv::Point(temp.at<double>(0,0), temp.at<double>(1, 0)), 10, cv::Scalar(0, 0, 255), -1);

				Mat c1 = H.col(0);
				Mat c2 = H.col(1);

				c2 = c2 - c2.dot(c1) * c1;

				c2 /= cv::norm(c2);

				Mat c3 = c1.cross(c2);

				//cout << cv::norm(c1) << endl;
				//cout << cv::norm(c2) << endl;
				//cout << cv::norm(c3) << endl;

				//printMat(c1);
				//printMat(c2);
				//printMat(c3);

				//cout << c2 << endl;

				//invertRotationAroundZ(c2);

				//cout << c2 << endl;

				Mat R(3, 3, CV_64F);
				hconcat(R, c1, R);
				hconcat(R, c2, R);
				hconcat(R, c3, R);

				R = R.colRange(3, R.cols);

				/*Mat rotationOnX180 = (Mat_<double>(3, 3) <<
					1, 0, 0,
					0, -1, 0,
					0, 0, -1);

				Mat rotationOnZ180 = (Mat_<double>(3, 3) <<
					-1, 0, 0,
					0, -1, 0,
					0, 0, 1);

				R = rotationOnX180 * R;
				R = rotationOnZ180 * R;*/

				//printMat(R);

				//cout << "Dot xy: " << c1.dot(c2) << endl;
				//cout << "Dot xz: " << c1.dot(c3) << endl;
				//cout << "Dot yz: " << c2.dot(c3) << endl;
		
				vector<double> rvec;
				Rodrigues(R, rvec);

				//printMat(rvec);
				//printMat(tvec);

				//cout << rotationVector[0] << endl << rotationVector[1] << endl << rotationVector[2] << endl;
				//cout << traslationVector[0] << endl << traslationVector[1] << endl << traslationVector[2] << endl;
				


				//drawFrameAxes(frame, matrixK, distCoeffs, rvec, tvec, 2 * squareSize);

				//drawChessboardCorners(frame, cv::Size(8, 6), corners, true);

				projectPoints(axis, rvec, tvec, matrixK, distCoeffs, imgPoints);
				frame = draw(frame, corners, imgPoints);

				//projectPoints(axisCube, rotationVector, traslationVector, matrixK, distCoeffs, imgPoints);
				//frame = drawCube(frame, corners, imgPoints);
				
			}
		}

		Mat TCamera(traslationVector);						// Convert the traslation vector to a Mat matrix
		Mat RCamera;
		Rodrigues(rotationVector, RCamera);					// Convert the rotation vector to a Mat matrix sense solvePnP() outputs it in Rodrigues notation

		Mat RChessBoard = RCamera.t();						// The traspose of the camera rotation is equal to the ChessBoard rotation
		Mat TChessBoard = (-1) * (RChessBoard * TCamera);	// The traspose of the camera rotation times the traslation of the camera times -1 is equal to the ChessBoard traslation

		Mat temp;
		hconcat(RChessBoard, TChessBoard, temp);

		Mat bottomRow = cv::Mat::zeros(1, 4, CV_32F);
		bottomRow.at<float>(0, 3) = 1;

		Mat g;
		vconcat(temp, bottomRow, g);

		if (gTo.empty())
		{
			gTo = g.clone();
			RChessBoardTo = RChessBoard.clone();
			TChessBoardTo = TChessBoard.clone();
		}
		else
		{
			gFrom = gTo.clone();
			RChessBoardFrom = RChessBoardTo.clone();
			TChessBoardFrom = TChessBoardTo.clone();

			gTo = g.clone();
			RChessBoardTo = RChessBoard.clone();
			TChessBoardTo = TChessBoard.clone();

			Mat RDerivative = (RChessBoardTo - RChessBoardFrom) / deltaT;

			Mat TDerivative = (TChessBoardTo - TChessBoardFrom) / deltaT;

			Mat wHat = RDerivative * RChessBoardTo.t();

			Mat v = TDerivative - (wHat * TChessBoardTo);

			Mat temp2;
			hconcat(wHat, v, temp2);

			bottomRow = cv::Mat::zeros(1, 4, CV_32F);

			Mat chiHat;
			vconcat(temp2, bottomRow, chiHat);

			Mat gFromInverted;
			invert(gFrom, gFromInverted);

			Mat gFromTo = gTo * gFromInverted;
			
			Mat predictNextGFromTo = ((Mat::eye(4, 4, CV_32F)) + chiHat) * gFromTo * deltaT;

			Mat predictNextG = predictNextGFromTo * gTo;

			Mat point3DHomogeneos;
			convertVecPointToMat(point3D, point3DHomogeneos);
			
			Mat predictNextGConverteDouble;
			predictNextG.convertTo(predictNextGConverteDouble, CV_64F);

			Mat iO;
			hconcat(Mat::eye(3, 3, CV_64F), Mat::zeros(3, 1, CV_64F), iO);

			Mat predictedPoints = matrixK * iO * predictNextGConverteDouble * point3DHomogeneos;

			Mat predictedPointsAbs = abs(predictedPoints);

			Mat row1 = predictedPointsAbs.rowRange(0, 1);

			double maxValueX;
			double minValueX;
			
			minMaxLoc(row1, &minValueX, &maxValueX, NULL, NULL);

			Mat row2 = predictedPointsAbs.rowRange(1, 2);

			double maxValueY;
			double minValueY;

			minMaxLoc(row2, &minValueY, &maxValueY, NULL, NULL);

			rectangle(frame, Point(minValueX, minValueY), Point(maxValueX, maxValueY), Scalar(0, 255, 0), 2);


			//cout << "Filas pred: " << predictedPoints.rows << endl;
			//cout << "Cols pred: " << predictedPoints.cols << endl;

			//printMat(predictedPoints);

			//Mat predictedPointsMeters = iK * predictedPoints;

			//vector<Point3f> predictedPointsVector;

			//convertMatToVecPoint(predictedPointsMeters, predictedPointsVector);

			//printXYPoints(predictedPointsVector);

			imshow("Chessboard", frame);
		}

		int key = waitKey(10);
		if (key == 27) {
			break;
		}
	}

	camera.release();
	cv::destroyWindow("Chessboard");

	return 0;
}


