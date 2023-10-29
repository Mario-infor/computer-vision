//#include <opencv2/opencv.hpp>
//#include <opencv2/calib3d.hpp>
//#include <string>
//#include <vector>
//#include <iostream>
//
//using namespace std;
//using namespace cv;
//
//// Draw all corners detected by findChessboardCorners method.
//void drawChessBoardCorners() {
//	int IM_HEIGHT = 1080;
//	int	IM_WIDTH = 1920;
//
//	cv::VideoCapture camera = cv::VideoCapture("Video/Chessboard.mp4");
//
//	if (!camera.isOpened()) {
//		std::cout << "No se puede abrir el dispositivo de video" << std::endl;
//		return;
//	}
//
//	double frmWidth = camera.get(cv::CAP_PROP_FRAME_WIDTH);
//	double frmHeight = camera.get(cv::CAP_PROP_FRAME_HEIGHT);
//	double scaleX = IM_WIDTH / frmWidth;
//	double scaleY = IM_HEIGHT / frmHeight;
//	cv::namedWindow("Chessboard");
//
//	int cont = 0;
//	cv::Mat frame;
//	while (true) {
//		bool retval = camera.read(frame);
//		if (retval) {
//			if (scaleX != 1.0 || scaleY != 1.0) {
//				cv::resize(frame, frame, cv::Size(), scaleX, scaleY, cv::INTER_AREA);
//			}
//
//			cv::Mat imCorners;
//			bool retval = cv::findChessboardCorners(frame, cv::Size(6, 8), imCorners);
//			if (retval) {
//				cv::drawChessboardCorners(frame, cv::Size(6, 8), imCorners, true);
//				cv::imshow("Chessboard", frame);
//			}
//		}
//
//		int key = cv::waitKey(1);
//		if (key == 27) {
//			break;
//		}
//		cont++;
//	}
//
//	camera.release();
//	cv::destroyWindow("Chessboard");
//}
//
//// Draw 3D axis.
//Mat draw(Mat img, vector<Point2f> corners, vector<Point2f> imgpts) {
//
//	Point2f corner(corners[0].x, corners[0].y);
//
//	line(img, corner, imgpts[0], Scalar(255, 0, 0), 10);
//	line(img, corner, imgpts[1], Scalar(0, 255, 0), 10);
//	line(img, corner, imgpts[2], Scalar(0, 0, 255), 10);
//
//	return img;
//}
//
//
//Mat drawCube(Mat img, vector<Point2f> corners, vector<Point2f> imgpts) {
//	// Convert imgpts to a vector of type cv::Point with integers numbers
//	std::vector<cv::Point> intImgPts;
//	for (const auto& pt : imgpts) {
//		intImgPts.push_back(cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
//	}
//
//	// Draw ground floor in green
//	std::vector<std::vector<cv::Point>> groundContours = { std::vector<cv::Point>(intImgPts.begin(), intImgPts.begin() + 4) };
//	cv::drawContours(img, groundContours, -1, cv::Scalar(0, 255, 0), -3);
//
//	// Dibujar pilares en color azul
//	for (int i = 0; i < 4; ++i) {
//		cv::line(img, intImgPts[i], intImgPts[i + 4], cv::Scalar(255), 3);
//	}
//
//	// Dibujar top layer en color rojo
//	std::vector<std::vector<cv::Point>> topContours = { std::vector<cv::Point>(intImgPts.begin() + 4, intImgPts.end()) };
//	cv::drawContours(img, topContours, -1, cv::Scalar(0, 0, 255), 3);
//
//	return img;
//}
//
//
//int mainSolvePnP() {
//
//	// Dimensions of the chessboard and size of the little squares.
//	int rowsChessBoard = 6;
//	int colsChessBoard = 8;
//	float squareSize = 28.45;
//
//	// K matrix for de camera calibration
//	Mat matrixK = (Mat_<double>(3, 3) <<
//		7.6808743880079578e+02, 0, 4.2089026071365419e+02,
//		0, 7.6808743880079578e+02, 3.0855623821214846e+02,
//		0, 0, 1);
//
//	float lineLenght = 2 * squareSize;
//
//	// Matrix for the axis we will draw on the picture.
//	Mat axis = (Mat_<float>(3, 3) <<
//		lineLenght, 0, 0,
//		0, lineLenght, 0,
//		0, 0, -lineLenght);
//
//	Mat axisCube = (Mat_<float>(8, 3) <<
//		0, 0, 0,
//		0, lineLenght, 0,
//		lineLenght, lineLenght, 0,
//		lineLenght, 0, 0,
//		0, 0, -lineLenght,
//		0, lineLenght, -lineLenght,
//		lineLenght, lineLenght, -lineLenght,
//		lineLenght, 0, -lineLenght);
//
//	vector<Point3f> point3D;
//	float rowPos = squareSize;
//	float colPos = squareSize;
//
//	// For loop to fill up de 3D point of the chessboard.
//	for (size_t i = 0; i < rowsChessBoard; i++)
//	{
//		for (size_t j = 0; j < colsChessBoard; j++)
//		{
//			point3D.push_back(Point3f(j * rowPos, i * colPos, 0));
//		}
//	}
//
//	// Capture the video file and verify that it was load correctly.
//	VideoCapture camera = VideoCapture("C:/Mario/Maestría/Clases Maestría/2do Semestre/Vision Computacional/Tarea Seguir Tablero/TareasVisionComputacional/TareaVisionSeguirTablero/Chessboard.mp4");
//	if (!camera.isOpened()) {
//		cout << "No se puede abrir el dispositivo de video" << endl;
//		return 0;
//	}
//
//	vector<Point2f> corners;			// chessboard points vector, output of findChessboardCorners() method
//	vector<Point2f> imgPoints;
//	vector<float> rotationVector;		// rotation vector, output of solvePnP() method
//	vector<float> traslationVector;		// traslation vector, output of solvePnP() method
//	Mat distCoeffs;					// empty vector to ignore the distorsion coefficients of the camera
//
//	namedWindow("Chessboard");
//	Mat frame;
//	Mat gray;
//
//	Mat gFrom;
//	Mat RChessBoardFrom;
//	Mat TChessBoardFrom;
//
//	Mat gTo;
//	Mat RChessBoardTo;
//	Mat TChessBoardTo;
//
//	float deltaT = 1.0;
//
//	Mat verPuntos = Mat::zeros(1920, 1080, CV_32F);
//
//	while (true) {
//
//		bool retval = camera.read(frame);
//
//		Mat resizedImage;
//		resize(frame, resizedImage, cv::Size(854, 480));
//
//		cvtColor(resizedImage, gray, cv::COLOR_RGB2GRAY);
//
//		if (retval) {
//			// openCV method that finds the corners inside the ChessBoard
//			bool retval = findChessboardCorners(gray, cv::Size(8, 6), corners);
//
//			if (retval) {
//
//				cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.1));
//
//				//drawChessboardCorners(frame, cv::Size(8, 6), corners, true);
//
//				// openCV method that outputs the rotation and traslation vectors of the camera using the points from the ChessBoard
//				solvePnP(point3D, corners, matrixK, distCoeffs, rotationVector, traslationVector);
//
//				//drawFrameAxes(resizedImage, matrixK, distCoeffs, rotationVector, traslationVector, squareSize);
//
//				projectPoints(axisCube, rotationVector, traslationVector, matrixK, distCoeffs, imgPoints);
//				resizedImage = drawCube(resizedImage, corners, imgPoints);
//				imshow("Chessboard", resizedImage);
//			}
//		}
//
//		int key = waitKey(1);
//		if (key == 27) {
//			break;
//		}
//	}
//
//	camera.release();
//	cv::destroyWindow("Chessboard");
//
//	return 0;
//}
//
//
