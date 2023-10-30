#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <corecrt_math_defines.h>
#include <stdlib.h>

using namespace std;
using namespace cv;

// This function evaluates and returns the derivatives(first and second) of a Gaussian function centered at the origin.
double derGaussian(double x, double sigma, int n)
{
    if (sigma <= 0)
        return nan("");

    double val = 0;

    if (n == 1)
        val = -(x / (sqrt(2 * M_PI) * pow(sigma, 3))) * exp(-0.5 * pow(x, 2) / pow(sigma, 2));
    else
        if (n == 2)
            val = ((pow(x, 2) / pow(sigma, 5)) - (1 / pow(sigma, 3))) * (exp(-0.5 * pow(x, 2) / (2 * pow(sigma, 2))) / sqrt(2 * M_PI));

    return val;
}

// Build a kernel(filter) from the first or second derivative of a Gaussian function.
Mat derGKernel(double sigma, int o)
{
    int n = ceil(6 * sigma);
    if (n % 2 == 0)
        n += 1;
    int c = (n + 1) / 2;
    Mat k(1, n, CV_64F, Scalar(0));
    if (o == 1)
    {
        k.at<double>(0, c - 1) = 0;
        int j = c - 2;
        double x = 0.5;
        for (int i = c; i < n; i++)
        {
            k.at<double>(0, i) = 0.5 * (derGaussian(x, sigma, 1) + derGaussian(x + 1, sigma, 1));
            k.at<double>(0, j) = -k.at<double>(0, i);
            j--;
            x++;
        }
    }
    else
    {
        if (o == 2)
        {
            k.at<double>(0, c - 1) = (derGaussian(0, sigma, 2) + derGaussian(0.5, sigma, 2)) / 2;
            int j = c - 2;
            double x = 0.5;
            for (int i = c; i < n; i++)
            {
                k.at<double>(0, i) = 0.5 * (derGaussian(x, sigma, 2) + derGaussian(x + 1, sigma, 2));
                k.at<double>(0, j) = k.at<double>(0, i);
                j--;
                x++;
            }
        }
    }
    return k;
}

// Evaluate x on gaussian function with a given miu and sigma.
double gaussian(double x, double miu, double sigma)
{
    if (sigma == 0)
        return nan("");

    double fact = 1. / (sqrt(2 * M_PI) * sigma);
    double val = fact * exp(-0.5 * pow((x - miu), 2) / pow(sigma, 2));
    return val;
}

// Computes a 2D Gaussian from two standard deviation values defined along ortogonal directions(sx, sy) and a third value that can be
// interpreted as a correlation factor rho, or the Gaussian main axis angle.
cv::Mat gaussian2d(double sx, double sy, double third, std::string kind = "covfact")
{
    if (kind != "covfact" && kind != "angle") {
        std::cout << "Error: third parameter should either be 'covfact' or 'angle'." << std::endl;
        return cv::Mat();
    }

    cv::Mat sigma;
    if (kind == "covfact") {
        double rho = third;
        if (rho > 1 || rho < -1) {
            std::cout << "Error: the third parameter (rho), should be defined in the interval (-1,1)" << std::endl;
            return cv::Mat();
        }
        sigma = (cv::Mat_<double>(2, 2) << sx * sx, rho * sx * sy, rho * sx * sy, sy * sy);
    }
    else {
        double theta = third;
        cv::Mat R = (cv::Mat_<double>(2, 2) << std::cos(theta), -std::sin(theta), std::sin(theta), std::cos(theta));
        sigma = R * (cv::Mat_<double>(2, 2) << sx * sx, 0, 0, sy * sy) * R.t();
    }

    cv::Mat S = sigma.inv();

    // Compute the Gaussian projection on the x and y axes.
    cv::Mat eigenValues, eigenVectors;
    cv::eigen(sigma, eigenValues, eigenVectors);
    cv::Mat pS = (cv::Mat_<double>(2, 2) << eigenVectors.at<double>(0, 0) * std::sqrt(eigenValues.at<double>(0, 0)),
        eigenVectors.at<double>(1, 0) * std::sqrt(eigenValues.at<double>(0, 0)),
        eigenVectors.at<double>(0, 1) * std::sqrt(eigenValues.at<double>(1, 0)),
        eigenVectors.at<double>(1, 1) * std::sqrt(eigenValues.at<double>(1, 0)));

    // Compute the horizontal filter size.
    int nx = std::ceil(6 * std::max(std::abs(pS.at<double>(0, 0)), std::abs(pS.at<double>(0, 1))));
    if (nx % 2 == 0) {
        nx += 1;
    }

    // Compute the vertical filter size.
    int ny = std::ceil(6 * std::max(std::abs(pS.at<double>(1, 0)), std::abs(pS.at<double>(1, 1))));
    if (ny % 2 == 0) {
        ny += 1;
    }

    cv::Mat g(ny, nx, CV_64F);
    int cx = (nx - 1) / 2;
    int cy = (ny - 1) / 2;

    double fact = 1 / (2 * M_PI * std::sqrt(cv::determinant(sigma)));
    double y = -cy;
    for (int i = 0; i < ny; i++)
    {
        int x = -cx;
        for (int j = 0; j < nx; j++)
        {
            cv::Mat p = (cv::Mat_<double>(2, 1) << x, y);
            double exponente = -0.5 * p.dot(S * p);
            g.at<double>(i, j) = fact * exp(exponente);

            x += 1;
        }
        y += 1;
    }

    // Normalize the filter coefficients so that their sum is equal to 1.
    double sum = cv::sum(g)[0];
    g /= sum;

    return g;
}

// Calculate a kernel based on a Gaussian function.
cv::Mat gKernel(double sigma)
{
    int n = ceil(6 * sigma);
    if (n % 2 == 0)
        n += 1;
    int c = (n + 1) / 2;
    cv::Mat k = cv::Mat::zeros(1, n, CV_64FC1);
    k.at<double>(c - 1) = (gaussian(0, 0, sigma) + gaussian(0.5, 0, sigma)) / 2;
    int j = c - 2;
    double x = 0.5;
    for (int i = c; i < n; i++)
    {
        k.at<double>(i) = 0.5 * (gaussian(x, 0, sigma) + gaussian(x + 1, 0, sigma));
        k.at<double>(j) = k.at<double>(i);
        j -= 1;
        x += 1;
    }
    return k;
}