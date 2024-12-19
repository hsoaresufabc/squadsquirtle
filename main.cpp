#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

// Password and constants
const string PASSWORD = "1234"; // Correct password
const char WILDCARD = '*';      // Password masking
const Rect HAND_REGION(200, 100, 200, 200); // Region of interest (ROI)

// Helper function to count fingers
int countFingers(const vector<Vec4i>& defects, const vector<Point>& contour) {
    int fingerCount = 0;

    for (const auto& defect : defects) {
        Point ptStart = contour[defect[0]];
        Point ptEnd = contour[defect[1]];
        Point ptFar = contour[defect[2]];

        // Use triangle geometry to count fingers
        double a = norm(ptStart - ptFar);
        double b = norm(ptEnd - ptFar);
        double c = norm(ptStart - ptEnd);
        double angle = acos((a * a + b * b - c * c) / (2 * a * b));

        if (angle < CV_PI / 2 && a > 50 && b > 50) {
            fingerCount++;
        }
    }
    return fingerCount;
}

// Function to remove background and isolate hand
Mat removeBackground(const Mat& roi) {
    Mat hsv, mask;

    // Convert ROI to HSV
    cvtColor(roi, hsv, COLOR_BGR2HSV);

    // Define skin color range and apply threshold
    inRange(hsv, Scalar(0, 30, 60), Scalar(20, 150, 255), mask);

    // Smooth the mask
    GaussianBlur(mask, mask, Size(5, 5), 0);

    // Perform morphological operations to clean up the mask
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    morphologyEx(mask, mask, MORPH_OPEN, kernel);

    return mask;
}

int main() {
    VideoCapture cap(0); // Open the default camera
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open the camera" << endl;
        return -1;
    }

    // Initialize variables
    Mat frame, roi, mask;
    string inputPassword = "";
    bool isReading = false;

    // Instructions for the user
    cout << "Press 'k' to input, 'c' to clear, 'q' to quit." << endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Draw the square region on the frame
        rectangle(frame, HAND_REGION, Scalar(255, 0, 0), 2);

        // Extract the region of interest (ROI)
        roi = frame(HAND_REGION);

        // Remove background
        mask = removeBackground(roi);

        // Find contours in the mask
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // Identify the largest contour
        int largestIndex = -1;
        double maxArea = 0;
        for (int i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i]);
            if (area > maxArea) {
                maxArea = area;
                largestIndex = i;
            }
        }

        if (largestIndex != -1) {
            vector<Point> hullPoints;
            vector<int> hullIndices;
            vector<Vec4i> defects;

            // Convex hull and convexity defects
            convexHull(contours[largestIndex], hullPoints);
            convexHull(contours[largestIndex], hullIndices);
            if (hullIndices.size() > 3) {
                convexityDefects(contours[largestIndex], hullIndices, defects);
            }

            // Draw contours and hull on the ROI
            drawContours(roi, contours, largestIndex, Scalar(0, 255, 0), 2);
            polylines(roi, hullPoints, true, Scalar(255, 0, 0), 2);

            // Count fingers
            int fingerCount = countFingers(defects, contours[largestIndex]);

            // If we are reading input, add the finger count to the password
            if (isReading) {
                inputPassword += to_string(fingerCount);
                isReading = false;
            }

            // Display the number of fingers on the frame
            putText(frame, "Fingers: " + to_string(fingerCount), Point(10, 50),
                    FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        }

        // Mask the password progress with '*'
        string maskedPassword(inputPassword.length(), WILDCARD);
        putText(frame, "Password: " + maskedPassword, Point(10, 100),
                FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

        // Display instructions
        putText(frame, "Press 'k' to input, 'c' to clear, 'q' to quit", Point(10, frame.rows - 20),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 1);

        // Validate password once it reaches the correct length
        if (inputPassword.length() == PASSWORD.length()) {
            if (inputPassword == PASSWORD) {
                putText(frame, "Access Granted!", Point(10, 150),
                        FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
                imshow("Gesture Detection - Squad Squirtle", frame);
                waitKey(3000); // Wait for 3 seconds
                break;
            } else {
                putText(frame, "Authentication Failed!", Point(10, 150),
                        FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
                inputPassword = ""; // Reset the password
                imshow("Gesture Detection - Squad Squirtle", frame);
                waitKey(3000); // Wait for 3 seconds
            }
        }

        // Display the windows
        imshow("Gesture Detection - Squad Squirtle", frame);
        imshow("Hand Region", mask);

        // Handle key events
        char key = waitKey(30);
        if (key == 'q') break;          // Quit the program
        if (key == 'k') isReading = true; // Start reading input
        if (key == 'c') {
            inputPassword = ""; // Clear the password
            cout << "Password cleared." << endl;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}

