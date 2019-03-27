#include "polygon_demo.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>


using namespace std;
using namespace cv;

PolygonDemo::PolygonDemo()
{
    m_data_ready = false;
}

PolygonDemo::~PolygonDemo()
{
}

void PolygonDemo::refreshWindow()
{
    Mat frame = Mat::zeros(480, 640, CV_8UC3);
    if (!m_data_ready)
        putText(frame, "Input data points (double click: finish)", Point(10, 470), FONT_HERSHEY_SIMPLEX, .5, Scalar(0, 148, 0), 1);

    drawPolygon(frame, m_data_pts, m_data_ready);
    if (m_data_ready)
    {
        // polygon area
        if (m_param.compute_area)
        {
            int area = polyArea(m_data_pts);
            char str[100];
            sprintf_s(str, 100, "Area = %d", area);
            putText(frame, str, Point(25, 25), FONT_HERSHEY_SIMPLEX, .8, Scalar(0, 255, 255), 1);
        }

		// x,y point print
		for (int i = 0; i < m_data_pts.size();i++)
		putText(frame, format("(%d, %d)", m_data_pts[i].x, m_data_pts[i].y), Point(m_data_pts[i].x+10, m_data_pts[i].y), FONT_HERSHEY_SIMPLEX, .5, Scalar(0, 255, 255), 1);


        // pt in polygon
        if (m_param.check_ptInPoly)
        {
            for (int i = 0; i < (int)m_test_pts.size(); i++)
            {
                if (ptInPolygon(m_data_pts, m_test_pts[i]))
                {
                    circle(frame, m_test_pts[i], 2, Scalar(0, 255, 0), CV_FILLED);
                }
                else
                {
                    circle(frame, m_test_pts[i], 2, Scalar(128, 128, 128), CV_FILLED);
                }
            }
        }

        // homography check
		//if (m_param.check_homography && m_data_pts.size() == 4)
		if (0)
        {
            // rect points
            int rect_sz = 100;
            vector<Point> rc_pts;
            rc_pts.push_back(Point(0, 0));
            rc_pts.push_back(Point(0, rect_sz));
            rc_pts.push_back(Point(rect_sz, rect_sz));
            rc_pts.push_back(Point(rect_sz, 0));
            rectangle(frame, Rect(0, 0, rect_sz, rect_sz), Scalar(255, 255, 255), 1);

            // draw mapping
            char* abcd[4] = { "A", "B", "C", "D" };
            for (int i = 0; i < 4; i++)
            {
                line(frame, rc_pts[i], m_data_pts[i], Scalar(255, 0, 0), 1);
                circle(frame, rc_pts[i], 2, Scalar(0, 255, 0), CV_FILLED);
                circle(frame, m_data_pts[i], 2, Scalar(0, 255, 0), CV_FILLED);
                putText(frame, abcd[i], m_data_pts[i], FONT_HERSHEY_SIMPLEX, .5, Scalar(0, 255, 255), 1);
            }
			
            // check homography
            int homo_type = classifyHomography(frame, rc_pts, m_data_pts);
            char type_str[100];
            switch (homo_type)
            {
            case NORMAL:
                sprintf_s(type_str, 100, "normal");
                break;
            case CONCAVE:
                sprintf_s(type_str, 100, "concave");
                break;
            case TWIST:
                sprintf_s(type_str, 100, "twist");
                break;
            case REFLECTION:
                sprintf_s(type_str, 100, "reflection");
                break;
            case CONCAVE_REFLECTION:
                sprintf_s(type_str, 100, "concave reflection");
               break;
			case NONE:
				sprintf_s(type_str, 100, "ERROR");
				break;

            }

            putText(frame, type_str, Point(15, 125), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 1);
        }
		///////////////fit circle or ellipse (0,1)
		int fit_circle = 0;
		int fit_ellipse = 1;
		if (fit_circle)
		{
			Point2d center;
			double radius = 0;
			fitCircle(m_data_pts, center, radius);

			{
				circle(frame, center, (int)(radius + 0.5), Scalar(0, 255, 0), 1);
			}
		}
		if (fit_ellipse)
		{
			Point m;
			Point v;
			double radius = 0;
			float theta=0;
			fitEllipse(m_data_pts, m, v, theta);
			//printf("%f", theta);

			{
	                    //buffer, 중심, 크기, 회전각도, 시작각, 끝각, 색깔
				ellipse(frame, m, v, theta, 0, 360, Scalar(255, 0, 0), 3, 8);

			}
		}
	}

    imshow("PolygonDemo", frame);
}

// return the area of polygon 넓이 구하기
int PolygonDemo::polyArea(const std::vector<cv::Point>& vtx)
{
	int area = 0;
	int abs_area;
	for (int i = 1; i < m_data_pts.size()-1; i++)
	{
		area = (m_data_pts[i].x- m_data_pts[0].x)*(m_data_pts[i + 1].y- m_data_pts[0].y)/2 -
			(m_data_pts[i + 1].x-m_data_pts[0].x)* (m_data_pts[i].y-m_data_pts[0].y)/2 + area;
	}
	
	abs_area = abs(area);

    return abs_area;
}

// return true if pt is interior point
bool PolygonDemo::ptInPolygon(const std::vector<cv::Point>& vtx, Point pt)
{
    return false;
}

// return homography type: NORMAL, CONCAVE, TWIST, REFLECTION, CONCAVE_REFLECTION (여기에 코딩)
int PolygonDemo::classifyHomography(cv::Mat& frame, const std::vector<cv::Point>& pts1, const std::vector<cv::Point>& pts2)
{
	int CrossProduct[100] = { 0 };
	char str[100];
    if (pts1.size() != 4 || pts2.size() != 4) return -1;

	//////////// 외적 계산/////////////////
	for (int i = 0; i < m_data_pts.size(); i++)
	{

		int i_pre = (i + m_data_pts.size()-1) % m_data_pts.size();
		int i_next = (i + 1) % m_data_pts.size();

		CrossProduct[i] = (m_data_pts[i].x - m_data_pts[i_pre].x)*(m_data_pts[i_next].y - m_data_pts[i].y)
			- (m_data_pts[i_next].x - m_data_pts[i].x)*(m_data_pts[i].y - m_data_pts[i_pre].y);

		sprintf_s(str, 100, "CrossProduct = %d", CrossProduct[i]);
		putText(frame, str, Point(m_data_pts[i].x, m_data_pts[i].y+10), FONT_HERSHEY_SIMPLEX, .3, Scalar(0, 255, 255), 1);
	}
	///////////외적을 이용해 변환 종류 판단/////////
	int positive = 0;
	int negative = 0;
	for (int i = 0; i < m_data_pts.size(); i++)
	{
		if (CrossProduct[i] > 0)
			positive++;
		else
			negative++;
	}
	if (positive == m_data_pts.size())
		return REFLECTION;
	else if (negative == m_data_pts.size())
		return NORMAL;
	else if (negative == 1)
		return CONCAVE;
	else if (positive == 1)
		return CONCAVE_REFLECTION;
	else if (positive == negative)
		return TWIST;
	else
		return NONE;
}

// estimate a circle that best approximates the input points and return center and radius of the estimate circle
bool PolygonDemo::fitCircle(const std::vector<cv::Point>& pts, cv::Point2d& center, double& radius)
{
    int n = (int)pts.size();
    if (n < 3) return false;
	
	//Mat J = Mat::zeros(n, 3, CV_64F);
	Mat J = Mat::zeros(n, 3, CV_32FC1);
	Mat X = Mat::zeros(3, 1, CV_32FC1);
	Mat Y = Mat::zeros(n, 1, CV_32FC1);
	Mat Jpinv;

	for (int i = 0; i < pts.size();  i++)
	{
		J.at<float>(i, 0) =  -2*pts[i].x;
		J.at<float>(i, 1) = -2*pts[i].y;
		J.at<float>(i, 2) = 1;
		Y.at<float>(i, 0) = -(pts[i].x*pts[i].x + pts[i].y*pts[i].y);
	}
	Mat d_svd, u, svd_t, svd;
	SVD::compute(J, d_svd, u, svd_t, SVD::FULL_UV);
	//printf("Singular Value\n");
	//cout << d_svd << endl;


	invert(J, Jpinv, DECOMP_SVD);
	X = Jpinv*Y;
	center.x = X.at<float>(0, 0);
	center.y = X.at<float>(1, 0);
	radius = sqrt(X.at<float>(0, 0)*X.at<float>(0, 0) + X.at<float>(1, 0)*X.at<float>(1, 0) - X.at<float>(2, 0));
	
	//printf("J mat\n");
	//cout << J << endl;

	//printf("Y mat\n");
	//cout << Y << endl;

	//printf("Jpinv mat\n");
	//cout << Jpinv << endl;

	//printf("X mat\n");
	//cout << X << endl;
	//Mat test = Mat::zeros(n, 1, CV_32FC1);
	//test = J*X;
	//printf("J*X\n");
	//cout << test << endl;
	//printf("Y mat\n");
	//cout << Y << endl;
    return 1;
}

bool PolygonDemo::fitEllipse(const std::vector<cv::Point>& pts, cv::Point& m, cv::Point& v, float& theta)
{
	////*********** ref : https://roadcom.tistory.com/30 ***********************
	int n = (int)pts.size();
	if (n < 5) return false;

	Mat J = Mat::zeros(n, 6, CV_32FC1);
	Mat X = Mat::zeros(6, 1, CV_32FC1);
	Mat Y = Mat::zeros(n, 1, CV_32FC1);
	for (int i = 0; i < pts.size(); i++)  	//matrix setting for Least Square Method 

	{
		J.at<float>(i, 0) = pts[i].x*pts[i].x;
		J.at<float>(i, 1) = pts[i].x*pts[i].y;
		J.at<float>(i, 2) = pts[i].y*pts[i].y;
		J.at<float>(i, 3) = pts[i].x;
		J.at<float>(i, 4) = pts[i].y;
		J.at<float>(i, 5) = 1;

	}
	Mat d_svd, u, svd_t, svd;
	
	SVD::compute(J, d_svd, u, svd_t, SVD::FULL_UV);  ///SVD
	//printf("Singular Value\n");
	//cout << d_svd << endl;

	transpose(svd_t, svd);

	for (int i = 0; i < 6; i++)
	{
		X.at<float>(i, 0) = svd.at<float>(i, 5);
	}
	printf("X mat");
	cout << X << endl;
	float a = X.at<float>(0, 0); 	float b = X.at<float>(1, 0); 	float c = X.at<float>(2, 0);
	float d = X.at<float>(3, 0);    float e = X.at<float>(4, 0);	float f = X.at<float>(5, 0);
	theta = 0.5*atan(b/(a-c));
		float cx = (2 * c*d - b*e) / (b*b - 4 * a*c);
	float cy = (2 * a*e - b*d) / (b*b - 4 * a*c);
	float cu = a*cx*cx + b*cx*cy + c*cy*cy - f;
	float w = sqrt(cu / (a*cos(theta)*cos(theta) + b* cos(theta)*sin(theta) + c*sin(theta)*sin(theta)));
	float h = sqrt(cu / (a*sin(theta)*sin(theta) - b* cos(theta)*sin(theta) + c*cos(theta)*cos(theta)));
	theta *= 180 / 3.14; //** 180/pi
	m.x = cx;
	m.y = cy;
	v.x = w;
	v.y = h;

	Mat out = J*X;
	//printf("mat J*mat X\n");
	//cout << out << endl;
	return 1;
}




void PolygonDemo::drawPolygon(Mat& frame, const std::vector<cv::Point>& vtx, bool closed)
{
    int i = 0;

    for (i = 0; i < (int)m_data_pts.size(); i++)
    {
        circle(frame, m_data_pts[i], 2, Scalar(255, 255, 255), CV_FILLED);
    }
    for (i = 0; i < (int)m_data_pts.size() - 1; i++)
    {
        line(frame, m_data_pts[i], m_data_pts[i + 1], Scalar(255, 255, 255), 1);
    }
    if (closed)
    {
        line(frame, m_data_pts[i], m_data_pts[0], Scalar(255, 255, 255), 1);
    }
	//putText(frame, format("(%d, %d)",m_data_pts[i].x, m_data_pts[i].y), m_data_pts[i], FONT_HERSHEY_SIMPLEX, .5, Scalar(0, 255, 255), 1);
	//putText(frame, format("(%d, %d)",m_data_pts[i].x, m_data_pts[i].y), m_data_pts[i], FONT_HERSHEY_SIMPLEX, .5, Scalar(0, 255, 255), 1);

}

void PolygonDemo::handleMouseEvent(int evt, int x, int y, int flags)
{
    if (evt == CV_EVENT_LBUTTONDOWN)
    {
        if (!m_data_ready)
        {
            m_data_pts.push_back(Point(x, y));
        }
        else
        {
            m_test_pts.push_back(Point(x, y));
        }
        refreshWindow();
    }
    else if (evt == CV_EVENT_LBUTTONUP)
    {
    }
    else if (evt == CV_EVENT_LBUTTONDBLCLK)
    {
        m_data_ready = true;
        refreshWindow();
    }
    else if (evt == CV_EVENT_RBUTTONDBLCLK)
    {
    }
    else if (evt == CV_EVENT_MOUSEMOVE)
    {
    }
    else if (evt == CV_EVENT_RBUTTONDOWN)
    {
        m_data_pts.clear();
        m_test_pts.clear();
        m_data_ready = false;
        refreshWindow();
    }
    else if (evt == CV_EVENT_RBUTTONUP)
    {
    }
    else if (evt == CV_EVENT_MBUTTONDOWN)
    {
    }
    else if (evt == CV_EVENT_MBUTTONUP)
    {
    }

    if (flags&CV_EVENT_FLAG_LBUTTON)
    {
    }
    if (flags&CV_EVENT_FLAG_RBUTTON)
    {
    }
    if (flags&CV_EVENT_FLAG_MBUTTON)
    {
    }
    if (flags&CV_EVENT_FLAG_CTRLKEY)
    {
    }
    if (flags&CV_EVENT_FLAG_SHIFTKEY)
    {
    }
    if (flags&CV_EVENT_FLAG_ALTKEY)
    {
    }
}
