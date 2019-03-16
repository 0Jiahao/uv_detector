#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <visualization_msgs/Marker.h>
#include <sstream>
#include <math.h>
#include <vector>
#include <time.h>
#include <UV_detector.h>
#include <kalman_filter.h>
#include <Eigen/Dense>

using namespace cv; 
using namespace std;

class my_detector
{  
	public:  
		my_detector()  
		{  
			image_transport::ImageTransport it(nh);
			//Topic subscribed 
			depsub = it.subscribe("/camera/aligned_depth_to_color/image_raw" 1, &my_detector::run,this);
			marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 1);
		}  

    void run(const sensor_msgs::ImageConstPtr& msg)  
		{  
			// image conversion
			cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
			cv::Mat depth = cv_ptr->image;
			// detect
			this->uv_detector.readdata(depth);
			this->uv_detector.detect();
			this->uv_detector.track();
			this->uv_detector.display_depth();
			this->uv_detector.display_U_map();
			this->uv_detector.display_bird_view();
			cout << this->uv_detector.bounding_box_B.size() << endl;
			// rviz visualization
			for(int i = 0; i < this->uv_detector.bounding_box_B.size(); i++)
			{
				Point2f obs_center = Point2f(this->uv_detector.bounding_box_B[i].x + this->uv_detector.bounding_box_B[i].width,
											this->uv_detector.bounding_box_B[i].y + this->uv_detector.bounding_box_B[i].height);
				cout << obs_center << endl;
			}
		}

	private:  
		ros::NodeHandle nh;   		// define node
    	image_transport::Subscriber depsub;		// define subscriber for depth image
		UVdetector uv_detector;
		ros::Publisher marker_pub;
};

int main(int argc, char **argv)  
{  
	//Initiate ROS  
	ros::init(argc, argv, "my_realsense_recorder");  

	//Create an object of class SubscribeAndPublish that will take care of everything  
	my_detector SAPObject; 

	ros::spin();  
	return 0;  
} 
