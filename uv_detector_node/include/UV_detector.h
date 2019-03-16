#ifndef UV_DETECTOR_H
#define UV_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <math.h>
#include <vector>
#include <kalman_filter.h>

using namespace std;
using namespace cv;

class UVbox
{
    public:
    // members
    int id; // its id
    int toppest_parent_id; // its toppest parent's id
    Rect bb; // bounding box

    // default constructor
    UVbox();
    // constructor for new line
    UVbox(int seg_id, int row, int left, int right);
};

class UVtracker
{
    public:
    // members
    vector<Rect> pre_bb; // bounding box information
    vector<Rect> now_bb; 
    vector<vector<Point2f> > pre_history; // thehistory of previous detection
    vector<vector<Point2f> > now_history; 
    vector<kalman_filter> pre_filter; // states includes x, y, vx, vy, width, depth
    vector<kalman_filter> now_filter;
    float overlap_threshold; // threshold to determind tracked or not

    // constructor
    UVtracker();

    // read new bounding box information
    void read_bb(vector<Rect> now_bb);

    // check tracking status
    void check_status();
};

class UVdetector
{
    public:
    // members
    Mat depth; // depth map
    Mat depth_low_res; // depth map with low resolution
    Mat U_map; // U map
    int min_dist; // lower bound of range of interest
    int max_dist; // upper bound of range of interest
    int row_downsample; // ratio (depth map's height / U map's height)
    float col_scale; // scale factor in horizontal direction
    float threshold_point; // threshold of point of interest
    float threshold_line; // threshold of line of interest
    int min_length_line; // min value of line's length
    bool show_bounding_box_U; // show bounding box or not
    vector<Rect> bounding_box_U; // extracted bounding boxes on U map
    vector<Rect> bounding_box_B; // bunding boxes on the bird's view map
    float fx; // focal length
    float fy;
    float px; // principle point
    float py;
    Mat bird_view; // bird's view map 
    UVtracker tracker; // tracker in bird's view map

    // constructor
    UVdetector();

    // read data
    void readdata(Mat depth);

    // extract U map
    void extract_U_map();

    // extract bounding box
    void extract_bb();

    // extract bird's view map
    void extract_bird_view();

    // detect
    void detect();

    // track the object
    void track();

    // output detection 
    void output();

    // display depth
    void display_depth();

    // display U map
    void display_U_map();

    // add tracking result to bird's view map
    void add_tracking_result();

    // display bird's view map
    void display_bird_view();
};

UVbox merge_two_UVbox(UVbox father, UVbox son);

#endif