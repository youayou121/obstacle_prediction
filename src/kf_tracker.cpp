
#include <iostream>
#include <iterator>
#include <opencv2/video/video.hpp>
#include <std_msgs/Float32MultiArray.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/LaserScan.h>
#include <laser_geometry/laser_geometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <string>

using namespace std;
using namespace cv;
class MyPoint : public pcl::PointXYZ
{
public:
  int id = -1;
  float width = 0;
  float height = 0;
  float dx = -1;
  float dy = -1;
  bool ocp = false;
  float distance = 100;
};
laser_geometry::LaserProjection projector;
nav_msgs::OccupancyGrid g_map;
ros::Publisher pub_cloud_tf;
ros::Publisher pub_markers;
ros::Publisher pos_pub;
vector<MyPoint> pre_cluster_centroids;
vector<cv::KalmanFilter> KF_v;
int pre_obj_num = 0;
float odom_x = 0;
float odom_y = 0;
float odom_theta = 0;
float odom_w = 0;

float odom_to_map_x = 0;
float odom_to_map_y = 0;
float odom_to_map_yaw = 0;

bool first_frame = true;
bool odom_flag = false;
bool costmap_flag = false;
int count_flag = 0;
float pre_time;
float dt;
std::string odom_topic = "odom";
std::string costmap_topic = "move_base/global_costmap/costmap";
std::string scan_topic = "scan";

bool sort_x(MyPoint p1, MyPoint p2)
{
  if (p1.x < p2.x)
  {
    return true;
  }
  else
  {
    return false;
  }
}

bool is_static(int i)
{
  for (int j = 0; j < 20; j++)
  {
    int i_left = i - j;
    int i_right = i + j;
    int i_up = i - j * g_map.info.width;
    int i_down = i + j * g_map.info.width;
    if (int(*(g_map.data.begin() + i_left)) > 0)
    {
      return true;
      if (int(*(g_map.data.begin() + i_right)) > 0)
      {
        return true;
        if (int(*(g_map.data.begin() + i_up)) > 0)
        {
          return true;
          if (int(*(g_map.data.begin() + i_down)) > 0)
          {
            return true;
          }
        }
      }
    }
  }

  return false;
}

void costmap_cb(const nav_msgs::OccupancyGridConstPtr &map)
{
  if (map->data.size() > 0)
  {
    g_map = *map;
    cout << "height: " << g_map.info.height << " width: " << g_map.info.width
         << " origin_x:" << g_map.info.origin.position.x << " origin_y:" << g_map.info.origin.position.y
         << " resolution: " << g_map.info.resolution
         << endl;
    costmap_flag = true;
  }
}

// void odom_cb(const geometry_msgs::PoseWithCovarianceStampedConstPtr odom)
void odom_cb(const nav_msgs::OdometryConstPtr odom)
{
  odom_x = odom->pose.pose.position.x;
  odom_y = odom->pose.pose.position.y;
  odom_theta = odom->pose.pose.orientation.z;
  odom_w = odom->pose.pose.orientation.w;
  odom_flag = true;
}

float calculate_distance(vector<MyPoint>::iterator mypcl1, vector<MyPoint>::iterator mypcl2)
{
  float dist = sqrt((mypcl1->x - mypcl2->x) * (mypcl1->x - mypcl2->x) + (mypcl1->y - mypcl2->y) * (mypcl1->y - mypcl2->y));
  return dist;
}

Mat calculate_dist_mat(vector<MyPoint> pre, vector<MyPoint> now)
{
  int cols = pre.size();
  int rows = now.size();
  Mat dist_mat = Mat(rows, cols, CV_32F);
  vector<MyPoint>::iterator pre_it = pre.begin();
  for (int i = 0; i < cols; i++)
  {

    vector<MyPoint>::iterator now_it = now.begin();
    for (int j = 0; j < rows; j++)
    {
      dist_mat.at<float>(j, i) = calculate_distance(pre_it, now_it);
      now_it++;
    }
    pre_it++;
  }
  return dist_mat;
}

void kf_tracker(const sensor_msgs::PointCloud2 input)
{
  // cout<<"kf_tracker"<<endl;
  if (count_flag > 100000)
  {
    count_flag = 0;
  }
  if (input.data.size() > 0 && count_flag % 1 == 0)
  {
    int obj_num = 0;
    float now_time = input.header.stamp.toSec();
    // 点云聚类
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(input, *input_cloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(input_cloud);
    vector<pcl::PointIndices> cluster_indices;

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.5);
    ec.setMinClusterSize(5);
    ec.setMaxClusterSize(2000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(input_cloud);
    ec.extract(cluster_indices);

    obj_num = cluster_indices.size();
    vector<MyPoint> now_cluster_centroids;

    // 保证卡尔曼滤波的数量大于当前点云数量
    if (KF_v.size() < obj_num)
    {
      for (int i = KF_v.size(); i < obj_num; i++)
      {
        KalmanFilter KF(6, 2, 0, CV_32F);
        setIdentity(KF.measurementMatrix);
        float sigmaP = 0.01;
        float sigmaQ = 0.1;
        setIdentity(KF.processNoiseCov, Scalar::all(sigmaP));
        setIdentity(KF.measurementNoiseCov, Scalar(sigmaQ));
        KF_v.push_back(KF);
      }
    }
    // cout << obj_num << " " << KF_v.size() << endl;
    // 计算当前帧每个团簇的中心点，如果是第一帧则初始化卡尔曼滤波，
    int id = 0;
    vector<KalmanFilter>::iterator KF_it = KF_v.begin();
    if (obj_num > 0)
    {
      for (auto ptid_it = cluster_indices.rbegin(); ptid_it != cluster_indices.rend(); ++ptid_it)
      {

        float x = 0.0;
        float y = 0.0;
        float max_x = -100;
        float max_y = -100;
        float min_x = 100;
        float min_y = 100;
        int numPts = 0;
        vector<int>::iterator pit;

        for (pit = ptid_it->indices.begin(); pit != ptid_it->indices.end(); pit++)
        {
          if (max_x < input_cloud->points[*pit].x)
          {
            max_x = input_cloud->points[*pit].x;
          }
          if (max_y < input_cloud->points[*pit].y)
          {
            max_y = input_cloud->points[*pit].y;
          }
          if (min_x > input_cloud->points[*pit].x)
          {
            min_x = input_cloud->points[*pit].x;
          }
          if (min_y > input_cloud->points[*pit].y)
          {
            min_y = input_cloud->points[*pit].y;
          }
          x += input_cloud->points[*pit].x;
          y += input_cloud->points[*pit].y;
          numPts++;
        }
        // cout << "max_x:" << max_x << " max_y:" << max_y << " min_x:" << min_x << " min_y:" << min_y << endl;
        MyPoint centroid;
        centroid.x = x / numPts;
        centroid.y = y / numPts;
        centroid.z = 0.0;
        centroid.width = max_x - min_x + 0.1;
        centroid.height = max_y - min_y + 0.1;
        centroid.distance = sqrt(pow((centroid.x - odom_x), 2) + pow((centroid.y - odom_y), 2));
        if (first_frame)
        {
          // 第一帧时初始化卡尔曼滤波器
          KF_it->statePost.at<float>(0) = centroid.x;
          KF_it->statePost.at<float>(1) = centroid.y;
          KF_it->statePost.at<float>(2) = 0;
          KF_it->statePost.at<float>(3) = 0;
          KF_it->statePost.at<float>(4) = 0;
          KF_it->statePost.at<float>(5) = 0;
          centroid.id = id;
          pre_cluster_centroids.push_back(centroid);
          id++;
          KF_it++;
        }
        else
        {
          now_cluster_centroids.push_back(centroid);
        }
      }

      if (first_frame)
      {
        pre_obj_num = obj_num;
        pre_time = input.header.stamp.toSec();
        sort(pre_cluster_centroids.begin(), pre_cluster_centroids.end(), sort_x);
      }
      else
      {
        sort(pre_cluster_centroids.begin(), pre_cluster_centroids.end(), sort_x);
        sort(now_cluster_centroids.begin(), now_cluster_centroids.end(), sort_x);
        // cout << now_cluster_centroids.begin()->x << endl;
        // 遍历所有卡尔曼滤波并设置dt
        for (vector<KalmanFilter>::iterator kf_it = KF_v.begin(); kf_it != KF_v.end(); kf_it++)
        {
          dt = now_time - pre_time;
          // float dt = 1.0f;
          // cout << "dt:" << dt << endl;
          kf_it->transitionMatrix = (Mat_<float>(6, 6) << 1, 0, dt, 0, 0.5 * pow(dt, 2), 0,
                                     0, 1, 0, dt, 0, 0.5 * pow(dt, 2),
                                     0, 0, 1, 0, dt, 0,
                                     0, 0, 0, 1, 0, dt,
                                     0, 0, 0, 0, 1, 0,
                                     0, 0, 0, 0, 0, 1);
        }

        // 计算距离矩阵
        Mat dist_mat = calculate_dist_mat(pre_cluster_centroids, now_cluster_centroids);

        // cout << "pre_obj_num:" << pre_obj_num << " obj_num: " << obj_num << " kf_size: " << KF_v.size() << endl;
        // cout << dist_mat.cols << "列" << dist_mat.rows << "行" << endl;
        // cout << dist_mat << endl;

        // 将当前帧与上一帧的点云聚类信息进行数据关联,遍历距离矩阵，按行找出每行的最小值并确定点云对应关系

        int min_num = obj_num;

        if (pre_obj_num < obj_num)
        {
          min_num = pre_obj_num;
        }
        for (int count = 0; count < min_num; count++)
        {
          int flag_cols = -1;
          int flag_rows = -1;
          float min_dist = 100;
          for (int i = 0; i < obj_num; i++)
          {
            for (int j = 0; j < pre_obj_num; j++)
            {
              if (min_dist > dist_mat.at<float>(i, j))
              {
                min_dist = dist_mat.at<float>(i, j);
                flag_cols = j;
                flag_rows = i;
              }
            }
          }
          if (flag_cols >= 0)
          {
            now_cluster_centroids.at(flag_rows).id = pre_cluster_centroids.at(flag_cols).id;
            now_cluster_centroids.at(flag_rows).dx = now_cluster_centroids.at(flag_rows).x - pre_cluster_centroids.at(flag_cols).x;
            now_cluster_centroids.at(flag_rows).dy = now_cluster_centroids.at(flag_rows).y - pre_cluster_centroids.at(flag_cols).y;
            pre_cluster_centroids.at(flag_cols).ocp = true;
            for (int i = 0; i < obj_num; i++)
            {
              dist_mat.at<float>(i, flag_cols) = 100;
            }
            for (int i = 0; i < pre_obj_num; i++)
            {
              dist_mat.at<float>(flag_rows, i) = 100;
            }
          }
        }

        // 遍历当前点云，如果有还没匹配的点云id，就遍历之前的点云，如果有之前的点云没被占有就将，该点云id设置为没被占有的点云id
        int void_id = 0;
        for (vector<MyPoint>::iterator it = now_cluster_centroids.begin(); it != now_cluster_centroids.end(); it++)
        {
          if (it->id < 0)
          {
            for (vector<MyPoint>::iterator mypcl_it = now_cluster_centroids.begin(); mypcl_it != now_cluster_centroids.end(); mypcl_it++)
            {
              for (vector<MyPoint>::iterator mypcl_pre_it = pre_cluster_centroids.begin(); mypcl_pre_it != pre_cluster_centroids.end(); mypcl_pre_it++)
              {
                if (mypcl_pre_it->ocp == false)
                {
                  mypcl_it->id = mypcl_pre_it->id;
                  mypcl_pre_it->ocp = true;
                }
                if (mypcl_pre_it->id == void_id)
                {
                  // cout << "void_id:" << void_id << endl;
                  void_id++;
                }
              }
            }
            if (it->id < 0)
            {
              it->id = void_id;
              it->ocp = true;
              // cout << "void_id::" << void_id << endl;
              void_id++;
            }
          }
        }

        // 遍历当前点云信息，如果是新添加的点云，则重新设置卡尔曼滤波，如果是有对应点云，则进行correct
        visualization_msgs::MarkerArray obs_markers;
        std_msgs::Float32MultiArray pos;
        int flag = 0;
        for (vector<MyPoint>::iterator mypcl_it = now_cluster_centroids.begin(); mypcl_it != now_cluster_centroids.end(); mypcl_it++)
        {
          if (mypcl_it->ocp)
          {
            KF_v.at(mypcl_it->id).init(6, 2, 0, CV_32F);
            KF_v.at(mypcl_it->id).transitionMatrix = (Mat_<float>(6, 6) << 1, 0, dt, 0, 0.5 * pow(dt, 2), 0,
                                                      0, 1, 0, dt, 0, 0.5 * pow(dt, 2),
                                                      0, 0, 1, 0, dt, 0,
                                                      0, 0, 0, 1, 0, dt,
                                                      0, 0, 0, 0, 1, 0,
                                                      0, 0, 0, 0, 0, 1);
            setIdentity(KF_v.at(mypcl_it->id).measurementMatrix);
            float sigmaP = 0.01;
            float sigmaQ = 0.1;
            setIdentity(KF_v.at(mypcl_it->id).processNoiseCov, Scalar::all(sigmaP));
            setIdentity(KF_v.at(mypcl_it->id).measurementNoiseCov, Scalar(sigmaQ));
            KF_v.at(mypcl_it->id).statePost.at<float>(0) = mypcl_it->x;
            KF_v.at(mypcl_it->id).statePost.at<float>(1) = mypcl_it->y;
            KF_v.at(mypcl_it->id).statePost.at<float>(2) = 0;
            KF_v.at(mypcl_it->id).statePost.at<float>(3) = 0;
            KF_v.at(mypcl_it->id).statePost.at<float>(4) = 0;
            KF_v.at(mypcl_it->id).statePost.at<float>(5) = 0;
          }
          else
          {
            Mat pred = KF_v.at(mypcl_it->id).predict();
            visualization_msgs::Marker marker;
            marker.id = flag;
            marker.type = visualization_msgs::Marker::CUBE;
            marker.header.frame_id = "map";
            marker.scale.x = mypcl_it->width;
            marker.scale.y = mypcl_it->height;
            marker.scale.z = 1.75;
            marker.action = visualization_msgs::Marker::ADD;
            marker.color.r = flag % 2 ? 1 : 0;
            marker.color.g = flag % 3 ? 1 : 0;
            marker.color.b = flag % 4 ? 1 : 0;
            marker.color.a = 0.5;
            marker.lifetime = ros::Duration(0.5);

            cout << flag << ": "
                 << " x:" << mypcl_it->x
                 << " y:" << mypcl_it->y
                 << " v_x:" << pred.at<float>(2)
                 << " v_y:" << pred.at<float>(3)
                 << " a_x:" << pred.at<float>(4)
                 << " a_y:" << pred.at<float>(5) << endl;

            pos.data.push_back(mypcl_it->x);
            pos.data.push_back(mypcl_it->y);
            for (int k = 2; k < 6; k++)
            {
              pos.data.push_back(pred.at<float>(k));
            }
            pos.data.push_back(mypcl_it->width);
            pos.data.push_back(mypcl_it->height);
            marker.pose.position.x = mypcl_it->x;
            marker.pose.position.y = mypcl_it->y;
            marker.pose.position.z = 0.375;
            obs_markers.markers.push_back(marker);
            // float meas[4] = {mypcl_it->x, mypcl_it->y, (mypcl_it->dx) / dt, (mypcl_it->dy) / dt};
            // float meas[4] = {mypcl_it->x, mypcl_it->y, mypcl_it->dx, mypcl_it->dy};
            float meas[2] = {mypcl_it->x, mypcl_it->y};
            Mat measMat = cv::Mat(2, 1, CV_32F, meas);
            // cout << dt << endl;
            // cout << measMat << endl;
            KF_v.at(mypcl_it->id).correct(measMat);
            flag++;
          }
        }
        // 遍历卡尔曼滤波器predict并且发布障碍物pos信息
        pos_pub.publish(pos);
        pub_markers.publish(obs_markers);
        // 输出前一帧点云信息
        for (vector<MyPoint>::iterator mypcl_pre_it = pre_cluster_centroids.begin(); mypcl_pre_it != pre_cluster_centroids.end(); mypcl_pre_it++)
        {
          // cout << "pre_cloud_id: " << mypcl_pre_it->id << ":" << mypcl_pre_it->ocp << " x:" << mypcl_pre_it->x << " y:" << mypcl_pre_it->y << " width:" << mypcl_pre_it->width << " height:" << mypcl_pre_it->height << endl;
        }
        // cout << endl;
        // 输出当前帧点云信息
        for (vector<MyPoint>::iterator mypcl_it = now_cluster_centroids.begin(); mypcl_it != now_cluster_centroids.end(); mypcl_it++)
        {

          // cout << "now_cloud_id: " << mypcl_it->id << ":" << mypcl_it->ocp << " x:" << mypcl_it->x << " y:" << mypcl_it->y << " width:" << mypcl_it->width << " height:" << mypcl_it->height << endl;
          mypcl_it->ocp = false;
        }
        // cout << endl;
        pre_cluster_centroids.swap(now_cluster_centroids);
      }
    }

    pre_time = now_time;
    first_frame = false;
    pre_obj_num = obj_num;
    // cout << endl;
  }
}

void scan_cb(const sensor_msgs::LaserScanConstPtr &scan_in)
{
  // cout << "scan_cb" << endl;
  if (scan_in->ranges.size() > 0 && odom_flag && costmap_flag && count_flag % 1 == 0)
  {
    sensor_msgs::PointCloud2 input;
    projector.projectLaser(*scan_in, input);
    pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(input, *pc);

    tf::Quaternion q = tf::Quaternion(0, 0, odom_theta, odom_w);
    double roll, pitch, yaw;
    Eigen::Affine3f tf3;
    tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
    int rotate_flag;
    ros::param::get("/rotate_flag", rotate_flag);
    // cout << rotate_flag << endl;
    if (rotate_flag == 1)
    {
      yaw = yaw + 3.1415926;
    }
    tf3 = pcl::getTransformation(odom_x, odom_y, 0, 0, 0, yaw);
    pcl::transformPointCloud(*pc, *pc, tf3); // 得到世界坐标系下的点云
    tf3 = pcl::getTransformation(odom_to_map_x, odom_to_map_y, 0, 0, 0, odom_to_map_yaw);
    pcl::transformPointCloud(*pc, *pc, tf3); // 得到世界坐标系下的点云
    for (auto it = pc->begin(); it != pc->end(); ++it)
    {
      float x = 0.0;
      float y = 0.0;
      float z = 0.0;
      int numPts = 0;

      float p_x, p_y;
      p_x = it->x;
      p_y = it->y;
      int mx, my;
      mx = (p_x - g_map.info.origin.position.x) / g_map.info.resolution;
      my = (p_y - g_map.info.origin.position.y) / g_map.info.resolution;
      int index = my * g_map.info.width + mx;
      if (!is_static(index))
      {
        x += p_x;
        y += p_y;
        clustered_cloud->points.push_back(*it);
        numPts++;
      }

      pcl::PointXYZ centroid;
      centroid.x = x / numPts;
      centroid.y = y / numPts;
    }
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*clustered_cloud, output);
    output.header.frame_id = "map";
    output.header.stamp = scan_in->header.stamp;
    pub_cloud_tf.publish(output);
    kf_tracker(output);
  }
  count_flag++;
  odom_flag = false;
}

void odom_to_map_cb(const std_msgs::Float32MultiArrayConstPtr pos)
{
  odom_to_map_x = pos->data.at(0);
  odom_to_map_y = pos->data.at(1);
  odom_to_map_yaw = pos->data.at(2);
  cout << odom_to_map_x << " " << odom_to_map_y << " " << odom_to_map_yaw << endl;
}

int main(int argc, char **argv)
{
  // ROS init
  ros::init(argc, argv, "kf_tracker");
  ros::NodeHandle nh;
  pos_pub = nh.advertise<std_msgs::Float32MultiArray>("pos", 1); // clusterCenter1
  pub_markers = nh.advertise<visualization_msgs::MarkerArray>("obs_markers", 1);
  pub_cloud_tf = nh.advertise<sensor_msgs::PointCloud2>("cloud2d_tf", 1);
  ros::Subscriber sub_odom = nh.subscribe(odom_topic, 1, odom_cb);
  ros::Subscriber sub_costmap = nh.subscribe(costmap_topic, 1, costmap_cb);
  ros::Subscriber sub_scan = nh.subscribe(scan_topic, 1, scan_cb);
  ros::Subscriber sub_pos = nh.subscribe("odom_to_map", 1, odom_to_map_cb);
  ros::spin();
}
