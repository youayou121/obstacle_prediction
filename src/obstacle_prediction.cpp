#include <ros/ros.h>
#include <opencv2/video/video.hpp>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseArray.h>
#include <obstacle_prediction/Obstacle.h>
#include <obstacle_prediction/ObstacleArray.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <laser_geometry/laser_geometry.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <vector>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/Float32MultiArray.h>
class KFO{
  public:
  bool ocp = false;
  cv::KalmanFilter kf;
};
ros::Publisher pub_cloud_filtered;
ros::Publisher pub_pos;
ros::Publisher pub_markers;
ros::Publisher pub_obstacles;
ros::Publisher pub_local_costmap;
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_local(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_global(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
laser_geometry::LaserProjection projector;
nav_msgs::OccupancyGrid global_costmap;
visualization_msgs::MarkerArray obst_markers;
obstacle_prediction::ObstacleArray pre_obst_arr;
obstacle_prediction::ObstacleArray curr_obst_arr;
int pre_obst_num = 0;
int obst_num = 0;
bool first_frame = true;
bool sub_map = false;
float pre_time = 0.0;
float dt = 0.0;
float curr_time = 0.0;
float dist_threshold = 0.3;
std::vector<KFO> kf_v;
void visulization_method()
{
  obst_markers.markers.resize(0);
  for (auto obst:curr_obst_arr.obstacles)
  {
    visualization_msgs::Marker marker_cube;
    visualization_msgs::Marker marker_arrow;
    float obst_yaw = atan2(obst.linear.y, obst.linear.x);
    float speed = sqrt((obst.linear.x*obst.linear.x)+(obst.linear.y*obst.linear.y));
    tf::Quaternion quaternion;
    quaternion.setRPY(0, 0, obst_yaw);
    marker_arrow.pose.orientation.x = quaternion.getX();
    marker_arrow.pose.orientation.y = quaternion.getY();
    marker_arrow.pose.orientation.z = quaternion.getZ();
    marker_arrow.pose.orientation.w = quaternion.getW();
    marker_cube.type = visualization_msgs::Marker::CUBE;
    marker_arrow.type = visualization_msgs::Marker::ARROW;
    marker_cube.id = obst.id * 2 - 1;
    marker_arrow.id = obst.id * 2;
    marker_arrow.header.frame_id = marker_cube.header.frame_id = "map";
    marker_arrow.header.stamp = marker_cube.header.stamp = ros::Time::now();
    marker_cube.scale.x = obst.length;
    marker_cube.scale.y = obst.width;
    marker_cube.scale.z = 1;
    marker_arrow.scale.x = 1*speed;
    marker_arrow.scale.y = 0.1;
    marker_arrow.scale.z = 0.1;
    marker_arrow.pose.position = marker_cube.pose.position = obst.position;
    marker_arrow.action = marker_cube.action = visualization_msgs::Marker::ADD;
    marker_cube.color.a = 0.5;
    marker_arrow.color.a = 1.0;
    marker_arrow.lifetime = marker_cube.lifetime = ros::Duration(0.5);
    obst_markers.markers.push_back(marker_cube);
    obst_markers.markers.push_back(marker_arrow);
  }
  pub_markers.publish(obst_markers);
}

void pub_local_costmap_method()
{
  nav_msgs::OccupancyGrid local_costmap;
  if (sub_map)
  {
    local_costmap.header.frame_id = "map";
    local_costmap.header.stamp.ros::Time::now();
    local_costmap.info = global_costmap.info;
    local_costmap.data.resize(global_costmap.data.size());
    for (auto obst:curr_obst_arr.obstacles)
    {
      float x = obst.position.x;
      float y = obst.position.y;
      int mx = (x - global_costmap.info.origin.position.x) / global_costmap.info.resolution;
      int my = (y - global_costmap.info.origin.position.y) / global_costmap.info.resolution;
      
      int size_x = obst.length/global_costmap.info.resolution/2;
      int size_y = obst.width/global_costmap.info.resolution/2;
      for(int mx_i = mx-size_x;mx_i<=mx+size_x;mx_i++)
      {
        for(int my_i = my-size_y;my_i <= my+size_y;my_i++)
        {
          int index = my_i * global_costmap.info.width + mx_i;
          if(index>=0&&index<local_costmap.data.size())
          {
            local_costmap.data[index] = 100;
          }
        }
      }
      
    }
  }
  else
  {
    local_costmap.header.frame_id = "map";
    local_costmap.header.stamp.ros::Time::now();
  } 
  
  
  pub_local_costmap.publish(local_costmap);
}

bool is_static(int i)
{
  for (int j = 0; j < 10; j++)
  {
    int i_left = i - j;
    int i_right = i + j;
    int i_up = i - j * global_costmap.info.width;
    int i_down = i + j * global_costmap.info.width;
    if (int(*(global_costmap.data.begin() + i_left)) > 0)
    {
      return true;
      if (int(*(global_costmap.data.begin() + i_right)) > 0)
      {
        return true;
        if (int(*(global_costmap.data.begin() + i_up)) > 0)
        {
          return true;
          if (int(*(global_costmap.data.begin() + i_down)) > 0)
          {
            return true;
          }
        }
      }
    }
  }
  return false;
}

cv::Mat calculate_dist_mat(obstacle_prediction::ObstacleArray pre, obstacle_prediction::ObstacleArray curr)
{
  int cols = pre.obstacles.size();
  int rows = curr.obstacles.size();
  cv::Mat dist_mat = cv::Mat(rows, cols, CV_32F);
  std::vector<obstacle_prediction::Obstacle>::iterator pre_it = pre.obstacles.begin();
  for (int i = 0; i < cols; i++)
  {
    std::vector<obstacle_prediction::Obstacle>::iterator curr_it = curr.obstacles.begin();
    for (int j = 0; j < rows; j++)
    {
      float dist = sqrt(
          (pre_it->position.x - curr_it->position.x) * (pre_it->position.x - curr_it->position.x) +
          (pre_it->position.y - curr_it->position.y) * (pre_it->position.y - curr_it->position.y));
      if (dist<dist_threshold)
      {
        dist_mat.at<float>(j, i) = dist;
      }
      else
      {
        dist_mat.at<float>(j, i) = -1;
      }
      curr_it++;
    }
    pre_it++;
  }
  return dist_mat;
}

void kf_tracker()
{
  obstacle_prediction::ObstacleArray obstacles;
  if (cloud_filtered->size() > 0)
  {
    curr_time = ros::Time::now().toSec();
    float dt = curr_time - pre_time;

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud_filtered);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.5);
    ec.setMinClusterSize(5);
    ec.setMaxClusterSize(2000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_filtered);
    ec.extract(cluster_indices);
    obst_num = cluster_indices.size();
    if (obst_num > 0)
    {
      //保证卡尔曼滤波器大于等于当前障碍物数量
      if (kf_v.size() < obst_num)
      {
        for (int i = kf_v.size(); i < obst_num; i++)
        {
          cv::KalmanFilter kf(6, 2, 0, CV_32F);
          setIdentity(kf.measurementMatrix);
          float sigmaP = 0.01;
          float sigmaQ = 0.1;
          setIdentity(kf.processNoiseCov, cv::Scalar::all(sigmaP));
          setIdentity(kf.measurementNoiseCov, cv::Scalar(sigmaQ));
          KFO kfo;
          kfo.ocp = false;
          kfo.kf = kf;
          kf_v.push_back(kfo);
        }
      }
      for (int i = 0; i < kf_v.size(); i++)
      {
        kf_v[i].ocp = false;
        kf_v[i].kf.transitionMatrix = (cv::Mat_<float>(6, 6) << 1, 0, dt, 0, 0.5 * pow(dt, 2), 0,
                                      0, 1, 0, dt, 0, 0.5 * pow(dt, 2),
                                      0, 0, 1, 0, dt, 0,
                                      0, 0, 0, 1, 0, dt,
                                      0, 0, 0, 0, 1, 0,
                                      0, 0, 0, 0, 0, 1);
      }

      int id = 0;
      curr_obst_arr.obstacles.resize(0);
      for (auto ptid_it = cluster_indices.rbegin(); ptid_it != cluster_indices.rend(); ++ptid_it)
      {
        float sum_x = 0.0;
        float sum_y = 0.0;
        float max_x = -100;
        float max_y = -100;
        float min_x = 100;
        float min_y = 100;
        for (std::vector<int>::iterator pit = ptid_it->indices.begin(); pit != ptid_it->indices.end(); pit++)
        {
          if (max_x < cloud_filtered->points[*pit].x)
          {
            max_x = cloud_filtered->points[*pit].x;
          }
          if (max_y < cloud_filtered->points[*pit].y)
          {
            max_y = cloud_filtered->points[*pit].y;
          }
          if (min_x > cloud_filtered->points[*pit].x)
          {
            min_x = cloud_filtered->points[*pit].x;
          }
          if (min_y > cloud_filtered->points[*pit].y)
          {
            min_y = cloud_filtered->points[*pit].y;
          }
          sum_x += cloud_filtered->points[*pit].x;
          sum_y += cloud_filtered->points[*pit].y;
        }
        obstacle_prediction::Obstacle obstacle;
        if(first_frame)
        {
          obstacle.id = id;
        }
        else
        {
          obstacle.id = -1;
        }
        obstacle.position.x = sum_x / ptid_it->indices.size();
        obstacle.position.y = sum_y / ptid_it->indices.size();
        obstacle.position.z = 0.0;
        obstacle.length= max_x - min_x + 0.1;
        obstacle.width = max_y - min_y + 0.1;
        obstacle.point_num = ptid_it->indices.size();
        curr_obst_arr.obstacles.push_back(obstacle);
        id++;
      }
      cv::Mat dist_mat = calculate_dist_mat(pre_obst_arr,curr_obst_arr);
      std::cout<<"obst_num:"<<obst_num<<std::endl;
      std::cout<<dist_mat<<std::endl;
      //通过dist_mat给当前障碍物的id赋值
      if(!first_frame)
      {
        for (int row = 0; row < obst_num; row++)
        {
          for(int col= 0; col<pre_obst_arr.obstacles.size();col++)
          {
            if (dist_mat.at<float>(row,col)>0)
            {
              curr_obst_arr.obstacles[row].id = pre_obst_arr.obstacles[col].id;
              kf_v[curr_obst_arr.obstacles[row].id].ocp = true;
              dist_mat.at<float>(row,col)=-1;
            }
          }
        }
      }
      
      for(int i=0;i<curr_obst_arr.obstacles.size();i++)
      {
        if(curr_obst_arr.obstacles[i].id==-1)
        {
          for(int j = 0;j<kf_v.size();j++)
          {
            if(!kf_v[j].ocp)
            {
              curr_obst_arr.obstacles[i].id = j;
              kf_v[j].ocp = true;
              kf_v[j].kf.statePost.at<float>(0) = curr_obst_arr.obstacles[i].position.x;
              kf_v[j].kf.statePost.at<float>(1) = curr_obst_arr.obstacles[i].position.y;
            }
          }
        }
      }
      
      if (first_frame)
      {
        // 第一帧时初始化卡尔曼滤波器
        for (auto obst:curr_obst_arr.obstacles)
        {
          kf_v[obst.id].kf.statePost.at<float>(0) = obst.position.x;
          kf_v[obst.id].kf.statePost.at<float>(1) = obst.position.y;
          kf_v[obst.id].kf.statePost.at<float>(2) = 0;
          kf_v[obst.id].kf.statePost.at<float>(3) = 0;
          kf_v[obst.id].kf.statePost.at<float>(4) = 0;
          kf_v[obst.id].kf.statePost.at<float>(5) = 0;
          kf_v[obst.id].ocp = true;
        }
      }
      
      else
      {
        for (int i = 0;i<curr_obst_arr.obstacles.size();i++)
        {
          std::cout << curr_obst_arr.obstacles[i].id << ": "<<std::endl;
          if (curr_obst_arr.obstacles[i].id<0)
          {
            continue;
          }
          cv::Mat pred = kf_v[curr_obst_arr.obstacles[i].id].kf.predict();
          std::cout 
          << " x:" << pred.at<float>(0)
          << " y:" << pred.at<float>(1)
          << " v_x:" << pred.at<float>(2)
          << " v_y:" << pred.at<float>(3)
          << " a_x:" << pred.at<float>(4)
          << " a_y:" << pred.at<float>(5) << std::endl;
          curr_obst_arr.obstacles[i].linear.x = pred.at<float>(2);
          curr_obst_arr.obstacles[i].linear.y = pred.at<float>(3);
          float meas[2] = {curr_obst_arr.obstacles[i].position.x,curr_obst_arr.obstacles[i].position.y};
          cv::Mat measMat = cv::Mat(2, 1, CV_32F, meas);
          kf_v[curr_obst_arr.obstacles[i].id].kf.correct(measMat);
        }
      }
      pre_obst_arr.obstacles = curr_obst_arr.obstacles;
      curr_obst_arr.Header.frame_id = "map";
      curr_obst_arr.Header.stamp = ros::Time::now();
      pub_obstacles.publish(curr_obst_arr);
      first_frame = false;
      pre_time = curr_time;
    }
  }
  
}

void global_costmap_cb(const nav_msgs::OccupancyGridConstPtr global_costmap_)
{
  global_costmap = *global_costmap_;
  sub_map = true;
}

void scan_cb(const sensor_msgs::LaserScanConstPtr scan)
{
  if (scan->ranges.size() > 0 && sub_map)
  {
    sensor_msgs::PointCloud2 input;
    projector.projectLaser(*scan, input);
    pcl::fromROSMsg(input, *cloud_local);
    static tf::TransformListener tf_listener;
    tf::StampedTransform trans_scan_to_map;
    try
    {
      tf_listener.waitForTransform("map", scan->header.frame_id, ros::Time(0), ros::Duration(2.0));
      tf_listener.lookupTransform("map", scan->header.frame_id, ros::Time(0), trans_scan_to_map);
    }
    catch (tf::TransformException &ex)
    {
      ROS_ERROR("look up trans_scan_to_base failed,due to:%s", ex.what());
      return;
    }
    pcl_ros::transformPointCloud(*cloud_local, *cloud_global, trans_scan_to_map);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    for (int i = 0; i < cloud_global->size(); i++)
    {
      float x = cloud_global->points[i].x;
      float y = cloud_global->points[i].y;
      int mx = (x - global_costmap.info.origin.position.x) / global_costmap.info.resolution;
      int my = (y - global_costmap.info.origin.position.y) / global_costmap.info.resolution;
      int index = my * global_costmap.info.width + mx;
      if (is_static(index))
      {
        inliers->indices.push_back(i);
      }
    }
    extract.setInputCloud(cloud_global);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*cloud_filtered);
    sensor_msgs::PointCloud2 output;
    cloud_filtered->header.frame_id = "map";
    pcl::toROSMsg(*cloud_filtered, output);
    pub_cloud_filtered.publish(output);
    kf_tracker();
    visulization_method();
  }
  pub_local_costmap_method();
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "obstacle_prediction_node");
  ros::NodeHandle nh;
  ros::Subscriber sub_scan = nh.subscribe("/scan", 1, scan_cb);
  ros::Subscriber sub_global_costmap = nh.subscribe("/global_costmap", 1, global_costmap_cb);
  ros::param::get("dist_threshold",dist_threshold);
  pub_cloud_filtered = nh.advertise<sensor_msgs::PointCloud2>("/cloud_filtered", 1);
  pub_pos = nh.advertise<std_msgs::Float32MultiArray>("/pos", 1);
  pub_markers = nh.advertise<visualization_msgs::MarkerArray>("/obs_markers", 1);
  pub_obstacles = nh.advertise<obstacle_prediction::ObstacleArray>("/obst_arr",1);
  pub_local_costmap = nh.advertise<nav_msgs::OccupancyGrid>("/local_costmap",1);
  ros::spin();
  return 0;
}
