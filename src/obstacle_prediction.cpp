#include <ros/ros.h>
#include <opencv2/video/video.hpp>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/OccupancyGrid.h>
#include <obstacle_prediction/Obstacle.h>
#include <obstacle_prediction/ObstacleArray.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <laser_geometry/laser_geometry.h>
#include <tf/transform_listener.h>
#include <vector>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/Float32MultiArray.h>
ros::Publisher pub_cloud_filtered;
ros::Publisher pub_pos;
ros::Publisher pub_markers;
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_local(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_global(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
laser_geometry::LaserProjection projector;
nav_msgs::OccupancyGrid global_costmap;
obstacle_prediction::ObstacleArray pre_obst_arr;
obstacle_prediction::ObstacleArray curr_obst_arr;
int pre_obst_num = 0;
int obst_num = 0;
bool first_frame = true;
float pre_time = 0.0;
float dt = 0.0;
float curr_time = 0.0;
std::vector<cv::KalmanFilter> kf_v;

bool sort_x(obstacle_prediction::Obstacle o1, obstacle_prediction::Obstacle o2)
{
  if (o1.position.x < o2.position.x)
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

void global_costmap_cb(const nav_msgs::OccupancyGridConstPtr global_costmap_)
{
  global_costmap = *global_costmap_;
}

void scan_cb(const sensor_msgs::LaserScanConstPtr scan)
{
  if (scan->ranges.size() > 0 && global_costmap.data.size() > 0)
  {
    sensor_msgs::PointCloud2 input;
    projector.projectLaser(*scan, input);
    pcl::fromROSMsg(input, *cloud_local);
    static tf::TransformListener tf_listener;
    tf::StampedTransform trans_scan_to_map;
    try
    {
      tf_listener.waitForTransform("map", scan->header.frame_id, ros::Time(0), ros::Duration(1.0));
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
  }
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
      dist_mat.at<float>(j, i) = sqrt(
        (pre_it->position.x-curr_it->position.x)*(pre_it->position.x-curr_it->position.x)+
        (pre_it->position.y-curr_it->position.y)*(pre_it->position.y-curr_it->position.y));
      curr_it++;
    }
    pre_it++;
  }
  return dist_mat;
}

obstacle_prediction::ObstacleArray kf_tracker()
{
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
  // 保证卡尔曼滤波的数量大于当前点云数量
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
      kf_v.push_back(kf);
    }
  }
  int id = 0;
  std::vector<cv::KalmanFilter>::iterator kf_it = kf_v.begin();
  if (obst_num > 0)
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
      std::vector<int>::iterator pit;

      for (pit = ptid_it->indices.begin(); pit != ptid_it->indices.end(); pit++)
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
        x += cloud_filtered->points[*pit].x;
        y += cloud_filtered->points[*pit].y;
        numPts++;
      }
      obstacle_prediction::Obstacle obstacle;
      obstacle.position.x = x / numPts;
      obstacle.position.y = y / numPts;
      obstacle.position.z = 0.0;
      obstacle.width = max_x - min_x + 0.1;
      obstacle.length = max_y - min_y + 0.1;
      if (first_frame)
      {
        // 第一帧时初始化卡尔曼滤波器
        kf_it->statePost.at<float>(0) = obstacle.position.x;
        kf_it->statePost.at<float>(1) = obstacle.position.y;
        kf_it->statePost.at<float>(2) = 0;
        kf_it->statePost.at<float>(3) = 0;
        kf_it->statePost.at<float>(4) = 0;
        kf_it->statePost.at<float>(5) = 0;
        obstacle.id = id;
        pre_obst_arr.obstacles.push_back(obstacle);
        id++;
        kf_it++;
      }
      else
      {
        curr_obst_arr.obstacles.push_back(obstacle);
      }
    }
    if (first_frame)
    {
      pre_obst_num = obst_num;
      pre_time = ros::Time::now().toSec();
      sort(pre_obst_arr.obstacles.begin(), pre_obst_arr.obstacles.end(), sort_x);
    }
    else
    {
      sort(pre_obst_arr.obstacles.begin(), pre_obst_arr.obstacles.end(), sort_x);
      sort(curr_obst_arr.obstacles.begin(), curr_obst_arr.obstacles.end(), sort_x);
      // 遍历所有卡尔曼滤波并设置dt
      for (std::vector<cv::KalmanFilter>::iterator kf_it = kf_v.begin(); kf_it != kf_v.end(); kf_it++)
      {
        dt = curr_time - pre_time;
        kf_it->transitionMatrix = (cv::Mat_<float>(6, 6) << 1, 0, dt, 0, 0.5 * pow(dt, 2), 0,
                                   0, 1, 0, dt, 0, 0.5 * pow(dt, 2),
                                   0, 0, 1, 0, dt, 0,
                                   0, 0, 0, 1, 0, dt,
                                   0, 0, 0, 0, 1, 0,
                                   0, 0, 0, 0, 0, 1);
      }

      // 计算距离矩阵
      cv::Mat dist_mat = calculate_dist_mat(pre_obst_arr, curr_obst_arr);

      int min_num = obst_num;

      if (pre_obst_num < obst_num)
      {
        min_num = pre_obst_num;
      }
      for (int count = 0; count < min_num; count++)
      {
        int flag_cols = -1;
        int flag_rows = -1;
        float min_dist = 100;
        for (int i = 0; i < obst_num; i++)
        {
          for (int j = 0; j < pre_obst_num; j++)
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
          curr_obst_arr.obstacles.at(flag_rows).id = pre_obst_arr.obstacles.at(flag_cols).id;
          // curr_obst_arr.obstacles.at(flag_rows).dx = curr_obst_arr.obstacles.at(flag_rows).x - pre_obst_arr.obstacles.at(flag_cols).x;
          // curr_obst_arr.obstacles.at(flag_rows).dy = curr_obst_arr.obstacles.at(flag_rows).y - pre_obst_arr.obstacles.at(flag_cols).y;
          pre_obst_arr.obstacles.at(flag_cols).ocp = true;
          for (int i = 0; i < obst_num; i++)
          {
            dist_mat.at<float>(i, flag_cols) = 100;
          }
          for (int i = 0; i < pre_obst_num; i++)
          {
            dist_mat.at<float>(flag_rows, i) = 100;
          }
        }
      }

      // 遍历当前点云，如果有还没匹配的点云id，就遍历之前的点云，如果有之前的点云没被占有就将，该点云id设置为没被占有的点云id
      int void_id = 0;
      for (std::vector<obstacle_prediction::Obstacle>::iterator it = curr_obst_arr.obstacles.begin(); it != curr_obst_arr.obstacles.end(); it++)
      {
        if (it->id < 0)
        {
          for (std::vector<obstacle_prediction::Obstacle>::iterator obst_it = curr_obst_arr.obstacles.begin(); obst_it != curr_obst_arr.obstacles.end(); obst_it++)
          {
            for (std::vector<obstacle_prediction::Obstacle>::iterator pre_obst_it = pre_obst_arr.obstacles.begin(); pre_obst_it != pre_obst_arr.obstacles.end(); pre_obst_it++)
            {
              if (pre_obst_it->ocp == false)
              {
                obst_it->id = pre_obst_it->id;
                pre_obst_it->ocp = true;
              }
              if (pre_obst_it->id == void_id)
              {
                void_id++;
              }
            }
          }
          if (it->id < 0)
          {
            it->id = void_id;
            it->ocp = true;
            void_id++;
          }
        }
      }

      // 遍历当前点云信息，如果是新添加的点云，则重新设置卡尔曼滤波，如果是有对应点云，则进行correct
      visualization_msgs::MarkerArray obs_markers;
      std_msgs::Float32MultiArray pos;
      int flag = 0;
      for (std::vector<obstacle_prediction::Obstacle>::iterator obst_it = curr_obst_arr.obstacles.begin(); obst_it != curr_obst_arr.obstacles.end(); obst_it++)
      {
        if (obst_it->ocp)
        {
          kf_v.at(obst_it->id).init(6, 2, 0, CV_32F);
          kf_v.at(obst_it->id).transitionMatrix = (cv::Mat_<float>(6, 6) << 1, 0, dt, 0, 0.5 * pow(dt, 2), 0,
                                                    0, 1, 0, dt, 0, 0.5 * pow(dt, 2),
                                                    0, 0, 1, 0, dt, 0,
                                                    0, 0, 0, 1, 0, dt,
                                                    0, 0, 0, 0, 1, 0,
                                                    0, 0, 0, 0, 0, 1);
          setIdentity(kf_v.at(obst_it->id).measurementMatrix);
          float sigmaP = 0.01;
          float sigmaQ = 0.1;
          setIdentity(kf_v.at(obst_it->id).processNoiseCov, cv::Scalar::all(sigmaP));
          setIdentity(kf_v.at(obst_it->id).measurementNoiseCov, cv::Scalar(sigmaQ));
          kf_v.at(obst_it->id).statePost.at<float>(0) = obst_it->position.x;
          kf_v.at(obst_it->id).statePost.at<float>(1) = obst_it->position.y;
          kf_v.at(obst_it->id).statePost.at<float>(2) = 0;
          kf_v.at(obst_it->id).statePost.at<float>(3) = 0;
          kf_v.at(obst_it->id).statePost.at<float>(4) = 0;
          kf_v.at(obst_it->id).statePost.at<float>(5) = 0;
        }
        else
        {
          cv::Mat pred = kf_v.at(obst_it->id).predict();
          visualization_msgs::Marker marker;
          marker.id = flag;
          marker.type = visualization_msgs::Marker::CUBE;
          marker.header.frame_id = "map";
          marker.scale.x = obst_it->width;
          marker.scale.y = obst_it->length;
          marker.scale.z = 1.75;
          marker.action = visualization_msgs::Marker::ADD;
          marker.color.r = flag % 2 ? 1 : 0;
          marker.color.g = flag % 3 ? 1 : 0;
          marker.color.b = flag % 4 ? 1 : 0;
          marker.color.a = 0.5;
          marker.lifetime = ros::Duration(0.5);

          std::cout << flag << ": "
               << " x:" << obst_it->position.x
               << " y:" << obst_it->position.y
               << " v_x:" << pred.at<float>(2)
               << " v_y:" << pred.at<float>(3)
               << " a_x:" << pred.at<float>(4)
               << " a_y:" << pred.at<float>(5) << std::endl;

          pos.data.push_back(obst_it->position.x);
          pos.data.push_back(obst_it->position.y);
          for (int k = 2; k < 6; k++)
          {
            pos.data.push_back(pred.at<float>(k));
          }
          pos.data.push_back(obst_it->width);
          pos.data.push_back(obst_it->length);
          marker.pose.position.x = obst_it->position.x;
          marker.pose.position.y = obst_it->position.y;
          marker.pose.position.z = 0.375;
          obs_markers.markers.push_back(marker);

          float meas[2] = {obst_it->position.x, obst_it->position.y};
          cv::Mat measMat = cv::Mat(2, 1, CV_32F, meas);

          kf_v.at(obst_it->id).correct(measMat);
          flag++;
        }
      }
      // 遍历卡尔曼滤波器predict并且发布障碍物pos信息
      pub_pos.publish(pos);
      pub_markers.publish(obs_markers);
      pre_obst_arr.obstacles.swap(curr_obst_arr.obstacles);
    }
    pre_time = curr_time;
    first_frame = false;
    pre_obst_num = obst_num;
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "obstacle_prediction_node");
  ros::NodeHandle nh;
  ros::Subscriber sub_scan = nh.subscribe("/scan", 1, scan_cb);
  ros::Subscriber sub_global_costmap = nh.subscribe("/global_costmap", 1, global_costmap_cb);
  pub_cloud_filtered = nh.advertise<sensor_msgs::PointCloud2>("/cloud_filtered", 1);
  pub_pos = nh.advertise<std_msgs::Float32MultiArray>("pos", 1); // clusterCenter1
  pub_markers = nh.advertise<visualization_msgs::MarkerArray>("obs_markers", 1);
  ros::spin();
  return 0;
}
