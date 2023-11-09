#include<ros/ros.h>
#include<sensor_msgs/LaserScan.h>
#include<nav_msgs/OccupancyGrid.h>
#include<pcl/filters/extract_indices.h>
#include<pcl_conversions/pcl_conversions.h>
#include<pcl_ros/transforms.h>
#include<laser_geometry/laser_geometry.h>
#include<tf/transform_listener.h>
ros::Publisher pub_cloud_filtered;
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_local(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_global(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
laser_geometry::LaserProjection projector;
nav_msgs::OccupancyGrid global_costmap;

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
    if(scan->ranges.size()>0 && global_costmap.data.size()>0)
    {
        sensor_msgs::PointCloud2 input;
        projector.projectLaser(*scan,input);
        pcl::fromROSMsg(input,*cloud_local);
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
        pcl_ros::transformPointCloud(*cloud_local,*cloud_global,trans_scan_to_map);
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
        for(int i=0;i<cloud_global->size();i++)
        {
            float x = cloud_global->points[i].x;
            float y = cloud_global->points[i].y;
            int mx = (x - global_costmap.info.origin.position.x) / global_costmap.info.resolution;
            int my = (y - global_costmap.info.origin.position.y) / global_costmap.info.resolution;
            int index = my * global_costmap.info.width + mx;
            if(is_static(index))
            {
                inliers->indices.push_back(i);
            }
        }
        extract.setInputCloud(cloud_global);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*cloud_filtered);
        sensor_msgs::PointCloud2 output;
        cloud_filtered->header.frame_id="map";
        pcl::toROSMsg(*cloud_filtered,output);
        pub_cloud_filtered.publish(output);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc,argv,"obstacle_prediction_node");
    ros::NodeHandle nh;
    ros::Subscriber sub_scan = nh.subscribe("/scan",1,scan_cb);
    ros::Subscriber sub_global_costmap = nh.subscribe("/global_costmap",1,global_costmap_cb);
    pub_cloud_filtered = nh.advertise<sensor_msgs::PointCloud2>("/cloud_filtered",1);
    ros::spin();
    return 0;
}
