#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/PointCloud2.h>

class Kinectnode
{
public:
    Kinectnode() : nh("~")
    {
        depth_image_sub = nh.subscribe("/kinect2/hd/image_depth_rect", 1, &Kinectnode::imageCallback, this);
        pointcloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/pointcloud", 1);
        transformed_point_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/map/transformed_point_cloud", 1);
    }

    void imageCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
    {
        // Transform point cloud from camera_link to map
        tf::StampedTransform transform;
        try
        {
            tf_listener_.lookupTransform("map", msg->header.frame_id, ros::Time(0), transform);
        }
        catch (tf::TransformException &ex)
        {
            ROS_WARN("%s", ex.what());
            return;
        }

        sensor_msgs::PointCloud2 transformed_cloud;
        tf::TransformListener tf_listener;
        tf_listener.transformPointCloud("map", *msg, transformed_cloud);

        // Publish transformed point cloud
        transformed_point_cloud_pub_.publish(transformed_cloud);
    }

private:
    ros::NodeHandle nh;
    ros::Subscriber depth_image_sub;
    ros::Publisher pointcloud_pub_;
    ros::Publisher transformed_point_cloud_pub_;
    tf::TransformListener tf_listener_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "kinect_node");
    Kinectnode kinect_node;
    ros::spin();
    return 0;
}

