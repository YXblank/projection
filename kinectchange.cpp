#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/LaserScan.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/time.h>
#include <pcl/ModelCoefficients.h>
#include <sensor_msgs/Image.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/crop_box.h>
#include <ros/ros.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/impl/instantiate.hpp>
#include <pcl/common/transforms.h>
class Kinectnode
{
public:
    Kinectnode() : nh("~")
    {
        depth_image_sub = nh.subscribe("/kinect2/hd/image_depth_rect",1, &Kinectnode::imageCallback, this);
        pointcloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/pointcloud", 1);
	transformed_point_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/map/transformed_point_cloud", 1);
	tf_buffer_ = new tf2_ros::Buffer();
        tf_listener_ = new tf2_ros::TransformListener(*tf_buffer_);
	
        ros:: Rate loop_rate(30);    
        scan_pub = nh.advertise<sensor_msgs::LaserScan>("/scan1", 1);
    }
    ~Kinectnode() {
        delete tf_buffer_;
        delete tf_listener_;
    }
    void pointCloudToLaserScan(const pcl::PointCloud<pcl::PointXYZ>::Ptr &_pointcloud, sensor_msgs::LaserScan &scan_msg)
    {
        scan_msg.header.stamp = ros::Time::now();      
        float angle_min, angle_max, range_min, range_max, angle_increment;
        angle_min = -M_PI;
        angle_max = M_PI;
        range_min = 0;
        range_max = 50;
        angle_increment = 0.008727; //计算扫描点个数
        unsigned int beam_size = ceil((M_PI) / angle_increment);
        scan_msg.angle_min = angle_min;
        scan_msg.angle_max = angle_max;
        scan_msg.range_min = range_min;
        scan_msg.range_max = range_max;
        scan_msg.angle_increment = angle_increment;
        scan_msg.time_increment = 3.35000768246e-05;
        scan_msg.scan_time = 0.0670001506805;
        scan_msg.header.frame_id = "laser";
        scan_msg.ranges.assign(beam_size, std::numeric_limits<float>::infinity());
        scan_msg.intensities.assign(beam_size, std::numeric_limits<float>::infinity());
        for (size_t i = 0; i < _pointcloud->points.size(); ++i)
        {
            float range = hypot(_pointcloud->points[i].y, (-1)*_pointcloud->points[i].x);
            float angle = atan2(_pointcloud->points[i].y, (-1)*_pointcloud->points[i].x);
            int index = (int)((angle - scan_msg.angle_min) / scan_msg.angle_increment);
            if (index >= 0 && index < beam_size)
            {
                if (std::isinf(scan_msg.ranges[index]))
                {
                    scan_msg.ranges[index] = range;
                }
                else
                {
                    if (range < scan_msg.ranges[index])
                    {
               
                        scan_msg.ranges[index] = range;
                    }
                }
                scan_msg.intensities[index] = 0; // point.intensity;
            }
        }
    }
    void imageCallback(const sensor_msgs::ImageConstPtr& depth_image_msg)
    {
        cv_bridge::CvImagePtr depth_ptr;
        try
        {
            depth_ptr = cv_bridge::toCvCopy(depth_image_msg, sensor_msgs::image_encodings::TYPE_16UC1);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        cv::Mat depth_image = depth_ptr->image;
        rgbdToPointCloud(depth_image,cloud);
        publishPointCloud(cloud);        
    }
    void publishPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud1(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        //vertical
        Eigen::Affine3f transform_x1 = Eigen::Affine3f::Identity();
        transform_x1.rotate(Eigen::AngleAxisf((3*M_PI/2), Eigen::Vector3f::UnitZ()));
        pcl::transformPointCloud(*cloud, *transformed_cloud1, transform_x1);

        Eigen::Affine3f transform_y = Eigen::Affine3f::Identity();
        transform_y.rotate(Eigen::AngleAxisf(3*M_PI/4 , Eigen::Vector3f::UnitY()));
        transformPointCloud(*transformed_cloud1, *transformed_cloud1, transform_y);
        Eigen::Affine3f transform_x = Eigen::Affine3f::Identity();
        
        transform_x.rotate(Eigen::AngleAxisf((M_PI), Eigen::Vector3f::UnitX()));
        pcl::transformPointCloud(*transformed_cloud1, *transformed_cloud1, transform_x);

        sensor_msgs::PointCloud2 cloud_msg;
        sensor_msgs::LaserScan scan_msg;
        pcl::toROSMsg(*transformed_cloud1, cloud_msg);
        cloud_msg.header.frame_id = "camera_link"; // Change frame_id as needed
        pointcloud_pub_.publish(cloud_msg);
        pointCloudToLaserScan(transformed_cloud1, scan_msg);
        scan_pub.publish(scan_msg);
	pointCloudCallback(cloud_msg);
    }          
  void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        try {
            geometry_msgs::TransformStamped transform = tf_buffer_->lookupTransform("map", msg->header.frame_id, ros::Time(0), ros::Duration(1.0));
            geometry_msgs::PointCloud transformed_point_cloud;
            tf2::doTransform(*msg, transformed_point_cloud, transform);
            sensor_msgs::PointCloud2 transformed_pc_msg;
            sensor_msgs::convertPointCloudToPointCloud2(transformed_point_cloud, transformed_pc_msg);
            transformed_point_cloud_pub_.publish(transformed_pc_msg);
        }
        catch (tf2::TransformException& ex) {
            ROS_WARN("%s", ex.what());
        }
    }  
    void rgbdToPointCloud(cv::Mat &depth_image,
                          pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
    {
        const double fx = 1053.0;
        const double fy = 1053.0;
        const double cx = 972;
        const double cy = 523;

        cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);

        for (int v = 0; v < depth_image.rows; ++v)
        {
            for (int u = 0; u < depth_image.cols; ++u)
            {
                float depth_value = depth_image.at<uint16_t>(v, u) * 0.001; // 深度值转换为米
                if (depth_value == 0.0)
                {
                    continue;
                }
                pcl::PointXYZ point;
                point.z = depth_value;
                point.x = (-1)*((u - cx) * depth_value / fx);
                point.y = ((v - cy) * depth_value / fy);
                cloud->points.push_back(point);
            }
        }
        cloud->width = cloud->points.size();
        cloud->height = 1;
    }
private:
    ros::NodeHandle nh;
    ros::Subscriber depth_image_sub;   
    ros::Publisher scan_pub;
    ros::Publisher pointcloud_pub_;
    tf2_ros::Buffer* tf_buffer_;
    tf2_ros::TransformListener* tf_listener_;
    ros::Subscriber point_cloud_sub_;
    ros::Publisher transformed_point_cloud_pub_;
};
int main(int argc, char **argv)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cv::Mat depth_image;
    cv::Mat color_image;
    ros::init(argc, argv, "kinect_node");
    ros::NodeHandle nh;
    Kinectnode processor;
    ros::spin();  
    return 0;
}


