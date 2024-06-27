#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

class MotionFilter {
public:
    MotionFilter() : nh("~"), threshold(0.1) { // 设置默认阈值为0.1
        // 初始化 ROS 节点句柄
        nh.param<std::string>("pointcloud_topic", pointcloud_topic, "/pointcloud");
        
        // 订阅点云数据
        pointcloud_sub = nh.subscribe(pointcloud_topic, 1, &MotionFilter::pointcloudCallback, this);
        
        // 发布滤除后的点云数据
        filtered_pointcloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/filtered_pointcloud", 1);
    }

    // 回调函数，处理点云数据
    void pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr& pointcloud_msg) {
        // 将点云数据转换为pcl点云格式
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*pointcloud_msg, *cloud);

        // 计算点的运动特征，这里简单地使用点的速度作为运动特征
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (size_t i = 1; i < cloud->points.size(); ++i) {
            double dx = cloud->points[i].x - cloud->points[i-1].x;
            double dy = cloud->points[i].y - cloud->points[i-1].y;
            double dz = cloud->points[i].z - cloud->points[i-1].z;
            double speed = std::sqrt(dx*dx + dy*dy + dz*dz); // 计算速度
            if (speed > threshold) {
                filtered_cloud->points.push_back(cloud->points[i]);
            }
        }

        // 将滤除后的点云数据发布出去
        sensor_msgs::PointCloud2 filtered_pointcloud_msg;
        pcl::toROSMsg(*filtered_cloud, filtered_pointcloud_msg);
        filtered_pointcloud_msg.header = pointcloud_msg->header;
        filtered_pointcloud_pub.publish(filtered_pointcloud_msg);
    }

private:
    ros::NodeHandle nh;
    ros::Subscriber pointcloud_sub;
    ros::Publisher filtered_pointcloud_pub;
    std::string pointcloud_topic;
    double threshold; // 运动阈值
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "motion_filter_node");
    
    // 创建运动滤波对象
    MotionFilter motion_filter;
    ros::spin();
    return 0;
}

