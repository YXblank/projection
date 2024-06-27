#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <my_pcl/MergedLaserScan.h>
// 自定义消息类型，用于存储两个激光消息的信息
struct MergedLaserScan
{
    std::vector<float> ranges1; // 第一个激光的距离信息
    std::vector<float> ranges2; // 第二个激光的距离信息
    std::vector<float> angles1; // 第一个激光的角度信息
    std::vector<float> angles2; // 第二个激光的角度信息
};

void callback(const sensor_msgs::LaserScanConstPtr& scan1_msg,
              const sensor_msgs::LaserScanConstPtr& scan2_msg,
              ros::Publisher& merged_scan_pub)
{
    // 创建新的消息对象
    my_pcl::MergedLaserScan merged_scan;

    // 将第一个激光消息的信息存储到新消息对象中
    merged_scan.ranges1 = scan1_msg->ranges;
    std::vector<float> angles1;
    std::vector<float> angles2;
    for (size_t i = 0; i < scan1_msg->ranges.size(); i++)
    {
        float angle1 = scan1_msg->angle_min + scan1_msg->angle_increment * i;
        angles1.push_back(angle1);
    }
    merged_scan.angles1 = angles1;

    // 将第二个激光消息的信息存储到新消息对象中
    merged_scan.ranges2 = scan2_msg->ranges;
    for (size_t i = 0; i < scan2_msg->ranges.size(); i++)
    {
        float angle2 = scan2_msg->angle_min + scan2_msg->angle_increment * i;
        angles2.push_back(angle2);
    }
    merged_scan.angles2 = angles2;

    // 发布新的激光消息
    merged_scan_pub.publish(merged_scan);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laser_sync_node");
    ros::NodeHandle nh;

    // 创建消息过滤器订阅器
    message_filters::Subscriber<sensor_msgs::LaserScan> scan1_sub(nh, "scan1_topic", 1);
    message_filters::Subscriber<sensor_msgs::LaserScan> scan2_sub(nh, "scan2_topic", 1);

    // 定义同步策略，此处选择近似时间同步
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::LaserScan, sensor_msgs::LaserScan> SyncPolicy;
    message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(10), scan1_sub, scan2_sub);

    // 创建新的激光话题发布器
    ros::Publisher merged_scan_pub = nh.advertise<MergedLaserScan>("merged_scan_topic", 10);

    sync.registerCallback(boost::bind(&callback, _1, _2, boost::ref(merged_scan_pub)));

    ros::spin();

    return 0;
}

