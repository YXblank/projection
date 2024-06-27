#include <ros/ros.h>
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
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/impl/instantiate.hpp>
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
class Kinectnode
{
public:
    Kinectnode() : nh("~")
    {
        //start=clock();
        //color_image_sub.subscribe(nh, "/kinect2/hd/image_color_rect", 1);
        depth_image_sub = nh.subscribe("/kinect2/hd/image_depth_rect",1, &Kinectnode::imageCallback, this);
        pointcloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/pointcloud", 1);
        ros:: Rate loop_rate(30);
       

       // sync.reset(new Sync(MySyncPolicy(10), color_image_sub, depth_image_sub));
        //sync->registerCallback(boost::bind(&Kinectnode::imageCallback, this));
        scan_pub = nh.advertise<sensor_msgs::LaserScan>("/scan1", 1);
    }

    /*void removeGroundPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
    {
        // 计算点云法向量
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        ne.setInputCloud(cloud);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        ne.setSearchMethod(tree);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
        ne.setKSearch(50);
        ne.compute(*cloud_normals);

        // 创建平面模型
        pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
        seg.setNormalDistanceWeight(0.1);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(1000);
        seg.setDistanceThreshold(0.02);
        seg.setInputCloud(cloud);
        seg.setInputNormals(cloud_normals);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.size() > 0)
        {
            // 创建提取器对象
            pcl::ExtractIndices<pcl::PointXYZ> extract;
            extract.setInputCloud(cloud);
            extract.setIndices(inliers);

            // 提取非平面点云
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
            extract.setNegative(true);
            extract.filter(*cloud_filtered);

            // 发布非平面点云消息
            sensor_msgs::PointCloud2 filtered_msg;
            pcl::toROSMsg(*cloud, filtered_msg);
            filtered_msg.header.frame_id = "camera_link"; 
            pointcloud_pub_.publish(filtered_msg);
            sensor_msgs::LaserScan scan_msg;
            // Fill in the LaserScan message with processed point cloud data
            // ...
            //pointCloudToLaserScan(cloud_filtered, scan_msg);
            //scan_pub.publish(scan_msg);
            cloudCallback(cloud);
        }
        else
        {
            ROS_WARN("No plane found in the point cloud.");
        }

        // 创建分割器
        //pcl::SACSegmentation<pcl::PointXYZ> seg;
        //pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        //pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

        // 设置分割器参数
        //seg.setOptimizeCoefficients(true);
        //seg.setModelType(pcl::SACMODEL_PLANE);
        //seg.setMethodType(pcl::SAC_RANSAC);
        //seg.setMaxIterations(1000);
        //seg.setDistanceThreshold(0.01);

        // 执行地面拟合
        //seg.setInputCloud(cloud);
        //seg.segment(*inliers, *coefficients);

        // 提取地面点索引
        //pcl::ExtractIndices<pcl::PointXYZ> extract;
        //extract.setInputCloud(cloud);
        //extract.setIndices(inliers);
        //extract.setNegative(true);

        // 过滤掉地面点
        ///pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        //extract.filter(*filtered_cloud);
        //pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        //pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

        // Create the segmentation object
        //pcl::SACSegmentation<pcl::PointXYZ> seg;
        //seg.setOptimizeCoefficients(true);
        //seg.setModelType(pcl::SACMODEL_PLANE);
        //seg.setMethodType(pcl::SAC_RANSAC);
        //seg.setMaxIterations(1000);
        //seg.setDistanceThreshold(0.5);

        // Segment the largest planar component from the input cloud
        //seg.setInputCloud(cloud);
        //seg.segment(*inliers, *coefficients);

        //if (inliers->indices.size() == 0)
        //{
        //PCL_ERROR("Could not estimate a planar model for the given dataset.");
        //return;
        //}

        // Extract the inliers
        //pcl::ExtractIndices<pcl::PointXYZ> extract;
        //extract.setInputCloud(cloud);
        //extract.setIndices(inliers);
        //extract.setNegative(true);
        //extract.filter(*cloud);
    }
    void downsamplePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
    {   
        sensor_msgs::LaserScan scan_msg;
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(cloud);
        vg.setLeafSize(0.01f, 0.01f, 0.1f); // 设置体素的大小
        vg.filter(*cloud);
        pointCloudToLaserScan(cloud, scan_msg);
        scan_pub.publish(scan_msg);
    }
    void removeOutliers(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
    {
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setMeanK(5);            // 设置用于计算邻域点平均距离的 K 值
        sor.setStddevMulThresh(1.0); // 设置标准差乘数阈值
        sor.filter(*cloud);
    }
    */
    void pointCloudToLaserScan(const pcl::PointCloud<pcl::PointXYZ>::Ptr &_pointcloud, sensor_msgs::LaserScan &scan_msg)
    {
        
        // 设置激光扫描的参数
        scan_msg.header.stamp = ros::Time::now();
        /*
        scan_msg.header.frame_id = "laser";
        //scan_msg.angle_min = -M_PI / 2;                         // 最小角度
        //scan_msg.angle_max = M_PI / 2;                          // 最大角度
        //scan_msg.angle_increment = M_PI / cloud->points.size(); // 角度增量
       // scan_msg.time_increment = 0.0;                          // 时间增量
       // scan_msg.scan_time = 0.1;                               // 扫描时间
       //scan_msg.range_min = 0;                               // 最小测距范围
        //scan_msg.range_max = 20.0;                             // 最大测距范围

        // 将点云数据转换为激光扫描数据
        scan_msg.ranges.resize(cloud->points.size());
        for (size_t i = 0; i < cloud->points.size(); ++i)
        {
            scan_msg.ranges[i] = std::sqrt(cloud->points[i].x * cloud->points[i].x +
                                           cloud->points[i].y * cloud->points[i].y);
        }
        */
       float angle_min, angle_max, range_min, range_max, angle_increment;
    // angle_min = -3.14159274101;
    // angle_max = 3.14159274101;
    // range_min = 0.0;
    // range_max = 40;
        angle_min = -M_PI;
        angle_max = M_PI;
        range_min = 0;
        range_max = 50;

        //角度分辨率，分辨率越小，转换后的误差越小
        //angle_increment = 0.00873878411949;
        // angle_increment = 0.00436332309619;
        angle_increment = 0.008727; //计算扫描点个数
        unsigned int beam_size = ceil((M_PI) / angle_increment);

        scan_msg.angle_min = angle_min;
        scan_msg.angle_max = angle_max;
        scan_msg.range_min = range_min;
        scan_msg.range_max = range_max;
        scan_msg.angle_increment = angle_increment;
        // output.time_increment = 0.000141855867696;
        // output.scan_time = 0.101994365454;
        scan_msg.time_increment = 3.35000768246e-05;
        scan_msg.scan_time = 0.0670001506805;
        //scan_msg.header.stamp = transform_time;
        scan_msg.header.frame_id = "laser";
        //end= clock();
        //double relocation_interval_time = double((end-start))/CLOCKS_PER_SEC;
        //td::cout<<"relocation_interval_time:  "<<relocation_interval_time<<std::endl;
        //先将所有数据用nan填充
        scan_msg.ranges.assign(beam_size, std::numeric_limits<float>::infinity());
        scan_msg.intensities.assign(beam_size, std::numeric_limits<float>::infinity());
        for (size_t i = 0; i < _pointcloud->points.size(); ++i)
        {
            //scan_msg.ranges[i] = std::sqrt(cloud->points[i].x * cloud->points[i].x +cloud->points[i].y * cloud->points[i].y);
            float range = hypot(_pointcloud->points[i].y, (-1)*_pointcloud->points[i].x);
            float angle = atan2(_pointcloud->points[i].y, (-1)*_pointcloud->points[i].x);
            int index = (int)((angle - scan_msg.angle_min) / scan_msg.angle_increment);
            if (index >= 0 && index < beam_size)
            {
                //如果当前内容为nan，则直接赋值
                if (std::isinf(scan_msg.ranges[index]))
                {
                    scan_msg.ranges[index] = range;
                }
                //否则，只有距离小于当前值时，才可以重新赋值
                else
                {
                    if (range < scan_msg.ranges[index])
                    {
                        // for(int i=index;i<index+5;i++)
                        // {
                        //     // if(fabs(output.ranges[i]-output.ranges[index])>0.4)
                        //     // {
                        //     //     output.ranges[i] = range;
                        //     // }
                        //     // output.ranges[i] = range;
                        // }
                        scan_msg.ranges[index] = range;
                    }
                }
                scan_msg.intensities[index] = 0; // point.intensity;
            }
        }
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& depth_image_msg)
    {
       // cv_bridge::CvImagePtr color_ptr;
        cv_bridge::CvImagePtr depth_ptr;

        try
        {
            //color_ptr = cv_bridge::toCvCopy(color_image_msg, sensor_msgs::image_encodings::BGR8);
            depth_ptr = cv_bridge::toCvCopy(depth_image_msg, sensor_msgs::image_encodings::TYPE_16UC1);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        //cv::Mat color_image = color_ptr->image;
        cv::Mat depth_image = depth_ptr->image;
        rgbdToPointCloud(depth_image,cloud);
        publishPointCloud(cloud);
        
    }
    void publishPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
    {
        
     
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud1(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        // Rotate around x-axis by 15 degrees
        //Eigen::Affine3f transform_x = Eigen::Affine3f::Identity();
        //transform_x.rotate(Eigen::AngleAxisf((M_PI /4+(1*M_PI)), Eigen::Vector3f::UnitX()));
        //pcl::transformPointCloud(*cloud, *transformed_cloud, transform_x);
        //Eigen::Affine3f transform_Z = Eigen::Affine3f::Identity();
        //transform_Z.rotate(Eigen::AngleAxisf(M_PI+(M_PI/4), Eigen::Vector3f::UnitZ()));
        //pcl::transformPointCloud(*cloud, *transformed_cloud, transform_Z);
        // Rotate around y-axis by 25 degrees
        //Eigen::Affine3f transform_y = Eigen::Affine3f::Identity();
        //transform_y.rotate(Eigen::AngleAxisf(M_PI , Eigen::Vector3f::UnitY()));
        //::transformPointCloud(*cloud, *transformed_cloud, transform_y);

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
       /*
        for (std::size_t i = 0; i < transformed_cloud1->points.size(); ++i)
        {
            std::cout << "    " << transformed_cloud1->points[i].x
                      << " " << transformed_cloud1->points[i].y
                      << " " << transformed_cloud1->points[i].z << std::endl;
           // if (transformed_cloud1->points[i].z<0.6 & /*transformed_cloud1->points[i].z>0.47 &*/// transformed_cloud1->points[i].y<0.46 & transformed_cloud1->points[i].x>-0.16 & transformed_cloud1->points[i].x<0.2 )
          // {
             //  transformed_cloud->points.push_back(transformed_cloud1->points[i]);
            //}
            
       // }
        
        sensor_msgs::PointCloud2 cloud_msg;
        sensor_msgs::LaserScan scan_msg;
        //cloudCallback(cloud);
        pcl::toROSMsg(*transformed_cloud1, cloud_msg);
        cloud_msg.header.frame_id = "map"; // Change frame_id as needed
        //removeGroundPlane(transformed_cloud);
        // Publish the transformed point cloud
        pointcloud_pub_.publish(cloud_msg);
        pointCloudToLaserScan(transformed_cloud1, scan_msg);
        scan_pub.publish(scan_msg);
    }
       
    
/*
    void cloudCallback(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
    {
        //removeGroundPlane(cloud);
        //double crop_box_min_z=0.0, crop_box_max_z=3.0;
  
        pcl::CropBox<pcl::PointXYZ> cropBox;
        cropBox.setInputCloud(cloud);
        cropBox.setMin(Eigen::Vector4f(-5, -5, 0.02, 1.0f));
        cropBox.setMax(Eigen::Vector4f(4, 4, 3,1.0f));
    //ROS_INFO("crop_box  max_y: %f, min_y: %f", crop_box_max_y, crop_box_min_y);
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtercloud(new pcl::PointCloud<pcl::PointXYZ>);
        cropBox.filter(*filtercloud);
        // Ground plane removal, voxel filtering, downsampling, outlier removal code goes here
        
        downsamplePointCloud(filtercloud);
        //removeGroundPlane(filtercloud);
        //removeOutliers(filtercloud);
        // Convert processed point cloud to LaserScan and publish
        //sensor_msgs::LaserScan scan_msg;
        // Fill in the LaserScan message with processed point cloud data
        // ...
        //pointCloudToLaserScan(filtercloud, scan_msg);
        //scan_pub.publish(scan_msg);
    }
    */


  
    void rgbdToPointCloud(cv::Mat &depth_image,
                          //cv::Mat &color_image,
                          pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
    {
        const double fx = 1053.0;
        const double fy = 1053.0;
        const double cx = 972;
        const double cy = 523;

        // 创建点云对象
        cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);

        for (int v = 0; v < depth_image.rows; ++v)
        {
            for (int u = 0; u < depth_image.cols; ++u)
            {
                // 读取深度值
                float depth_value = depth_image.at<uint16_t>(v, u) * 0.001; // 深度值转换为米

                // 忽略深度值为 0 的点
                if (depth_value == 0.0)
                {
                    continue;
                }

                // 计算三维坐标
                pcl::PointXYZ point;
                point.z = depth_value;
                point.x = (-1)*((u - cx) * depth_value / fx);
                point.y = ((v - cy) * depth_value / fy);

                // 从彩色图像中获取颜色信息
                /*cv::Vec3b color = color_image.at<cv::Vec3b>(v, u);
                point.r = color[2];
                point.g = color[1];
                point.b = color[0];
*/
                // 将点加入点云
                cloud->points.push_back(point);
            }
        }

        cloud->width = cloud->points.size();
        cloud->height = 1;
    }

private:
    ros::NodeHandle nh;
   // message_filters::Subscriber<sensor_msgs::Image> color_image_sub;
    ros::Subscriber depth_image_sub;
    //typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
    //typedef message_filters::Synchronizer<MySyncPolicy> Sync;
   // boost::shared_ptr<Sync> sync;
    ros::Publisher scan_pub;
    ros::Publisher pointcloud_pub_;
};


int main(int argc, char **argv)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cv::Mat depth_image;
    cv::Mat color_image;
    // Initialize ROS node
    ros::init(argc, argv, "kinect_node");
    ros::NodeHandle nh;

    Kinectnode processor;

    

    ros::spin();
    
    return 0;
}
