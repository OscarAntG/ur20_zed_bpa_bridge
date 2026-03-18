#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <glm/glm.hpp>
#include "bpa.h"

using namespace std::placeholders;

class BPABridgeNode : public rclcpp::Node 
{
public:
    BPABridgeNode() : Node("bpa_bridge") 
    {
        this->declare_parameter<double>("bpa_radius", 0.05);

        mask_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "isolated_mask_cloud", 10, std::bind(&BPABridgeNode::callbackBPABridge, this, _1));
        surface_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("mask_surface",10);
    }
private:
    void callbackBPABridge(const sensor_msgs::msg::PointCloud2::SharedPtr message){
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr mask_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        try{
            pcl::fromROSMsg(*message, *mask_cloud);
        } catch(const std::exception& e){
            RCLCPP_ERROR(this->get_logger(), "Failed to convert to PCL point cloud: %s", e.what());
            return;
        }

        if(mask_cloud->empty()){
            RCLCPP_WARN(this->get_logger(), "Received empty point cloud.");
            return;
        }

        pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree = std::make_shared<pcl::search::KdTree<pcl::PointXYZRGB>>();
        ne.setSearchMethod(tree);
        ne.setInputCloud(mask_cloud);
        ne.setKSearch(20);
        pcl::PointCloud<pcl::Normal>::Ptr normals = std::make_shared<pcl::PointCloud<pcl::Normal>>();
        ne.compute(*normals);

        // Bridging ROS2 and BPA
        std::vector<bpa::Point> bpa_input;
        bpa_input.reserve(mask_cloud->points.size());

        for (size_t i = 0; i < mask_cloud->points.size(); ++i) {
            if (std::isfinite(mask_cloud->points[i].x) && std::isfinite(normals->points[i].normal_x)) {
                bpa_input.push_back({
                    glm::vec3(mask_cloud->points[i].x, mask_cloud->points[i].y, mask_cloud->points[i].z),
                    glm::vec3(normals->points[i].normal_x, normals->points[i].normal_y, normals->points[i].normal_z)
                });
            }
        }

        double radius = this->get_parameter("bpa_radius").as_double();
        RCLCPP_INFO(this->get_logger(), "Running BPA on %zu points with radius: %f", bpa_input.size(), radius);
        std::vector<bpa::Triangle> triangles = bpa::reconstruct(bpa_input, radius);

        // Prepare for publishing
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = message->header.frame_id;
        marker.header.stamp = message->header.stamp;
        marker.ns = "bpa_surface";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;

        marker.scale.x = 1.0; marker.scale.y = 1.0; marker.scale.z = 1.0; 

        marker.color.r = 0.0f; marker.color.g = 0.8f; marker.color.b = 0.2f; marker.color.a = 1.0f;

        for (const auto& tri: triangles) {
            geometry_msgs::msg::Point p1, p2, p3;

            p1.x = tri[0].x; p1.y = tri[0].y; p1.z = tri[0].z;
            p2.x = tri[1].x; p2.y = tri[1].y; p2.z = tri[1].z;
            p3.x = tri[2].x; p3.y = tri[2].y; p3.z = tri[2].z;

            marker.points.push_back(p1);
            marker.points.push_back(p2);
            marker.points.push_back(p3);
        }
        surface_publisher_->publish(marker);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr mask_subscriber_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr surface_publisher_;
};
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BPABridgeNode>(); 
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}