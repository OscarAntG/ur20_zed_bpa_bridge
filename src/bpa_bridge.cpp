#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/mesh.hpp>
#include <shape_msgs/msg/mesh_triangle.hpp>
#include <glm/glm.hpp>
#include "bpa.h"

using namespace std::placeholders;

class BpaBridge : public rclcpp::Node 
{
public:
    BpaBridge() : Node("bpa_bridge") 
    {
        rclcpp::QoS qos_profile(10);
        qos_profile.transient_local();

        this->declare_parameter<double>("bpa_radius", 0.05);
        accumulated_cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

        mask_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "isolated_mask_cloud", 10, std::bind(&BpaBridge::callbackBPABridge, this, _1));
        surface_marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("mask_marker_surface", 10);
        surface_collision_publisher_ = this->create_publisher<moveit_msgs::msg::CollisionObject>("collision_object", qos_profile);
        reset_surface_server_ = this->create_service<std_srvs::srv::Trigger>(
            "reset_surface", std::bind(&BpaBridge::callbackResetSurface, this, _1,_2));
    }
private:
    bool surface_locked_ = false;
    int current_frame_count_ = 0;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr accumulated_cloud_;

    void callbackBPABridge(const sensor_msgs::msg::PointCloud2::SharedPtr message){
        if(surface_locked_)
            return;
        
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr mask_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        try{
            pcl::fromROSMsg(*message, *mask_cloud);
        } catch(const std::exception& e){
            RCLCPP_ERROR(this->get_logger(), "Failed to convert to PCL point cloud: %s", e.what());
            return;
        }

        int valid_point_count = 0;
        for (const auto& pt : mask_cloud->points) {
            if (std::isfinite(pt.x))
            valid_point_count++;
        }
        if (valid_point_count < 50){
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
                "Waiting for SAM3 target... only %d points found.", valid_point_count);
            return; 
        }
        
        int target_frames = 10;
        *accumulated_cloud_ += *mask_cloud;
        current_frame_count_++;

        RCLCPP_INFO(this->get_logger(), "Accumulating frame %d of %d", current_frame_count_, target_frames);
        if (current_frame_count_ < target_frames)
            return;
        RCLCPP_INFO(this->get_logger(), "Finished collecting point cloud frames.");

        pcl::VoxelGrid<pcl::PointXYZRGB> vg;
        vg.setInputCloud(accumulated_cloud_);
        vg.setLeafSize(0.01f, 0.01f, 0.01f);
        vg.filter(*mask_cloud);

        RCLCPP_INFO(this->get_logger(), "%d valid points, commencing surface reconstruction...", valid_point_count);

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
        marker.ns = "waste_surface_marker";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 1.0; marker.scale.y = 1.0; marker.scale.z = 1.0; 
        marker.color.r = 0.0f; marker.color.g = 0.8f; marker.color.b = 0.2f; marker.color.a = 1.0f;
        
        moveit_msgs::msg::CollisionObject collision_object;
        collision_object.header.frame_id = message->header.frame_id;
        collision_object.header.stamp = rclcpp::Time(0, 0, this->get_clock()->get_clock_type());
        collision_object.id = "mask_collision_surface";
        collision_object.operation = collision_object.ADD;
        shape_msgs::msg::Mesh mesh;
        int vertex_index = 0;

        for (const auto& tri: triangles) {
            shape_msgs::msg::MeshTriangle triangle_indices;
            geometry_msgs::msg::Point p1, p2, p3;

            p1.x = tri[0].x; p1.y = tri[0].y; p1.z = tri[0].z;
            p2.x = tri[1].x; p2.y = tri[1].y; p2.z = tri[1].z;
            p3.x = tri[2].x; p3.y = tri[2].y; p3.z = tri[2].z;

            marker.points.push_back(p1);
            marker.points.push_back(p2);
            marker.points.push_back(p3);

            mesh.vertices.push_back(p1);
            mesh.vertices.push_back(p2);
            mesh.vertices.push_back(p3);
            triangle_indices.vertex_indices[0] = vertex_index++;
            triangle_indices.vertex_indices[1] = vertex_index++;
            triangle_indices.vertex_indices[2] = vertex_index++;
            mesh.triangles.push_back(triangle_indices);
        }
        collision_object.meshes.push_back(mesh);
        geometry_msgs::msg::Pose pose;
        pose.orientation.w = 1.0;
        collision_object.mesh_poses.push_back(pose);

        surface_marker_publisher_->publish(marker);
        surface_collision_publisher_->publish(collision_object);
        surface_locked_ = true;
    }

    void callbackResetSurface(const std::shared_ptr<std_srvs::srv::Trigger::Request> request, const std::shared_ptr<std_srvs::srv::Trigger::Response> response){
        (void)request;
        accumulated_cloud_->clear();
        current_frame_count_ = 0;
        surface_locked_ = false;
        response->success = true;
        response->message = "Surface reset, ready for next snapshot!";
        RCLCPP_INFO(this->get_logger(), "Resetting surface...");
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr mask_subscriber_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr surface_marker_publisher_;
    rclcpp::Publisher<moveit_msgs::msg::CollisionObject>::SharedPtr surface_collision_publisher_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_surface_server_;
};
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BpaBridge>(); 
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}