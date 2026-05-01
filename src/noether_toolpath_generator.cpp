#include "rclcpp/rclcpp.hpp"
#include <shape_msgs/msg/mesh.hpp>
#include <shape_msgs/msg/mesh_triangle.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <noether_tpp/tool_path_planners/raster/plane_slicer_raster_planner.h>
#include <noether_tpp/tool_path_planners/raster/direction_generators/fixed_direction_generator.h>
#include <noether_tpp/tool_path_planners/raster/origin_generators/fixed_origin_generator.h>
#include <noether_tpp/mesh_modifiers/normals_from_mesh_faces_modifier.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PolygonMesh.h>
#include <Eigen/Dense>
#include <tf2/LinearMath/Transform.hpp>
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2/LinearMath/Vector3.hpp>

using namespace std::placeholders;

class NoetherToolpathGenerator : public rclcpp::Node
{
public:
    NoetherToolpathGenerator() : Node("noether_toolpath_generator") {
        rclcpp::QoS qos_profile_mesh(10);
        qos_profile_mesh.transient_local();
        rclcpp::QoS qos_profile_path(1);
        qos_profile_path.transient_local();

        path_publisher_ = this->create_publisher<geometry_msgs::msg::PoseArray>(
            "/noether_toolpath", qos_profile_path);
        mesh_subscriber_ = this->create_subscription<moveit_msgs::msg::CollisionObject>(
            "/collision_object", qos_profile_mesh, std::bind(&NoetherToolpathGenerator::callbackGenerateToolpath, this, _1));
        RCLCPP_INFO(this->get_logger(), "Tool path generator running...");
    }
private:
    void callbackGenerateToolpath(const moveit_msgs::msg::CollisionObject::SharedPtr message){
        if(message->meshes.empty()){
            RCLCPP_WARN(this->get_logger(), "Collision object is empty, returning...");
            return;
        }
        if(message->mesh_poses.empty()){
            RCLCPP_WARN(this->get_logger(), "Collision object has no poses, returning...");
            return;
        }
        const auto& ros_mesh = message->meshes[0];
        const auto& ros_pose = message->mesh_poses[0];

        tf2::Transform transform;
        transform.setOrigin(tf2::Vector3(ros_pose.position.x, ros_pose.position.y, ros_pose.position.z));
        tf2::Quaternion q(ros_pose.orientation.x, ros_pose.orientation.y, ros_pose.orientation.z, ros_pose.orientation.w);
        transform.setRotation(q);

        pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
        pcl::PolygonMesh pcl_mesh;

        for(const auto& vertex:ros_mesh.vertices){
            tf2::Vector3 local_v(vertex.x, vertex.y, vertex.z);
            tf2::Vector3 world_v = transform * local_v;
            pcl_cloud.push_back(pcl::PointXYZ(world_v.x(), world_v.y(), world_v.z()));
        }

        pcl::toPCLPointCloud2(pcl_cloud, pcl_mesh.cloud);
    
        for(const auto& triangle:ros_mesh.triangles){
            pcl::Vertices pcl_triangle;
            pcl_triangle.vertices.push_back(triangle.vertex_indices[0]);
            pcl_triangle.vertices.push_back(triangle.vertex_indices[1]);
            pcl_triangle.vertices.push_back(triangle.vertex_indices[2]);
            pcl_mesh.polygons.push_back(pcl_triangle);
        }

        RCLCPP_INFO(this->get_logger(), "PCL Conversion complete! Mesh has %zu polygons.", pcl_mesh.polygons.size());

        // Mesh modification: normal estimation
        noether::NormalsFromMeshFacesMeshModifier normal_modifier;
        try{
            std::vector<pcl::PolygonMesh> processed_meshes = normal_modifier.modify(pcl_mesh);

            if(processed_meshes.empty()){
                RCLCPP_ERROR(this->get_logger(), "Normal modifier returned an empty mesh list.");
                return;
            }
            pcl_mesh = processed_meshes[0];
            RCLCPP_INFO(this->get_logger(), "Normal estimation and modification successful!");
        }catch(const std::exception& e){
            RCLCPP_ERROR(this->get_logger(), "Normal estimation and modification failed: %s", e.what());
            return;
        }

        // Toolpath planner configuration

        // Parallel lines (-1 * UnitZ for top->bottom)
        // auto dir_gen = std::make_unique<noether::FixedDirectionGenerator>(-1 * Eigen::Vector3d::UnitZ());
        // auto origin_gen = std::make_unique<noether::FixedOriginGenerator>(Eigen::Vector3d::Zero());
        // noether::PlaneSlicerRasterPlanner planner(std::move(dir_gen), std::move(origin_gen));
        // planner.setLineSpacing(0.03);
        
        // Crosshatching
        auto dir_y = std::make_unique<noether::FixedDirectionGenerator>(-1 * Eigen::Vector3d::UnitY());
        auto orig_y = std::make_unique<noether::FixedOriginGenerator>(Eigen::Vector3d::Zero());
        noether::PlaneSlicerRasterPlanner planner_y(std::move(dir_y), std::move(orig_y));
        planner_y.setLineSpacing(0.02);

        auto dir_z = std::make_unique<noether::FixedDirectionGenerator>(-1 * Eigen::Vector3d::UnitZ());
        auto orig_z = std::make_unique<noether::FixedOriginGenerator>(Eigen::Vector3d::Zero());
        noether::PlaneSlicerRasterPlanner planner_z(std::move(dir_z), std::move(orig_z));
        planner_z.setLineSpacing(0.02);
        
        // Toolpath generation
        noether::ToolPaths paths;
        try{
            noether::ToolPaths horizontal_paths = planner_y.plan(pcl_mesh);
            noether::ToolPaths vertical_paths = planner_z.plan(pcl_mesh);

            paths = horizontal_paths;
            paths.insert(paths.end(), vertical_paths.begin(), vertical_paths.end());
        }catch(const std::exception& e){
            RCLCPP_ERROR(this->get_logger(), "Toolpath generation failed: %s", e.what());
            return;
        }
        if(paths.empty()){
            RCLCPP_ERROR(this->get_logger(), "No valid toolpaths generated.");
            return;
        }

        // Toolpath visualization
        geometry_msgs::msg::PoseArray toolpath_array_msg;
        toolpath_array_msg.header.frame_id = message->header.frame_id;
        toolpath_array_msg.header.stamp = this->now();

        int total_surfaces = 0;
        int total_sweeps = 0;
        int total_waypoints = 0;

        for(auto& surface:paths){
            total_sweeps+= surface.size();

            for(size_t i = 0; i < surface.size(); i++){
                auto& sweep = surface[i];
                total_waypoints += sweep.size();
                if(i % 2 != 0)
                    std::reverse(sweep.begin(), sweep.end());

                for(const auto& waypoint:sweep){
                    geometry_msgs::msg::Pose pose;

                    pose.position.x = waypoint.translation().x();
                    pose.position.y = waypoint.translation().y();
                    pose.position.z = waypoint.translation().z();

                    Eigen::Quaterniond q(waypoint.rotation());
                    pose.orientation.x = q.x();
                    pose.orientation.y = q.y();
                    pose.orientation.z = q.z();
                    pose.orientation.w = q.w();

                    toolpath_array_msg.poses.push_back(pose);
                }
            }
        }
        RCLCPP_INFO(this->get_logger(), "Rastering Complete! %d Surfaces | %d Sweeps | %d Waypoints.", total_surfaces, total_sweeps, total_waypoints);
        path_publisher_->publish(toolpath_array_msg);
    }

    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr path_publisher_;
    rclcpp::Subscription<moveit_msgs::msg::CollisionObject>::SharedPtr mesh_subscriber_;
};
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<NoetherToolpathGenerator>(); 
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}