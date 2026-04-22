#include "rclcpp/rclcpp.hpp"
#include <std_srvs/srv/trigger.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <shape_msgs/msg/mesh.hpp>
#include <shape_msgs/msg/mesh_triangle.hpp>
#include <Eigen/Dense>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/crop_box.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/normal_3d.h>

using namespace std::placeholders;

class NoetherSurfaceFitting : public rclcpp::Node 
{
public:
    NoetherSurfaceFitting() : Node("noether_surface_fitting") 
    {
        rclcpp::QoS qos_profile(10);
        qos_profile.transient_local();

        this->declare_parameter<std::string>("surface_type", "plane");
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // cloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        //     "/crop_box_point_cloud", 10);
        wall_surface_marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>(
            "/wall_surface_marker", qos_profile);
        wall_surface_mesh_publisher_ = this->create_publisher<shape_msgs::msg::Mesh>(
            "/wall_surface_mesh", qos_profile);
        cloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/zedx/zed_node/point_cloud/cloud_registered", 10, std::bind(&NoetherSurfaceFitting::callbackFitPointCloud, this, _1));
        record_wall_tf_server_= this->create_service<std_srvs::srv::Trigger>(
            "/record_wall_tf", std::bind(&NoetherSurfaceFitting::callbackRecordWallTF, this, _1, _2));
        scan_wall_plane_server_ = this->create_service<std_srvs::srv::Trigger>(
            "/scan_exposed_wall", std::bind(&NoetherSurfaceFitting::callbackScanExposedWall, this, _1, _2));

        RCLCPP_INFO(this->get_logger(), "Noether surface fitting node has started! Wait for surface trigger...");
            }
private:
    bool tf_recorded_ = false;
    bool capture_next_frame_ = false;
    float crop_x_ = 0.0, crop_y_ = 0.0, crop_z_ = 0.0;

    void callbackRecordWallTF(const std::shared_ptr<std_srvs::srv::Trigger::Request> request, const std::shared_ptr<std_srvs::srv::Trigger::Response> response){
        (void) request;
        geometry_msgs::msg::TransformStamped transformStamped;
        
        try{
            transformStamped = tf_buffer_->lookupTransform(
                "zedx_left_camera_frame",   // target frame
                "tool0",                    // source frame, tentatively tool0
                tf2::TimePointZero);

            // Revert when testing with UR20
            crop_x_ = transformStamped.transform.translation.x;
            crop_y_ = transformStamped.transform.translation.y;
            crop_z_ = transformStamped.transform.translation.z;
            // crop_x_ = 2.28;
            // crop_y_ = 0.2;
            // crop_z_ = 0.0;
            tf_recorded_ = true;

            RCLCPP_INFO(this->get_logger(), "Probing Point at [X: %.2f, Y: %.2f, Z: %.2f] from zedx_left_camera_frame", crop_x_, crop_y_, crop_z_);

            response->success = true;
            response->message = "TF point found, retracting arm from camera view...";
        } catch (const tf2::TransformException & ex) {
            RCLCPP_ERROR(this->get_logger(), "TF point not found: %s", ex.what());
            response->success = false;
            response->message = "Failed to locate tool0 in TF tree.";
        }
    }

    void callbackScanExposedWall(const std::shared_ptr<std_srvs::srv::Trigger::Request> request, const std::shared_ptr<std_srvs::srv::Trigger::Response> response){
        (void) request;

        if(!tf_recorded_){
            RCLCPP_ERROR(this->get_logger(), "NO TF COORDINATES RECEIVED, CANNOT SCAN YET!");
            response->success = false;
            response->message = "Call /record_wall_tf service first...";
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Arm retracted, initiating surface fitting...");
        capture_next_frame_ = true;
        response->success = true;
        response->message = "Capturing snapshot for surface fitting...";
    }
    
    void callbackFitPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr message){
        if(!capture_next_frame_){
            return;
        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr raw_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cropped_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

        try{
            pcl::fromROSMsg(*message, *raw_cloud);
        }catch (const std::exception& e){
            RCLCPP_ERROR(this->get_logger(), "PCL conversion failed: %s", e.what());
            return;
        }

        pcl::CropBox<pcl::PointXYZRGB> bounding_box;    // Assuming 1m tool width, bounding centered at 5cm
        bounding_box.setMin(Eigen::Vector4f(crop_x_ - 0.25, crop_y_ - 0.5, crop_z_ - 0.15, 1.0));
        bounding_box.setMax(Eigen::Vector4f(crop_x_ + 0.25, crop_y_ + 0.5, crop_z_ + 0.05, 1.0));
        bounding_box.setInputCloud(raw_cloud);
        bounding_box.filter(*cropped_cloud);

        RCLCPP_INFO(this->get_logger(), "Exposed wall cropping complete. Isolated %zu points.", cropped_cloud->size());
        capture_next_frame_ = false;
        tf_recorded_ = false;

        if(cropped_cloud->points.empty()){
            RCLCPP_WARN(this->get_logger(), "Cropped cloud is empty!");
            return;
        }

        // Wall MSAC Segmentation (Plane or Cylinder)
        std::string surface_type = this->get_parameter("surface_type").as_string();

        shape_msgs::msg::Mesh final_mesh;
        visualization_msgs::msg::Marker final_marker;
        final_marker.header = message->header;
        final_marker.ns = "wall_marker_surface";
        final_marker.id = 1;
        final_marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
        final_marker.action = visualization_msgs::msg::Marker::ADD;
        final_marker.scale.x = 1.0; final_marker.scale.y = 1.0; final_marker.scale.z = 1.0;
        final_marker.color.r = 0.0f; final_marker.color.g = 0.5f; final_marker.color.b = 1.0f; final_marker.color.a = 0.7f;

        if (surface_type == "plane"){
            RCLCPP_INFO(this->get_logger(), "Fitting mode: PLANE");
            pcl::ModelCoefficients::Ptr coefficients = std::make_shared<pcl::ModelCoefficients>();
            pcl::PointIndices::Ptr inliers = std::make_shared<pcl::PointIndices>();
            pcl::SACSegmentation<pcl::PointXYZRGB> seg;
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_PLANE);
            seg.setMethodType(pcl::SAC_MSAC);     // Prior testing had better results with MSAC
            seg.setDistanceThreshold(0.01);
            seg.setInputCloud(cropped_cloud);
            seg.segment(*inliers, *coefficients);

            if(inliers->indices.empty()){
                RCLCPP_ERROR(this->get_logger(), "Plane segmentation failed!");
                return;
            }
            float a = coefficients->values[0];
            float b = coefficients->values[1];
            float c = coefficients->values[2];
            float d = coefficients->values[3];
            RCLCPP_INFO(this->get_logger(), "Plane model successfully segmented! %.2fx + %.2fy + %.2fz + %.2f = 0", a, b, c, d);

            // Plane surface fitting
            Eigen::Vector3f normal(a, b, c);
            normal.normalize();
            
            Eigen::Vector3f plane_center(crop_x_, crop_y_, crop_z_);

            Eigen::Vector3f world_up(0, 0, 1);
            Eigen::Vector3f plane_right = world_up.cross(normal).normalized();
            Eigen::Vector3f plane_up = normal.cross(plane_right).normalized();

            // Surface mesh (Tool path planner)
            float width = 1.0;
            float height = 1.0;

            geometry_msgs::msg::Point p1, p2, p3, p4;
            Eigen::Vector3f corner;

            corner = plane_center - (plane_right * width / 2.0) - (plane_up * height / 2.0);
            p1.x = corner.x(); p1.y = corner.y(); p1.z = corner.z();
            corner = plane_center + (plane_right * width / 2.0) - (plane_up * height / 2.0);
            p2.x = corner.x(); p2.y = corner.y(); p2.z = corner.z();
            corner = plane_center + (plane_right * width / 2.0) + (plane_up * height / 2.0);
            p3.x = corner.x(); p3.y = corner.y(); p3.z = corner.z();
            corner = plane_center - (plane_right * width / 2.0) + (plane_up * height / 2.0);
            p4.x = corner.x(); p4.y = corner.y(); p4.z = corner.z();

            final_mesh.vertices = {p1, p2, p3, p4};
            shape_msgs::msg::MeshTriangle t1, t2;
            t1.vertex_indices = {0, 1, 2};
            t2.vertex_indices = {0, 2, 3};
            final_mesh.triangles = {t1, t2};
            final_marker.points = {p1, p2, p3, p1, p3, p4};

            RCLCPP_INFO(this->get_logger(), "Plane mesh initialized! Publishing...");
            RCLCPP_INFO(this->get_logger(), "Plane marker initialized! Publishing...");

        } else if (surface_type == "cylinder"){
            RCLCPP_INFO(this->get_logger(), "Fitting mode: CYLINDER");
            pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
            pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree = std::make_shared<pcl::search::KdTree<pcl::PointXYZRGB>>();
            pcl::PointCloud<pcl::Normal>::Ptr cloud_normals = std::make_shared<pcl::PointCloud<pcl::Normal>>();
            ne.setSearchMethod(tree);
            ne.setInputCloud(cropped_cloud);
            ne.setKSearch(50);
            ne.compute(*cloud_normals);

            pcl::ModelCoefficients::Ptr coefficients = std::make_shared<pcl::ModelCoefficients>();
            pcl::PointIndices::Ptr inliers = std::make_shared<pcl::PointIndices>();
            pcl::SACSegmentationFromNormals<pcl::PointXYZRGB, pcl::Normal> seg;
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_CYLINDER);
            seg.setMethodType(pcl::SAC_MSAC);
            seg.setNormalDistanceWeight(0.1);
            seg.setMaxIterations(10000);
            seg.setDistanceThreshold(0.01);
            seg.setInputCloud(cropped_cloud);
            seg.setInputNormals(cloud_normals);
            seg.segment(*inliers, *coefficients);

            if(inliers->indices.empty()){
                RCLCPP_ERROR(this->get_logger(), "Cylinder segmentation failed!");
                return;
            }

            Eigen::Vector3f axis_pt(coefficients->values[0], coefficients->values[1], coefficients->values[2]); 
            Eigen::Vector3f axis_dir(coefficients->values[3], coefficients->values[4], coefficients->values[5]);
            axis_dir.normalize();
            float radius = coefficients->values[6];

            RCLCPP_INFO(this->get_logger(), "Cylinder found");

            Eigen::Vector3f center_probe(crop_x_, crop_y_, crop_z_);
            float t = (center_probe - axis_pt).dot(axis_dir);
            Eigen::Vector3f closest_axis_pt = axis_pt + t * axis_dir;
            Eigen::Vector3f radial_vec = (center_probe - closest_axis_pt).normalized();
            Eigen::Vector3f tangent_vec = axis_dir.cross(radial_vec).normalized();
            float width = 1.0;
            float height = 1.0;
            int segments = 10;
            float angle_span = width / radius;
            
            int v_idx = 0;
            for (int i = 0; i <= segments; ++i){
                float current_angle = -angle_span/2.0 + (angle_span*i/segments);
                Eigen::Vector3f r_rotated = radial_vec * std::cos(current_angle) + tangent_vec * std::sin(current_angle);
                Eigen::Vector3f p_bottom = closest_axis_pt - (axis_dir * height) / 2.0 + (r_rotated * radius);
                Eigen::Vector3f p_top = closest_axis_pt + (axis_dir * height) / 2.0 + (r_rotated * radius);

                geometry_msgs::msg::Point pbottom, ptop;
                pbottom.x = p_bottom.x(); pbottom.y = p_bottom.y(); pbottom.z = p_bottom.z();
                ptop.x = p_top.x(); ptop.x = p_top.x(); ptop.z = p_top.z();

                final_mesh.vertices.push_back(pbottom);
                final_mesh.vertices.push_back(ptop);

                if (i > 0) {
                    shape_msgs::msg::MeshTriangle tri1, tri2;
                    tri1.vertex_indices = {static_cast<uint32_t>(v_idx - 2), static_cast<uint32_t>(v_idx - 1), static_cast<uint32_t>(v_idx)};
                    tri2.vertex_indices = {static_cast<uint32_t>(v_idx - 2), static_cast<uint32_t>(v_idx), static_cast<uint32_t>(v_idx + 1)};
                    final_mesh.triangles.push_back(tri1);
                    final_mesh.triangles.push_back(tri2);

                    final_marker.points.push_back(final_mesh.vertices[v_idx - 2]);
                    final_marker.points.push_back(final_mesh.vertices[v_idx - 1]);
                    final_marker.points.push_back(final_mesh.vertices[v_idx]);
                    
                    final_marker.points.push_back(final_mesh.vertices[v_idx - 2]);
                    final_marker.points.push_back(final_mesh.vertices[v_idx]);
                    final_marker.points.push_back(final_mesh.vertices[v_idx + 1]);
                }
                v_idx += 2;
            }
        } else {
            RCLCPP_ERROR(this->get_logger(), "Unknown surface_type parameter! Use 'plane' or 'cylinder'.");
            return;
        }
        
        // ROS publishing
        sensor_msgs::msg::PointCloud2 plane_cloud_msg;
        pcl::toROSMsg(*cropped_cloud, plane_cloud_msg);
        plane_cloud_msg.header = message->header;
        // cloud_publisher_->publish(plane_cloud_msg);
        wall_surface_mesh_publisher_->publish(final_mesh);
        wall_surface_marker_publisher_->publish(final_marker);
    }

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr wall_surface_marker_publisher_;
    rclcpp::Publisher<shape_msgs::msg::Mesh>::SharedPtr wall_surface_mesh_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_subscriber_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr record_wall_tf_server_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr scan_wall_plane_server_;
};
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<NoetherSurfaceFitting>(); 
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}