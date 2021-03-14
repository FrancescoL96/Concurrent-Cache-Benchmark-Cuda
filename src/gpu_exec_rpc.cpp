#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/multi_array_dimension.hpp"
#include "conc_benchmark_cuda/srv/conc_array.hpp"

#include "conc_bench_utils.cuh"

using namespace std::chrono_literals;


float *d_matrix_device;

void gpu_sum(const std::shared_ptr<conc_benchmark_cuda::srv::ConcArray::Request> request,
    std::shared_ptr<conc_benchmark_cuda::srv::ConcArray::Response> response)
{
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), 
                "Request: {size: %i, sum_id: %i}", 
                request->arr.data.size(), request->sum_id);
    auto start = std::chrono::high_resolution_clock::now();

    auto shape = std_msgs::msg::MultiArrayDimension();
    shape.label = "N";
    shape.size = N;
    shape.stride = N;
    
    sum_gpu(request->arr.data, d_matrix_device, request->sum_id);

    response->arr.data = request->arr.data;
    response->arr.layout.dim.push_back(shape);
    response->arr.layout.data_offset = 0;

    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Duration: %.2f ms", 
                diff.count() * 1e-9);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    init_gpu(&d_matrix_device);
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), 
                "GPU service online: waiting for clients");

    std::shared_ptr<rclcpp::Node> gpu_server = rclcpp::Node::make_shared("gpu_server");

    rclcpp::Service<conc_benchmark_cuda::srv::ConcArray>::SharedPtr service = 
        gpu_server->create_service<conc_benchmark_cuda::srv::ConcArray>("gpu_sum", &gpu_sum);

    rclcpp::spin(gpu_server);
    rclcpp::shutdown();
}