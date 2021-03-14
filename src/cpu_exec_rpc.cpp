#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/multi_array_dimension.hpp"
#include "conc_benchmark_cuda/srv/conc_array.hpp"

#include "conc_bench_utils.cuh"

using namespace std::chrono_literals;


void cpu_sum(const std::shared_ptr<conc_benchmark_cuda::srv::ConcArray::Request> request,
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
    
    sum_cpu(request->arr.data, request->sum_id);

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
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), 
                "CPU service online: waiting for clients");

    std::shared_ptr<rclcpp::Node> cpu_server = rclcpp::Node::make_shared("cpu_server");

    rclcpp::Service<conc_benchmark_cuda::srv::ConcArray>::SharedPtr service = 
        cpu_server->create_service<conc_benchmark_cuda::srv::ConcArray>("cpu_sum", &cpu_sum);

    rclcpp::spin(cpu_server);
    rclcpp::shutdown();
}