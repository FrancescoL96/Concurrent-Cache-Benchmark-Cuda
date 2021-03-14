#include <chrono>
#include <cstdlib>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/multi_array_dimension.hpp"
#include "conc_benchmark_cuda/srv/conc_array.hpp"

#include "conc_bench_utils.cuh"

using namespace std::chrono_literals;


int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("matrix_supplier_client");
    rclcpp::Client<conc_benchmark_cuda::srv::ConcArray>::SharedPtr cpu_client =
        node->create_client<conc_benchmark_cuda::srv::ConcArray>("cpu_sum");
    rclcpp::Client<conc_benchmark_cuda::srv::ConcArray>::SharedPtr gpu_client =
        node->create_client<conc_benchmark_cuda::srv::ConcArray>("gpu_sum");

    auto request = std::make_shared<conc_benchmark_cuda::srv::ConcArray::Request>();

    while (!cpu_client->wait_for_service(1s) || !gpu_client->wait_for_service(1s))
    {
        if (!rclcpp::ok())
        {
            RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Interrupted while waiting for the service. Exiting.");
            return 0;
        }
        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Waiting for CPU and GPU services");
    }

    std::vector<float> m;

    auto start = std::chrono::high_resolution_clock::now();
    for (int run = 0; run < RUNS; run++)
    {
        m.clear();
        matrix_supplier(m);

        for (int sum_id = 0; sum_id < SUMS; sum_id++)
        {
            RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Loop: {run: %i, sum_id: %i}", 
                        run, sum_id);
            
            auto shape = std_msgs::msg::MultiArrayDimension();
            shape.label = "N";
            shape.size = N;
            shape.stride = N;

            request->arr.data = m;
            request->arr.layout.dim.push_back(shape);
            request->arr.layout.data_offset = 0;
            request->sum_id = sum_id;
            auto cpu_result = cpu_client->async_send_request(request);
            auto gpu_result = gpu_client->async_send_request(request);

            // Wait for the result.
            if (rclcpp::spin_until_future_complete(node, cpu_result) !=
                    rclcpp::executor::FutureReturnCode::SUCCESS 
                || rclcpp::spin_until_future_complete(node, gpu_result) !=
                    rclcpp::executor::FutureReturnCode::SUCCESS)
            {
                RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), 
                             "Failed to call CPU and GPU services");
                break;
            }

            RCLCPP_INFO(rclcpp::get_logger("rclcpp"), 
                    "Received CPU sum: {size: %i}", 
                    cpu_result.get()->arr.data.size());
            RCLCPP_INFO(rclcpp::get_logger("rclcpp"), 
                    "Received GPU sum: {size: %i}", 
                    gpu_result.get()->arr.data.size());

            if (sum_id % 2 != 0)
            {
                std::copy(gpu_result.get()->arr.data.begin(), 
                          gpu_result.get()->arr.data.end() - (N / 2),
                          m.begin());
                std::copy(cpu_result.get()->arr.data.begin() + (N / 2), 
                          cpu_result.get()->arr.data.end(), 
                          m.begin() + (N / 2));
            }
            else
            {
                std::copy(cpu_result.get()->arr.data.begin(), 
                          cpu_result.get()->arr.data.end() - (N / 2),
                          m.begin());
                std::copy(gpu_result.get()->arr.data.begin() + (N / 2), 
                          gpu_result.get()->arr.data.end(), 
                          m.begin() + (N / 2));
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Duration: %.2f ms", 
                diff.count() * 1e-9);
    rclcpp::shutdown();
    return 0;
}