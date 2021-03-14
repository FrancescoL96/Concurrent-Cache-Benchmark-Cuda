#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/multi_array_dimension.hpp"
#include "conc_benchmark_cuda/msg/conc_array.hpp"

#include "conc_bench_utils.cuh"

using namespace std::chrono_literals;


class MatrixSupplier : public rclcpp::Node
{
public:
    MatrixSupplier() : Node("matrix_supplier"), 
        current_run_(0), 
        current_sum_id_(0),
        cpu_node_run_(-1), 
        cpu_node_sum_id_(-1),
        gpu_node_run_(-1), 
        gpu_node_sum_id_(-1)
    {
        publisher_ = this->create_publisher<conc_benchmark_cuda::msg::ConcArray>(
            "random_matrix", 10);
        cpu_sum_subscription_ = this->create_subscription<conc_benchmark_cuda::msg::ConcArray>(
            "cpu_sum", 10, 
            std::bind(&MatrixSupplier::retrieve_cpu_sum_cb, this, std::placeholders::_1));
        gpu_sum_subscription_ = this->create_subscription<conc_benchmark_cuda::msg::ConcArray>(
            "gpu_sum", 10, 
            std::bind(&MatrixSupplier::retrieve_gpu_sum_cb, this, std::placeholders::_1));
        timer_ = this->create_wall_timer(
            5s, std::bind(&MatrixSupplier::publish_matrix_cb, this));

        current_matrix_.clear();
        matrix_supplier(current_matrix_);

        start = this->now();
    }

private:
    void publish_matrix_cb()
    {
        if (current_sum_id_ == SUMS) 
        {
            current_run_++;
            current_sum_id_ = 0;
        }

        auto message = conc_benchmark_cuda::msg::ConcArray();
        auto shape = std_msgs::msg::MultiArrayDimension();
        shape.label = "N";
        shape.size = N;
        shape.stride = N;

        if (current_run_ == RUNS)
        {
            current_run_ = 0;
            current_sum_id_ = 0;

            rclcpp::Time end = this->now();
            RCLCPP_INFO(this->get_logger(), "Final duration: %.2f ms", 
                    (end - start).nanoseconds() * 1e-6);

            start = this->now();
            current_matrix_.clear();
            matrix_supplier(current_matrix_);
        }

        message.arr.data = current_matrix_;
        message.arr.layout.dim.push_back(shape);
        message.arr.layout.data_offset = 0;
        message.run = current_run_;
        message.sum_id = current_sum_id_;

        RCLCPP_INFO(this->get_logger(), "Size: %i, Run: %i.", 
                    message.arr.data.size(), current_run_);
        publisher_->publish(message);
    }

    void check_cpu_gpu() 
    {
        if ((current_run_ != cpu_node_run_ || current_sum_id_ != cpu_node_sum_id_)
            || (current_run_ != gpu_node_run_ || current_sum_id_ != gpu_node_sum_id_))
        {
            return;
        }

        RCLCPP_INFO(this->get_logger(), 
                    "Check: {run: %i, sum_id: %i}", 
                    current_run_, current_sum_id_);
        current_sum_id_++;
        timer_->cancel();
        publish_matrix_cb();
    }

    void retrieve_cpu_sum_cb(const conc_benchmark_cuda::msg::ConcArray::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), 
                    "Received CPU sum: {size: %i, run: %i, sum_id: %i}", 
                    msg->arr.data.size(), msg->run, msg->sum_id);

        cpu_node_run_ = msg->run;
        cpu_node_sum_id_ = msg->sum_id;

        if (cpu_node_run_ != current_run_ || cpu_node_sum_id_ != current_sum_id_)
        {
            return;
        }

        if (current_sum_id_ % 2 != 0)
        {
            std::copy(msg->arr.data.begin() + (N / 2), msg->arr.data.end(),
                      current_matrix_.begin() + (N / 2));
        }
        else 
        {
            std::copy(msg->arr.data.begin(), msg->arr.data.end() - (N / 2),
                      current_matrix_.begin());
        }
        check_cpu_gpu();
    }

    void retrieve_gpu_sum_cb(const conc_benchmark_cuda::msg::ConcArray::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), 
                    "Received GPU sum: {size: %i, run: %i, sum_id: %i}", 
                    msg->arr.data.size(), msg->run, msg->sum_id);

        gpu_node_run_ = msg->run;
        gpu_node_sum_id_ = msg->sum_id;

        if (gpu_node_run_ != current_run_ || gpu_node_sum_id_ != current_sum_id_)
        {
            return;
        }

        if (current_sum_id_ % 2 != 0)
        {
            std::copy(msg->arr.data.begin(), msg->arr.data.end() - (N / 2),
                      current_matrix_.begin());
        }
        else 
        {
            std::copy(msg->arr.data.begin() + (N / 2), msg->arr.data.end(),
                      current_matrix_.begin() + (N / 2));
        }
        check_cpu_gpu();
    }

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<conc_benchmark_cuda::msg::ConcArray>::SharedPtr publisher_;
    int current_run_;
    int current_sum_id_;

    int cpu_node_run_;
    int cpu_node_sum_id_;
    int gpu_node_run_;
    int gpu_node_sum_id_;

    std::vector<float> current_matrix_;

    rclcpp::Subscription<conc_benchmark_cuda::msg::ConcArray>::SharedPtr cpu_sum_subscription_;
    rclcpp::Subscription<conc_benchmark_cuda::msg::ConcArray>::SharedPtr gpu_sum_subscription_;

    rclcpp::Time start;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MatrixSupplier>());
    rclcpp::shutdown();
    return 0;
}