#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/multi_array_dimension.hpp"
#include "conc_benchmark_cuda/msg/conc_array.hpp"

#include "conc_bench_utils.cuh"

using namespace std::chrono_literals;


class CPUExec : public rclcpp::Node
{
public:
    CPUExec() : Node("cpu_exec"), 
        current_run_(-1), 
        current_sum_id_(-1), 
        current_matrix_()
    {
        subscription_ = this->create_subscription<conc_benchmark_cuda::msg::ConcArray>(
            "random_matrix", 10, 
            std::bind(&CPUExec::retrieve_matrix_cb, this, std::placeholders::_1));
        publisher_ = this->create_publisher<conc_benchmark_cuda::msg::ConcArray>(
            "cpu_sum", 10);
        publish_timer_ = this->create_wall_timer(
            5s, std::bind(&CPUExec::publish_matrix_cb, this));
    }

private:
    void publish_matrix_cb()
    {
        if (current_run_ == -1 || current_sum_id_ == -1)
        {
            RCLCPP_INFO(this->get_logger(), "Wait for matrix_supplier");
            return;
        }

        rclcpp::Time start = this->now();

        auto message = conc_benchmark_cuda::msg::ConcArray();
        auto shape = std_msgs::msg::MultiArrayDimension();
        shape.label = "N";
        shape.size = N;
        shape.stride = N;
        message.arr.layout.dim.push_back(shape);
        message.arr.layout.data_offset = 0;
        message.arr.data.clear();
        sum_cpu(current_matrix_, current_sum_id_);
        message.run = current_run_;
        message.sum_id = current_sum_id_;
        message.arr.data = current_matrix_;

        publisher_->publish(message);

        rclcpp::Time end = this->now();
        RCLCPP_INFO(this->get_logger(), "Run: %i, SumID: %i, Duration: %.2f ms", 
                    current_run_, current_sum_id_, (end - start).nanoseconds() * 1e-6);
    }

    void retrieve_matrix_cb(const conc_benchmark_cuda::msg::ConcArray::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), 
                    "Received matrix: {size: %i, run: %i, sum_id: %i}", 
                    msg->arr.data.size(), msg->run, msg->sum_id);

        current_matrix_ = msg->arr.data;
        current_run_ = msg->run;
        current_sum_id_ = msg->sum_id;

        publish_timer_->cancel();
        publish_matrix_cb();
    }

    rclcpp::TimerBase::SharedPtr publish_timer_;
    rclcpp::Publisher<conc_benchmark_cuda::msg::ConcArray>::SharedPtr publisher_;
    rclcpp::Subscription<conc_benchmark_cuda::msg::ConcArray>::SharedPtr subscription_;
    int current_run_;
    int current_sum_id_;
    std::vector<float> current_matrix_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CPUExec>());
    rclcpp::shutdown();
    return 0;
}