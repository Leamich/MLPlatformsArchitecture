#include <torch/torch.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

std::string mnist_path() { return DATA_DIR; }

int main() {
    auto dataset = torch::data::datasets::MNIST(mnist_path())
                       .map(torch::data::transforms::Stack<>());
    auto loader = torch::data::make_data_loader(
        std::move(dataset), torch::data::DataLoaderOptions().batch_size(64));

    int64_t count = 0;
    double sum = 0.0;
    float gmin = 1e9f, gmax = -1e9f;
    size_t nb = 0;
    for (auto& batch : *loader) {
        auto d = batch.data;
        gmin = std::min(gmin, d.min().item<float>());
        gmax = std::max(gmax, d.max().item<float>());
        sum += d.mean().item<double>() * static_cast<double>(d.numel());
        count += d.numel();
        if (nb == 0) {
            std::cout << "first batch: data.sizes() = " << d.sizes()
                      << "  targets.sizes() = " << batch.target.sizes() << "\n";
        }
        ++nb;
    }
    std::cout << "#batches = " << nb << "  #samples = " << count << "\n";
    std::cout << "global pixel: min = " << gmin << "  max = " << gmax
              << "  mean = " << (sum / count) << "\n";
    return 0;
}