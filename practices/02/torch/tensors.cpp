#include <torch/torch.h>

#include <iostream>

int main() {
    auto a = torch::randn({3, 4});
    auto b = torch::arange(12).reshape({3, 4}).to(torch::kFloat);
    auto c = a + b;

    std::cout << "a.sizes() = " << a.sizes() << "  dtype = " << a.dtype()
              << "  device = " << a.device() << "\n";
    std::cout << "b =\n" << b << "\n";
    std::cout << "c[0][0] = " << c[0][0].item<float>() << "\n";

    auto d = torch::randn({4, 2});
    auto mm = a.mm(d);
    std::cout << "a.mm(d).sizes() = " << mm.sizes() << "\n";

    std::cout << "a.transpose(0,1).sizes() = " << a.transpose(0, 1).sizes()
              << "\n";
    std::cout << "b.mean() = " << b.mean().item<float>() << "\n";
    return 0;
}