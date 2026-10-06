#include <torch/torch.h>

struct Net : torch::nn::Module {
    Net() {
        fc1 = register_module("fc1", torch::nn::Linear(784, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 32));
        fc3 = register_module("fc3", torch::nn::Linear(32, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
        x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
        x = torch::relu(fc2->forward(x));
        x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
        return x;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

int main() {
    auto net = std::make_shared<Net>();

    auto data_loader = torch::data::make_data_loader(
        torch::data::datasets::MNIST(DATA_DIR).map(
            torch::data::transforms::Stack<>()),
        /*batch_size=*/64);

    torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

    for (size_t epoch = 1; epoch <= 2; ++epoch) {
        size_t batch_index = 0;
        for (auto& batch : *data_loader) {
            optimizer.zero_grad();
            torch::Tensor prediction = net->forward(batch.data);
            torch::Tensor loss = torch::nll_loss(prediction, batch.target);
            loss.backward();
            optimizer.step();
            if (++batch_index % 100 == 0) {
                torch::Tensor pred_cls = torch::argmax(prediction, /*dim=*/1);
                assert(pred_cls.size(0) == batch.target.size(0));

                float acc = (pred_cls == batch.target)
                                .to(torch::ScalarType::Float)
                                .mean()
                                .item<float>();

                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                          << " | Loss: " << loss.item<float>()
                          << " | Acc: " << acc << std::endl;
                torch::save(net, "net.pt");
            }
        }
    }

    // Reload and verify predictions are unchanged.
    auto net2 = std::make_shared<Net>();
    torch::load(net2, "net.pt");
    net2->eval();
    net->eval();

    int cnt_batches = 3;
    int mistakes = 0;
    for (auto& batch : *data_loader) {
        torch::Tensor prediction = torch::argmax(net->forward(batch.data), 1);
        torch::Tensor prediction2 = torch::argmax(net2->forward(batch.data), 1);

        mistakes += (prediction != prediction2)
                        .to(torch::ScalarType::Int)
                        .sum()
                        .item<int>();

        if (--cnt_batches == 0) break;
    }

    std::cout << "Mistakes (net != net2): " << mistakes << std::endl;
}