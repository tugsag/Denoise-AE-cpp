#include <random>
#include <algorithm>
#include <torch/torch.h>
#include <data.h>
#include <model.h>


using namespace std;
namespace fs = filesystem;
torch::Device device(torch::kCUDA);


int main(){
    auto model = make_shared<ConvAE>();
    model->to(device);
    torch::optim::Adam optimizer(model->parameters());

    vector<tuple<string, string>> data = read_data();
    cout << "Total data size: " << data.size() << endl;
    // Need shuffle here? Prolly not tbh
    auto rng = default_random_engine {};
    shuffle(begin(data), end(data), rng);
    int test_split = (int)data.size() * 0.2;
    vector<tuple<string, string>> test = vector<tuple<string, string>>(data.begin(), data.begin() + test_split);
    vector<tuple<string, string>> train = vector<tuple<string, string>>(data.begin() + test_split, data.end());

    auto train_dataset = ReconDataset(train)
        .map(torch::data::transforms::Stack<>());
    auto test_dataset = ReconDataset(test)
        .map(torch::data::transforms::Stack<>());

    auto train_dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        train_dataset, torch::data::DataLoaderOptions().batch_size(8).workers(8)
    );
    auto test_dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        test_dataset, torch::data::DataLoaderOptions().batch_size(8).workers(8)
    );

    cout << fixed << setprecision(4);

    cout << "training" << endl;
    for(size_t epoch=1; epoch<=10; ++epoch){
        cout << "epoch: " << epoch << endl;
        float running_loss = 0;
        model->train();
        for(auto& batch: *train_dataloader){
            optimizer.zero_grad();
            torch::Tensor preds = model->forward(batch.data.to(device));
            torch::Tensor loss = torch::nn::functional::mse_loss(preds, batch.target.to(device));
            loss.backward();
            optimizer.step();
            running_loss += loss.item<float>();
        }
        cout << "epoch agg loss: " << running_loss << endl;
        cout << "epoch loss: " <<  running_loss / train_dataloader.size() << endl;
        
        cout << "validating" << endl;
        float valid_loss = 0;
        model->eval();
        for(auto& batch: *test_dataloader){
            torch::Tensor preds = model->forward(batch.data.to(device));
            torch::Tensor loss = torch::nn::functional::mse_loss(preds, batch.target.squeeze().to(device));
            valid_loss += loss.item<float>();
        }
        cout << "valid agg loss: " << valid_loss << endl;
        cout << "valid loss: " << valid_loss / test_dataloader.size() << endl;
    }

    return 0;
}
