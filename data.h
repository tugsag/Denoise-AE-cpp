#include <vector>
#include <tuple>
#include <filesystem>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <regex>

using namespace std;
namespace fs = filesystem;

vector<tuple<string, string>> read_data(){
    // Simply gather all pngs in chunked, assuming gt images have also been chunked
    vector<tuple<string, string>> data;
    string path = "chunked";
    regex iso_re("ISO3200|ISO6400|ISOH1|ISO1600|ISOH2|ISO5000|
                    ISOH3|ISO4000|ISO2500|ISO800|ISO400")
    for (const auto & entry: fs::directory_iterator(path)){
        fs::path subpath = entry.path();
        string noisy_im{sp.string()};
        string clean_im = regex_replace(noisy_im, iso_re, "ISO200");
        tuple<string, string> paths = {noisy_im, clean_im};
        data.push_back(paths);
    }
    
    return data;    
}

class ReconDataset: public torch::data::Dataset<ReconDataset>{
    private:
        vector<tuple<string, string>> data;

    public:
        ReconDataset(vector<tuple<string, string>> d){
            data = d;
        }

        torch::data::Example<> get(size_t index) override{
            string noisy_path = std::get<0>(data[index]);
            string clean_path = std::get<1>(data[index]);
            // Read and normalize images
            cv::Mat nim_raw = cv::imread(noisy_path);
            cv::Mat cim_raw = cv::imread(clean_path);
            cv::Mat nim;
            cv::Mat cim;
            nim_raw.convertTo(nim, -1, 1.0/255, 0);
            cim_raw.convertTo(cim, -1, 1.0/255, 0);
            // Tensorize ims
            torch::Tensor ntensor = torch::from_blob(nim.data, {nim.rows, nim.cols, nim.channels()}, torch::kByte).clone();
            torch::Tensor ctensor = torch::from_blob(cim.data, {cim.rows, cim.cols, cim.channels()}, torch::kByte).clone();
            ntensor = ntensor.permute({2, 0, 1});
            ctensor = ctensor.permute({2, 0, 1});
            ntensor = ntensor.to(torch::kFloat32)/255;
            ctensor = ctensor.to(torch::kFloat32)/255;

            return {ntensor, ctensor}
        }

        torch::optional<size_t> size() const override{
            return data.size();
        }
}
