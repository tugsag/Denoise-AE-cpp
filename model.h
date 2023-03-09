#include <torch/torch.h>

using namespace std;


struct ConvAE : public torch::nn::Module{
    ConvNet(){
        // Encoder
        c1 = register_module<ConvL>("c1", make_shared<ConvL>(3, 32, 7, 1, true));
        cstride1 = register_module<ConvL>("cstride1", make_shared<ConvL>(32, 64, 5, 2, true));
        r1 = register_module<Residual>("r1", make_shared<Residual>(64, 64, 3));
        r2 = register_module<Residual>("r2", make_shared<Residual>(64, 64, 3));
        r3 = register_module<Residual>("r3", make_shared<Residual>(64, 64, 3));
        r4 = register_module<Residual>("r4", make_shared<Residual>(64, 64, 3));
        cstride2 = register_module<ConvL>("cstride2", make_shared<ConvL>(64, 128, 3, 2, true));
        c2 = register_module<ConvL>("c2", make_shared<ConvL>(128, 64, 3, 1, true));
        // Decoder
        dc1 = register_module("dc1", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64, 128, 3)
            .stride(1)));
        dcstride1 = register_module("dcstride1", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(128, 64, 3)
            .stride(2)));
        dcstride2 = register_module("dcstride2", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64, 64, 3)
            .stride(2)
            .output_padding(1)));
        dc2 = register_module("dc2", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64, 32, 3)
            .stride(1)));
        dc3 = register_module("dc3", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(32, 3, 7)
            .stride(1)));
    }

    torch::Tensor forward(torch::Tensor x){
        // Encoder
        e = c1->forward(x);
        e = cstride1->forward(e);
        e = r1->forward(e);
        e = r2->forward(e);
        e = r3->forward(e);
        e = r4->forward(e);
        e = cstride2->forward(e);
        e = c2->forward(e);
        // Decoder
        d = dc1->forward(e);
        d = dcstride1->forward(d);
        d = dcstride2->forward(d);
        d = dc2->forward(d);
        d = dc3->forward(d);
        return d;
    }

    torch::nn::ConvTranspose2d dc1{nullptr}, dcstride1{nullptr}, dcstride2{nullptr}, dc2{nullptr}, dc3{nullptr};
    Residual r1, r2, r3, r4;
    ConvL c1, cstride1, cstride2, c2;
};

struct Residual: public torch::nn::Module{
    Residual(int64_t inch, int64_t outch, int64_t kernel, int64_t stride, int64_t pad){
        c1 = register_module("c1", torch::nn::Conv2d(torch::nn::Conv2dOptions(inch, outch, kernel)
            .stride(stride)
            .padding(pad)));
        c2 = register_module("c2", torch::nn::Conv2d(torch::nn::Conv2dOptions(outch, outch, kernel)
            .stride(stride)
            .padding(pad)));
        b1 = register_module("b1", torch::nn::BatchNorm2d(outch));
        b2 = register_module("b2", torch::nn::BatchNorm2d(outch));
    }

    torch::Tensor forward(torch::Tensor x){
        auto r = c1->forward(x);
        r = torch::nn::functional::leaky_relu(b1->forward(r));
        r = c2->forward(r);
        r = torch::nn::functional::leaky_relu(b2->forward(r));

        return x + r;
    }


    torch::nn::Conv2d c1{nullptr}, c2{nullptr};
    torch::nn::BatchNorm2d b1{nullptr}, b1{nullptr};
    int64_t inch, outch, kernel;
    int64_t stride = 1;
    int64_t pad = 1;
}

struct ConvL: public torch::nn:Module{
    ConvL(int64_t inch, int64_t outch, int64_t kernel, int64_t stride, bool batch){
        c1 = register_module("c1", torch::nn::Conv2d(torch::nn::Conv2dOptions(inch, outch, kernel).stride(stride)));
        b1 = register_module("b1", torch::nn::BatchNorm2d(outch));
    }

    torch::Tensor forward(torch::Tensor x){
        x = torch::nn::functional::leaky_relu(c1->forward(x));
        if(batch){
            x = b1->forward(x)
        }

        return x;
    }

    
    torch::nn::Conv2d c1{nullptr};
    torch::nn::BatchNorm2d b1{nullptr};
    int64_t inch, outch, kernel, stride;
    bool batch;
}