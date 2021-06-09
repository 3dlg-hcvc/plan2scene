// Imported from neural texture: https://github.com/henzler/neuraltexture
#include <torch/extension.h>
#include <vector>

// C++ interface
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// ################### CUDA forward declarations ###################
torch::Tensor noise_cuda_forward(torch::Tensor position,torch::Tensor seed);

torch::Tensor noise_forward(torch::Tensor position,torch::Tensor seed) {
    CHECK_INPUT(position);
    CHECK_INPUT(seed);

    return noise_cuda_forward(position, seed);
}


// ################### CUDA backward declarations ###################
torch::Tensor noise_cuda_backward(torch::Tensor position, torch::Tensor seed);

torch::Tensor noise_backward(torch::Tensor position, torch::Tensor seed) {
    CHECK_INPUT(position);
    CHECK_INPUT(seed);

    return noise_cuda_backward(position, seed);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &noise_forward, "Noise forward (CUDA)");
  m.def("backward", &noise_backward, "Noise backward (CUDA)");
}