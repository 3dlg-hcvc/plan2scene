// Imported from neural texture: https://github.com/henzler/neuraltexture

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

const double PHI = 1.61803398874989484820459 * 00000.1;
const double PI = 3.14159265358979323846264 * 00000.1;
const double THETA = (3.14159265358979323846264 / 4.0) * 00000.1;
const double SQ2 = 1.41421356237309504880169 * 10000.0;


__device__ __forceinline__ int get_neighbour_offset(unsigned int i, unsigned int j) {
    int neighbour_offset = (i >> j) & 1;
    return neighbour_offset;
}

template <typename scalar_t>
__device__ __forceinline__ int d_floor(scalar_t a) {
    return 0.0;
}

// https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
template <typename scalar_t>
__device__ __forceinline__ scalar_t get_nearest_noise(
    torch::TensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> position,
    scalar_t __restrict__ seed,
    const int dim) {

    auto d = 0.0;

    for (int index_dim = 0; index_dim < dim; index_dim++) {
        auto a = PHI;

        if (index_dim == 1) {
            a = PI;
        }

        if (index_dim == 2) {
            a = THETA;
        }

        auto p = position[index_dim];
        auto p_floor = floor(p);
        auto b = p_floor * (seed + PHI) - a;
        d += b * b;
    }

    auto s = sqrt(d + 1.0e-8);
    auto t = tan(s) * SQ2;
    auto noise = t - floor(t);

    return noise;
}

// ######################### Forward #############################
template <typename scalar_t>
__device__ __forceinline__ scalar_t get_bilinear_noise(
    torch::TensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> position,
    scalar_t __restrict__ seed,
    const int dim) {

    scalar_t noise = 0;

    // calculate bilinear noise
    // reference to bilinear interpolation:
    // https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/interpolation/bilinear-filtering
    for(unsigned int j = 0; j < pow(2, dim); j++) {

        auto weight = 1.0;

        // calculate weights for interpolation
        for (unsigned int i = 0; i < dim; i++) {
            auto lambda = (position[i] - 0.5) - floor(position[i] - 0.5);
            auto offset = get_neighbour_offset(j,i);

            if (offset == 0) {
                weight = weight * (1 - lambda);
            }
            else {
                weight = weight * lambda;
            }
        }

        for(unsigned int p = 0; p < dim; p++) {
            auto offset = get_neighbour_offset(j,p);
            position[p] += offset - 0.5;
        }

        auto nearest_noise = get_nearest_noise(position, seed, dim);

        noise = noise + weight * nearest_noise;

        for(unsigned int q = 0; q < dim; q++) {
            auto offset = get_neighbour_offset(j,q);
            position[q] -=  offset - 0.5;
        }
    }

    return noise;
}

template <typename scalar_t>
__global__ void noise_cuda_forward_kernel(
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> position,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> nearest_noise,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> bilinear_noise,
    const int batch_size,
    const int dim,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> seed
  ) {

    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < batch_size) {

        auto current_position = position[index];
        auto current_seed = seed[index];

        nearest_noise[index] = get_nearest_noise(current_position, current_seed, dim);
        bilinear_noise[index] = get_bilinear_noise(current_position, current_seed, dim);
    }
}

torch::Tensor noise_cuda_forward(
    torch::Tensor position,
    torch::Tensor seed) {

    const auto batch_size = position.size(0);
    const int dim = position.size(1);

    auto options = torch::TensorOptions().dtype(position.type().scalarType()).device(torch::kCUDA);
    auto nearest_noise = torch::zeros({batch_size}, options);
    auto bilinear_noise = torch::zeros({batch_size}, options);

    const int threads = 512;
    const dim3 blocks((batch_size / threads)+1);

    AT_DISPATCH_FLOATING_TYPES(position.type(), "noise_cuda_forward_kernel", ([&] {
        noise_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            position.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            nearest_noise.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            bilinear_noise.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            batch_size,
            dim,
            seed.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>()
        );
    }));

    return torch::stack({nearest_noise, bilinear_noise}, 0);
}

// ######################### Backward #############################


template <typename scalar_t>
__global__ void noise_cuda_backward_kernel(
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> position,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> seed,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_position,
    const int batch_size,
    const int dim
  ) {

    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < batch_size) {

        auto current_d_position = d_position[index];
        auto current_position = position[index];
        auto current_seed = seed[index];

       for(unsigned int j = 0; j < pow(2, dim); j++) {

            scalar_t weight = 1.0;
            scalar_t d_weight[] = {1,1,1};

            for (unsigned int i = 0; i < dim; i++) {
                auto offset = get_neighbour_offset(j,i);

                auto lambda = (current_position[i] - 0.5) - floor(current_position[i] - 0.5);

                if (offset == 0) {
                    weight = weight * (1 - lambda);
                }
                else {
                    weight = weight * lambda;
                }

                // calculate gradients with respect to each dim
                for(unsigned int p = 0; p < dim; p++) {

                    auto pos = (current_position[p] - 0.5);

                    if (offset == 0) {
                        if (p == i) {
                            d_weight[p] *= -1 + d_floor(pos);
                        } else {
                            d_weight[p] *= 1 - (pos - floor(pos));
                        }

                    } else {
                        if (p != i) {
                            d_weight[p] *= pos - floor(pos);
                        }
                    }
                }
            }

            for(unsigned int p = 0; p < dim; p++) {
                auto offset = get_neighbour_offset(j,p);
                current_position[p] += offset - 0.5;
            }

            auto nearest_noise = get_nearest_noise(current_position, current_seed, dim);

            // gradients for nearest are always 0
            // product rule: (weight * nearest)` = weight * d_nearest + d_weight[i] * nearest
            for (unsigned int i = 0; i < dim; i++) {
                current_d_position[i] += d_weight[i] * nearest_noise;
            }

            for(unsigned int q = 0; q < dim; q++) {
                auto offset = get_neighbour_offset(j,q);
                current_position[q] -= offset - 0.5;
            }
        }
    }
}


torch::Tensor noise_cuda_backward(torch::Tensor position, torch::Tensor seed) {
    const auto batch_size = position.size(0);
    const int dim = position.size(1);

    const int threads = 512;
    const dim3 blocks((batch_size / threads)+1);

    auto d_position = torch::zeros_like(position);

    AT_DISPATCH_FLOATING_TYPES(d_position.type(), "noise_cuda_backward_kernel", ([&] {
        noise_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
            position.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            seed.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            d_position.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            batch_size,
            dim
        );
    }));

    return d_position;
}









