# Imported from neural texture: https://github.com/henzler/neuraltexture
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='noise_cuda',
    ext_modules=[
        CUDAExtension('noise_cuda', [
            'noise_cuda.cpp',
            'noise_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
