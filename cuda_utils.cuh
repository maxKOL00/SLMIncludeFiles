#pragma once
//Created by Felix Ronchen
#include "math_utils.cuh"

#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <math.h>
// Add noexcept and maybe inline some things

#include "basic_fileIO.h"
#include "errorMessage.h"

namespace cuda_utils {

    inline void cuda_synchronize(const std::string& file, int line) {
        if (cudaSuccess != cudaDeviceSynchronize()) {
            errBox("cuda synchronize error in file", __FILE__, __LINE__);
            throw std::runtime_error(std::string("cuda synchronization error in file ") + file + " line " + std::to_string(line));
        }
    }
    


    __global__ void pad_array(
        cufftDoubleComplex* __restrict padded_array,
        const cufftDoubleComplex* __restrict unpadded_array,
        unsigned int N_padded, unsigned int N
    );

    __global__ void unpad_array(
        const cufftDoubleComplex* __restrict padded_array,
        cufftDoubleComplex* __restrict unpadded_array,
        unsigned int N_padded, unsigned int N
    );

    __global__ void fft_shift(
        cufftDoubleComplex* __restrict arr_shifted,
        const cufftDoubleComplex* __restrict arr_unshifted,
        unsigned int N_x, unsigned int N_y        
    );

    __global__ void multiply_by_quadratic_phase_factor(
        cufftDoubleComplex* dst,
        unsigned int number_of_pixels_padded,
        double c
    );


    __global__ void shifted_intensity_distribution(
        double* __restrict dst,
        const cufftDoubleComplex* __restrict src,
        unsigned int N_x, unsigned int N_y
    );

    __global__ void reset_fft_array(
        cufftDoubleComplex* dst,
        double val
    );

    __global__ void set_phase_only_array(
        cufftDoubleComplex* __restrict dst,
        const double* __restrict phasemap
    );

    // If overwrite is true the content in dst will be overwritten
    // otherwise it will be added
    __global__ void extract_phasemap(
        double* __restrict dst,
        const cufftDoubleComplex* __restrict src,
        bool overwrite// = false
    );

    __global__ void scale_array(
        double* dst,
        double scale_factor
    );

    __global__ void scale_array(
        cufftDoubleComplex* dst,
        double scale_factor
    );

    __global__ void add_phase(
        cufftDoubleComplex* __restrict dst,
        const double* __restrict src,
        unsigned int number_of_pixels_padded
    );

    double get_norm(
        const cufftDoubleComplex* src,
        unsigned int size
    );

    __global__ void simulate_two_FFTs_in_a_row(
        cufftDoubleComplex* __restrict dst,
        const cufftDoubleComplex* __restrict src,
        unsigned int width, unsigned int height
    );


    void save_phasemap(
        const std::string& filename,
        const cufftDoubleComplex* arr,
        unsigned int width, unsigned int height
    );
}
