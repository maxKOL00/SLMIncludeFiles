#pragma once
//Created by Felix Ronchen

#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>

// Because the target array is very sparsely populated compared to the overall
// number of sites, only the nonzero sites are tracked
// z is currently not used (2D only)
struct Point {
    unsigned int x;
    unsigned int y;
    unsigned int z;
};


struct Tweezer {
    Point position_in_fft_array;
    Point position_in_camera_image;
    double target_intensity;
    double current_intensity;
    double current_phase;
};
