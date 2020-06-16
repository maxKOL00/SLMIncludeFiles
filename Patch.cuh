#pragma once

#include "device_launch_parameters.h"

class Patch {
    public:
        size_t x_0, y_0;
        size_t patch_size_x, patch_size_y;

        Patch();
        Patch(size_t x, size_t y);
        Patch(size_t x, size_t y, size_t patch_size);
        Patch(size_t x, size_t y, size_t patch_size_x, size_t patch_size_y);

        void move_to(size_t x_upper_left_new, size_t y_upper_left_new);
        void move_by(size_t x_shift, size_t y_shift);


        __device__ __host__ bool contains(size_t x, size_t y) const {
            return ((x_0 <= x) && (x < x_0 + patch_size_x)
                && (y_0 <= y) && (y < y_0 + patch_size_y));
        }

    private:
        void init_edges(void);

};
