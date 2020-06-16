#pragma once

#include <vector>
#include <stdexcept> // For runtime error

#include "Windows.h" // For byte
#include "basic_fileIO.h"

struct Bitmap {
    Bitmap(const std::vector<byte>& pixel_data, size_t width, size_t height);
    Bitmap(const byte* pixel_data, size_t width, size_t height);
    Bitmap(const std::string& filename);

    void save(const std::string& filename);

    size_t width;
    size_t height;
    std::vector<byte> pixel_data;
};
