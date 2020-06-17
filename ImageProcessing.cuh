#pragma once
#include "Parameters.h"
#include "math_utils.cuh"
#include "cuda_utils.cuh"
#include "basic_fileIO.h"

#include <iostream>
#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>

struct Tweezer;

class ImageProcessing {

	public:
		explicit ImageProcessing(const Parameters& params);

		std::vector<unsigned int> create_mask(
			const byte* image_data,
			unsigned int width, unsigned int height,
			unsigned int num_peaks_x, unsigned int num_peaks_y
		) const;


		void correct_image(
			byte* slm_image_ptr,
			const byte* phase_correction_ptr,
			const byte* lut_ptr
		) const;


		void expand_to_sensor_size(
			byte* dst,
			const byte* src
		) const;

		// Add a circular blazed grating to an array of size slm_px_x * slm_px_y
		// offset from the left by horizontal_offset pixels
		void add_blazed_grating(
			byte* dst
		) const;


		void fresnel_lens(
			byte* arr, size_t width, size_t height, double delta_z_um
		) const;


		void shift_fourier_image(
			byte* phasemap, 
			double shift_x_um, double shift_y_um
		) const noexcept;


		template <typename T>
		void invert_camera_image(
			T* image, size_t width, size_t height
		) const noexcept {
			T temp;
			for (size_t i = 0; i < height; i++) {
				for (size_t j = 0; j < width / 2; j++) {
					temp = image[i * width + j];

					image[i * width + j] = image[(height - 1 - i) * width + (width - 1 - j)];
					image[(height - 1 - i) * width + (width - 1 - j)] = temp;
				}
			}
		}

		// A Bitmap type would come in handy, I will change that soon
		std::tuple<std::vector<byte>, size_t, size_t> crop_tweezer_array_image(
			const byte* image_data,
			size_t width, size_t height,
			size_t num_peaks_x, size_t num_peaks_y
		) const;


	private:
		// General parameters
		size_t slm_px_x, slm_px_y;
		double pixel_size_um;
		double wavelength_um, focal_length_um;
		size_t number_of_pixels_unpadded;

		// Correction and grating generation
		size_t lut_patch_size_x, lut_patch_size_y;
		size_t lut_patch_num_x, lut_patch_num_y;

		size_t horizontal_offset;

		size_t blazed_grating_period_px;
		byte blazed_grating_max;

		// unsigned int to avoid compiler warnings
		unsigned int block_size, num_blocks_slm;
};

