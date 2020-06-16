#pragma once
//Created by Felix Ronchen
//Modified by Max Kolanz

#include <iostream> // std::cout
#include <cmath> // exp, sqrt, atan2
#include <vector> // std::vector
#include <numeric> // std::iota
#include <algorithm> // std::for_each
#include <iterator> // std::distance
#include <type_traits> // std::is_integral

#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>

#include "Constants.h"
#include <Windows.h>

// Useful reoccuring math/statistics functions
namespace math_utils {

	constexpr inline __host__ __device__ double PI(void) noexcept {
		return 3.1415926535897932385;
	}


	template<typename T>
	constexpr inline bool is_power_of_two(
		T number
	) noexcept {
		static_assert(std::is_integral<T>::value, "is_power_of_two requires integral argument type");
		return !(number == 0) && !(number & (number - 1));
	}


	template<typename T>
	inline bool is_even(
		T number
	) noexcept {
		static_assert(std::is_integral<T>::value, "is_even requires integral argument type");
		return !(number & 1);
	}


	// The mean and std_dev calculations always return double because
	// this is only what is needed here

	// Mean
	template<class InputIt>
	inline double mean(
		const InputIt first, const InputIt last
	) {
		static_assert(
			std::is_integral<typename std::iterator_traits<InputIt>::value_type>::value ||
			std::is_floating_point<typename std::iterator_traits<InputIt>::value_type>::value,
			"mean: Data type must be integral or floating point"
			);

		const size_t size = std::distance(first, last);
		if (!size) {
			throw std::length_error("first and last must be different");
		}

		return std::accumulate(first, last, 0.0) / size;
	}

	// std-dev for unknown mean, applies bessel correction
	//This correction is made to correct for the fact that these sample statistics tend 
	//to underestimate the actual parameters found in the population.
	//refers to (size - 1) instead of size
	template<class InputIt>
	inline double std_dev(
		const InputIt first, const InputIt last
	) {
		static_assert(
			std::is_integral<typename std::iterator_traits<InputIt>::value_type>::value ||
			std::is_floating_point<typename std::iterator_traits<InputIt>::value_type>::value,
			"std-dev: Data type must be integral or floating point"
			);
		double sum_of_squares = 0.0;
		double total = 0.0;
		const size_t size = std::distance(first, last);
		if (size < 2) {
			throw std::length_error("Array size must be larger than 1");
		}

		std::for_each(
			first, last, [&sum_of_squares, &total](const double d) {
				sum_of_squares += d * d;
				total += d;
			}
		);
		return std::sqrt(sum_of_squares / (size - 1) - std::pow(total, 2.0) / ((size - 1) * size));
	}


	//void gaussian_filter_parallel(
	//	double* arr2d,
	//	size_t N_x, size_t N_y,
	//	size_t gaussian_kernel_size,
	//	double sigma
	//);


	//__global__ void gaussian_filter_parallel_kernel(
	//	double* dst, const double* src,
	//	double* gaussian_kernel,
	//	size_t N_x, size_t N_y,
	//	int gaussian_kernel_size
	//);

	// Note that this is the usual definition
	// for gaussian beams sigma -> sigma / sqrt(2) and for intensities
	// sigma -> sigma / 2
	inline __host__ __device__ double gaussian(
		double x, double x_0, double sigma_x
	) noexcept {
		return exp(-(x - x_0) * (x - x_0) / (2.0 * sigma_x * sigma_x));
	}

	inline __host__ __device__ double gaussian2d(
		double x, double y,
		double x_0, double y_0,
		double sigma_x, double sigma_y
	) noexcept {

		return gaussian(x, x_0, sigma_x) * gaussian(y, y_0, sigma_y);
	}

	// Not the most efficient implementation but it works for now
	// I also wrote a parallel version but I have to test it first because things
	// have changed a lot
	template<typename T>
	void gaussian_filter(
		double* dst, const T* src,
		int width, int height,
		int gauss_kernel_size, double sigma
	) {
		std::vector<double> gauss_kernel(gauss_kernel_size * gauss_kernel_size);

		// Fill kernel
		double total = 0.0;
		const double center = (double(gauss_kernel_size) - 1.0) / 2.0;

		for (int i = 0; i < gauss_kernel_size; i++) {
			for (int j = 0; j < gauss_kernel_size; j++) {
				gauss_kernel[i * gauss_kernel_size + j] =
					gaussian2d(i, j, center, center, sigma, sigma);
				total += gauss_kernel[i * gauss_kernel_size + j];
			}
		}
		// Normalize
		for (auto& gaussian_weight : gauss_kernel) {
			gaussian_weight /= total;
		}

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {

				// Make sure it does not wrap
				int y_min = (std::max)(-gauss_kernel_size / 2, 0 - y);
				int x_min = (std::max)(-gauss_kernel_size / 2, 0 - x);

				int y_max = (std::min)(gauss_kernel_size / 2 + 1, (int)height - y);
				int x_max = (std::min)(gauss_kernel_size / 2 + 1, (int)width - x);

				double local_intensity = 0.0;

				// Declare it with auto here? I dont know if thats an performance issue
				auto gaussian_kernel_it = gauss_kernel.cbegin();
				double weight;

				for (int i_rel = y_min; i_rel < y_max; i_rel++) {
					for (int j_rel = x_min; j_rel < x_max; j_rel++) {

						weight = *gaussian_kernel_it;

						local_intensity += weight * src[(y + i_rel) * (int)width + (x + j_rel)];

						std::advance(gaussian_kernel_it, 1);
					}
				}
				dst[y * width + x] = local_intensity;
			}
		}
	}

	template <typename InputIt, typename T>
	std::vector<size_t> find_peaks2d(
		const InputIt first, const InputIt last,
		size_t width,
		T threshold
	) {
		T val = 0;
		std::vector<size_t> result;

		const size_t height = std::distance(first, last) / width;

		for (size_t i = 1; i < height - 1; i++) {
			for (size_t j = 1; j < width - 1; j++) {
				val = *std::next(first, i * width + j);
				if (
					// Horizontal/vertical terms
					(*std::next(first, (i - 1) * width + j) < val)
					&& (*std::next(first, i * width + j - 1) < val)
					&& (*std::next(first, (i + 1) * width + j) < val)
					&& (*std::next(first, i * width + j + 1) < val)

					// Diagonal terms
					&& (*std::next(first, (i - 1) * width + (j - 1)) < val)
					&& (*std::next(first, (i - 1) * width + (j + 1)) < val)
					&& (*std::next(first, (i + 1) * width + (j - 1)) < val)
					&& (*std::next(first, (i + 1) * width + (j + 1)) < val)
					&& (threshold < val)
					) {
					result.push_back(i * width + j);
				}
			}
		}
		return result;
	}


	inline __host__ __device__ double phase(
		const cufftDoubleComplex& z
	) {
		// Not sure if this is actually necessary
		// atan2 returns 0 for atan2(0,0); - Max
		/*if ((z.x == 0.0) && (z.y == 0.0)) {
			return 0;
		}*/

		return atan2(z.y, z.x);
	}


	// Modulus for positive and negative numbers, returns value in range [0, modulus)
	inline __host__ __device__ size_t mod(int val, size_t modulus) {
		const int result = val % modulus;
		return result < 0 ? size_t(result + modulus) : size_t(result);
	}


	inline __host__ __device__ byte rad_to_grayscale(
		double phase
	) noexcept {
		if (phase < 0) {
			phase = phase + 2.0 * PI();
		}
		// If input phase is more than 2pi it will just wrap
		const double temp = 256.0 * phase / (2.0 * PI());
		return byte(temp);
	}


	inline __host__ __device__ double grayscale_to_rad(
		byte phase
	) noexcept {
		const double result = 2.0 * PI() * (double)(phase) / 256.0;
		return result > PI() ? (result - 2.0 * PI()) : result;
	}


	inline __host__ __device__ double amplitude(
		const cufftDoubleComplex& z
	) noexcept {
		return sqrt(z.y * z.y + z.x * z.x);
	}


	// calculate the square of the absolute value
	inline __host__ __device__ double intensity(
		const cufftDoubleComplex& z
	) noexcept {
		return z.y * z.y + z.x * z.x;
	}


	// calculate the square of the absolute value
	inline __host__ __device__ double intensity(
		const double z
	) noexcept {
		return z * z;
	}

	inline __host__ __device__ bool is_in_circle(
		long long x, long long y,
		long long x_center, long long y_center,
		long long radius
	) noexcept {

		const long long x_rel = x - x_center;
		const long long y_rel = y - y_center;

		if (x_rel * x_rel + y_rel * y_rel <= radius * radius) {
			return true;
		}
		return false;
	}

	// Given a container of flattend 2d indices the elements are row wise
	// sorted by their x indices
	template <typename FlattenedIndexContainer>
	void sort_row_wise_by_x_coordinate(
		FlattenedIndexContainer first, FlattenedIndexContainer last, size_t width, size_t elements_per_row
	) {
		if (mod(std::distance(first, last), elements_per_row) != 0) {
			throw std::length_error(
				"sort_row_wise_by_x_coordinate:\
				Container size not evenly divisible by elements_per_row"
			);
		}

		// Lambda to compare x-coords
		auto sort_by_x = [width](size_t p1, size_t p2) {
			return p1 % width < p2% width;
		};

		// Sort each row
		for (auto it = first; it < last; std::advance(it, elements_per_row)) {
			std::sort(it, std::next(it, elements_per_row), sort_by_x);
		}
	}
}
