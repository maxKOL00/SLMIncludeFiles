#pragma once

#include <iostream> // std::cout 
#include <fstream> //std::ifstream/std::ofstream
#include <vector> // std::vector
#include <string> // std::string
#include <sstream> // std::stringstream
#include <algorithm> // min, for_each
#include <functional> // function
#include <type_traits> // std::is_integral, std::is_floating_point
#include <iomanip> // std::setw, std::stprecision
#include <filesystem>

#include <Windows.h>

using byte = unsigned char;

namespace basic_fileIO {
	// Useful for debugging
	// With vector call begin()/cbegin() and end()/cend() with c-style array give pointer
	// to first element and pointer to first + array-size
	template<class InputIt>
	void save_one_column_data(
		const std::string& filename,
		const InputIt first, const InputIt last
	) {
		static_assert(
			std::is_integral_v<typename std::iterator_traits<InputIt>::value_type> ||
			std::is_floating_point_v<typename std::iterator_traits<InputIt>::value_type>,
			"print_data: Data type must be integral or floating point"
			);

		std::ofstream f_out(filename);
		f_out << std::setprecision(10);

		for (auto it = first; it != last; std::advance(it, 1)) {
			f_out << *it << "\n";
		}

		// Somehow this throws a C2088 sometimes?!
		//std::for_each(first, last, [&f_out](const auto& val){
		//	f_out << *val << "\n";
		//});
	}

	template<class InputIt1, class InputIt2>
	void save_two_column_data(
		const std::string& filename,
		const InputIt1 first_column_first, const InputIt1 first_column_last,
		const InputIt2 second_column_first
	) {

		static_assert(
			std::is_integral_v<typename std::iterator_traits<InputIt1>::value_type> ||
			std::is_floating_point_v<typename std::iterator_traits<InputIt1>::value_type> ||
			std::is_integral_v<typename std::iterator_traits<InputIt2>::value_type> ||
			std::is_floating_point_v<typename std::iterator_traits<InputIt2>::value_type>,
			"print_data: Data types must be integral or floating point"
			);

		// (std::min) instead of std::min 
		// to prevent collisions with min macro from Windows.h
		size_t size = std::distance(first_column_first, first_column_last);

		std::ofstream f_out(filename);
		f_out << std::setprecision(10);
		for (size_t i = 0; i < size; i++) {
			f_out << *std::next(first_column_first, i) << " " << *std::next(second_column_first, i) << "\n";
		}
	}

	void save_as_bmp(
		const std::string& filename, const byte* src,
		size_t width, size_t height
	);

	void save_as_bmp(
		const std::string& filename, const double* src,
		size_t width, size_t height
	);

	// This can be used to save bitmaps of arbitrary types
	// with a user defined function that gives a transformation rule
	// I used it to save cufftDoublecomplex phases/intensities
	// I made some changes needs testing first
	template <typename T>
	void save_as_bmp(
		const std::string& filename, const T* src,
		size_t width, size_t height, std::function<byte(T)> f
	) {
		const auto InputFirst = src;
		const auto InputLast = std::next(src, width * height - 1);
		std::vector<byte> result(width * height);
		std::transform(InputFirst, InputLast, result.begin(), f);

		save_as_bmp(filename, result.data(), width, height);
	}

	// Read bitmaps
	void read_from_bmp(
		const std::string& filename, byte* dst,
		size_t width, size_t height
	);

	// Load correction files
	void load_LUT(
		byte* lut_ptr,
		size_t lut_patch_num_x, size_t lut_patch_num_y
	);

	// I would like to pass it as CUDAArray2D but somehow this does not work right now
	// I think it has something to do with some main source files being cuda files
	void load_phase_correction(
		byte* phase_correction_ptr, unsigned int width, unsigned int height
	);

	std::string create_filepath(
		const std::string& filename, const std::string& folder
	);

}
