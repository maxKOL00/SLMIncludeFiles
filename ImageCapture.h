#pragma once
#include <iostream>

#include "Parameters.h"
#include "VimbaCPP/Include/VimbaCPP.h"
#include "statusBox.h"

struct ImageCaptureException : public std::runtime_error {
	std::string msg;
	ImageCaptureException(const char* message) throw() : std::runtime_error(message) {
		msg = std::string(message);
	}
	const char* what() const throw() {
		return msg.c_str();
	}
};


class ImageCapture {

	public:
		// Captures an image with the camera and writes the pixel data into a 
		// container of size pixel_data + width * height
		void capture_image(
			byte* pixel_data, size_t width, size_t height
		) const;

		// Binary search to adjust the exposure time such that 
		// no more than <max_counts> pixels have a value of <pixel_value> or more
		void adjust_exposure_time_automatically(
			byte pixel_value, size_t max_counts, statusBox *box
		);
		
		// Set exposure time manually
		void set_exposure_time(
			double time_us
		);
		
		// Get the exposure time
		double get_exposure_time_us(
			void
		) const ;


		explicit ImageCapture(
			const Parameters& params
		);

		~ImageCapture() noexcept;

	private:
		size_t camera_px_h, camera_px_v;


		std::string ID;
		AVT::VmbAPI::CameraPtr camera_ptr;
		AVT::VmbAPI::VimbaSystem& system = AVT::VmbAPI::VimbaSystem::GetInstance();
		
		mutable AVT::VmbAPI::FeaturePtr exposure_time_ptr;
		
		mutable AVT::VmbAPI::FramePtr frame_ptr;

		//A raw c array is needed because vimba is not as modern as it thinks it is
		//Edit: it's shit
		//Edit2: I did some tests with test arrays that do nothing except for being
		//		 allocated and deleted in ctor/dtor which worked. I think the API
		//		 assigns a new pointer to point to the same memory location as buf
		//		 and calls delete[] when it is cleaned. Therefore a manual deletion
		//		 is not only not necessary but will actually crash as it tries to free
		//		 the same memory twice.

		// I made this mutable because it is more of a helper variable to work around the api.
		// Its value(s) are not important because they are immediately copied to the
		// container given by the caller. With mutable capture_image() can be const.
		mutable byte* buf;

		double min_timeout_from_frame_rate_ms;

		// Pass as first last so function can be noexcept because no range checking is needed
		size_t number_of_pixels_with_value_or_more_in_image(
			const byte* first, const byte* last, byte value
		) const noexcept;

};
