#pragma once

#include <string>
#include "cuda_utils.cuh"

template<typename T>
class CUDAArray {
public:
    CUDAArray(unsigned int size) : m_size(size) {
        if (cudaSuccess != cudaMallocManaged(&m_data, m_size * sizeof(T))) {
            throw std::runtime_error("Could not allocate memory for slm_image_ptr.");

        }
        if (cudaSuccess != cudaDeviceSynchronize()) {
            throw std::runtime_error("Could not allocate memory for slm_image_ptr.");

        }
    }

    CUDAArray<T>& operator= (const CUDAArray<T>& other) {
        if (m_size != other.size()) {
            throw std::out_of_range("Invalid size");
        }
        const auto other_begin = other.data();

        for (size_t i = 0; i < m_size; i++) {
            m_data[i] = *(other_begin + i);
        }

        return *this;
    }

    CUDAArray(const CUDAArray& rhs) {
        m_size = rhs.size();

        if (cudaSuccess != cudaMallocManaged(&m_data, m_size * sizeof(T))) {
            throw std::runtime_error("Could not allocate memory for slm_image_ptr.");

        }
        if (cudaSuccess != cudaDeviceSynchronize()) {
            throw std::runtime_error("Could not allocate memory for slm_image_ptr.");

        }

        const auto rhs_begin = rhs.data();

        for (size_t i = 0; i < m_size; i++) {
            m_data[i] = *(rhs_begin + i);
        }
    }

    ~CUDAArray() noexcept {
        cudaFree(m_data);
    }

    T* data() const noexcept {
        return m_data;
    }

    T& at(unsigned int i) {
        if (i < m_size) {
            return m_data[i];
        }
        else {
            throw std::out_of_range("Index " + std::to_string(i) << " out of range.\
                                     Size is " + std::to_string(m_size));
        }
    }

    const T& at(unsigned int i) const {
        if (i < m_size) {
            return m_data[i];
        }
        else{
            throw std::out_of_range("Index " + std::to_string(i) << " out of range.\
                                     Size is " + std::to_string(m_size));
        }
    }

    auto size(void) const noexcept {
        return m_size;
    }



    private:
        T* m_data;
        unsigned int m_size;
};


template<typename T>
class CUDAArray2D {
    public:
        CUDAArray2D(unsigned int width, unsigned int height) : m_width(width), m_height(height) {
            if (cudaSuccess != cudaMallocManaged(&m_data, m_width * m_height * sizeof(T))) {
                throw std::runtime_error("Could not allocate memory for slm_image_ptr.");
                
            }
            if (cudaSuccess != cudaDeviceSynchronize()) {
                throw std::runtime_error("Could not allocate memory for slm_image_ptr.");

            }
        }

        CUDAArray2D<T>& operator= (const CUDAArray2D<T>& other) {
            if (m_width != other.width()) {
                throw std::out_of_range("Widths not equal");
            }
            if (m_height != other.height())  {
                throw std::out_of_range("Heights not equal");
            }
            const auto other_begin = other.data();

            for (size_t i = 0; i < m_width * m_height; i++) {
                m_data[i] = *(other_begin + i);
            }

            return *this;
        }

        CUDAArray2D(const CUDAArray2D& rhs) {
            m_width = rhs.width();
            m_height = rhs.height();

            if (cudaSuccess != cudaMallocManaged(&m_data, m_width * m_height * sizeof(T))) {
                throw std::runtime_error("Could not allocate memory for slm_image_ptr.");
            }
            if (cudaSuccess != cudaDeviceSynchronize()) {
                throw std::runtime_error("Could not allocate memory for slm_image_ptr.");
            }

            const auto rhs_begin = rhs.data();

            for (size_t i = 0; i < m_width * m_height; i++) {
                m_data[i] = *(rhs_begin + i);
            }
        }

        ~CUDAArray2D () noexcept {
            cudaFree(m_data);
        }

        T* data() const noexcept {
            return m_data;
        }

        T& at(unsigned int y, unsigned int x) {
            this->check_if_x_is_in_bounds(x);
            this->check_if_y_is_in_bounds(y);

            return m_data[y * m_width + x];
        }

        const T& at(unsigned int y, unsigned int x) const {
            this->check_if_x_is_in_bounds(x);
            this->check_if_y_is_in_bounds(y);
            
            return m_data[y * m_width + x];
        }

        auto width(void) const noexcept {
            return m_width;
        }

        auto height(void) const noexcept {
            return m_height;
        }



    private:
        unsigned int m_width, m_height;
        T* m_data;

        void check_if_x_is_in_bounds(unsigned int x) const {
            if (x > m_width - 1) {
                throw std::out_of_range("Index " + std::to_string(x) + " out of range for a width of " + std::to_string(m_width));
            }
        }

        void check_if_y_is_in_bounds(unsigned int y) const {
            if (y > m_height - 1) {
                throw std::out_of_range("Index " + std::to_string(y) + " out of range for a height of " + std::to_string(m_height));
            }
        }
};
