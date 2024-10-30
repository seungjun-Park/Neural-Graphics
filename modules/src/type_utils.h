#pragma once

#include <c10/core/ScalarType.h>
#include <cuda_runtime.h>
#include <type_traits>

// to support half precision cuda version

template<typename T>
struct type_mapper
{
	using type = T;
};

#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
#include <cuda_fp16.h>
template<>
struct type_mapper<c10::Half>
{
	using type = half;
};

#endif

#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
#include <cuda_bf16.h>
template<>
struct type_mapper<c10::BFloat16>
{
	using type = nv_bfloat16;
};
#endif

template<typename T>
using mapped_type = typename type_mapper<T>::type;

// Addtional half arithmetic

namespace c10
{
	// uint64_t	
	inline C10_HOST_DEVICE Half operator+(Half a, uint64_t b) {
		return a + static_cast<Half>(b);
	}
	inline C10_HOST_DEVICE Half operator-(Half a, uint64_t b) {
		return a - static_cast<Half>(b);
	}
	inline C10_HOST_DEVICE Half operator*(Half a, uint64_t b) {
		return a * static_cast<Half>(b);
	}
	inline C10_HOST_DEVICE Half operator/(Half a, uint64_t b) {
		return a / static_cast<Half>(b);
	}

	inline C10_HOST_DEVICE Half operator+(uint64_t a, Half b) {
		return static_cast<Half>(a) + b;
	}
	inline C10_HOST_DEVICE Half operator-(uint64_t a, Half b) {
		return static_cast<Half>(a) - b;
	}
	inline C10_HOST_DEVICE Half operator*(uint64_t a, Half b) {
		return static_cast<Half>(a) * b;
	}
	inline C10_HOST_DEVICE Half operator/(uint64_t a, Half b) {
		return static_cast<Half>(a) / b;
	}

	// uint32_t

	inline C10_HOST_DEVICE Half operator+(Half a, uint32_t b) {
		return a + static_cast<Half>(b);
	}
	inline C10_HOST_DEVICE Half operator-(Half a, uint32_t b) {
		return a - static_cast<Half>(b);
	}
	inline C10_HOST_DEVICE Half operator*(Half a, uint32_t b) {
		return a * static_cast<Half>(b);
	}
	inline C10_HOST_DEVICE Half operator/(Half a, uint32_t b) {
		return a / static_cast<Half>(b);
	}

	inline C10_HOST_DEVICE Half operator+(uint32_t a, Half b) {
		return static_cast<Half>(a) + b;
	}
	inline C10_HOST_DEVICE Half operator-(uint32_t a, Half b) {
		return static_cast<Half>(a) - b;
	}
	inline C10_HOST_DEVICE Half operator*(uint32_t a, Half b) {
		return static_cast<Half>(a) * b;
	}
	inline C10_HOST_DEVICE Half operator/(uint32_t a, Half b) {
		return static_cast<Half>(a) / b;
	}

	// uint16_t

	inline C10_HOST_DEVICE Half operator+(Half a, uint16_t b) {
		return a + static_cast<Half>(b);
	}
	inline C10_HOST_DEVICE Half operator-(Half a, uint16_t b) {
		return a - static_cast<Half>(b);
	}
	inline C10_HOST_DEVICE Half operator*(Half a, uint16_t b) {
		return a * static_cast<Half>(b);
	}
	inline C10_HOST_DEVICE Half operator/(Half a, uint16_t b) {
		return a / static_cast<Half>(b);
	}

	inline C10_HOST_DEVICE Half operator+(uint16_t a, Half b) {
		return static_cast<Half>(a) + b;
	}
	inline C10_HOST_DEVICE Half operator-(uint16_t a, Half b) {
		return static_cast<Half>(a) - b;
	}
	inline C10_HOST_DEVICE Half operator*(uint16_t a, Half b) {
		return static_cast<Half>(a) * b;
	}
	inline C10_HOST_DEVICE Half operator/(uint16_t a, Half b) {
		return static_cast<Half>(a) / b;
	}

	// uint8_t

	inline C10_HOST_DEVICE Half operator+(Half a, uint8_t b) {
		return a + static_cast<Half>(b);
	}
	inline C10_HOST_DEVICE Half operator-(Half a, uint8_t b) {
		return a - static_cast<Half>(b);
	}
	inline C10_HOST_DEVICE Half operator*(Half a, uint8_t b) {
		return a * static_cast<Half>(b);
	}
	inline C10_HOST_DEVICE Half operator/(Half a, uint8_t b) {
		return a / static_cast<Half>(b);
	}

	inline C10_HOST_DEVICE Half operator+(uint8_t a, Half b) {
		return static_cast<Half>(a) + b;
	}
	inline C10_HOST_DEVICE Half operator-(uint8_t a, Half b) {
		return static_cast<Half>(a) - b;
	}
	inline C10_HOST_DEVICE Half operator*(uint8_t a, Half b) {
		return static_cast<Half>(a) * b;
	}
	inline C10_HOST_DEVICE Half operator/(uint8_t a, Half b) {
		return static_cast<Half>(a) / b;
	}
}