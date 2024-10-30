#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cassert>
#include <vector>

// replace to at::IntArrayRef in device code.
// it's usage is equal to at::IntArrayRef in deform conv cpu version.


#ifndef IMPLEMENTED_DIM
#define IMPLEMENTED_DIM 3
#endif // !IMPLEMENTED_DIM

template<typename T, uint8_t size>
struct Array
{
	__host__ __device__ T operator[](uint8_t index) const
	{
		assert(index < size);
		return elements[index];
	}

	__host__ __device__ T& operator[](uint8_t index)
	{
		assert(index < size);
		return elements[index];
	}

	T elements[size];
};

template<uint8_t size>
using UInt8Array = Array<uint8_t, size>;

template<uint8_t size>
using Int8Array = Array<int8_t, size>;

template<uint8_t size>
using UInt16Array = Array<uint16_t, size>;

template<uint8_t size>
using Int16Array = Array<int16_t, size>;

template<uint8_t size>
using Int32Array = Array<int32_t, size>;

template<uint8_t size>
using UInt32Array = Array<uint32_t, size>;

template<uint8_t size>
using Int64Array = Array<int64_t, size>;

template<uint8_t size>
using UInt64Array = Array<uint64_t, size>;

template<uint8_t size>
using FloatArray = Array<float_t, size>;

template<uint8_t size>
using DoubleArray = Array<double_t, size>;

template<typename T, uint8_t size>
Array<T, size> ArrayRef2Array(at::ArrayRef<T> arr)
{
	assert(arr.size() == size);
	Array<T, size> target;
	for (uint8_t i = 0; i < size; i++)
	{
		target[i] = arr[i];
	}

	return target;
}

template<uint8_t size>
UInt16Array<size> IntArrayRef2UInt16Array(at::IntArrayRef arr)
{
	assert(arr.size() == size);
	UInt16Array<size> target;
	for (uint8_t i = 0; i < size; i++)
	{
		target[i] = static_cast<uint16_t>(arr[i]);
	}

	return target;
}

template<uint8_t size>
UInt8Array<size> IntArrayRef2UInt8Array(at::IntArrayRef arr)
{
	assert(arr.size() == size);
	UInt8Array<size> target;
	for (uint8_t i = 0; i < size; i++)
	{
		target[i] = static_cast<uint8_t>(arr[i]);
	}

	return target;
}

template<typename T, uint8_t size>
Array<T, size> vector2Array(std::vector<T>& vec)
{
	assert(vec.size() == size);
	Array<T, size> target;
	for (uint8_t i = 0; i < size; i++)
	{
		target[i] = vec[i];
	}

	return target;
}

template<typename T, uint8_t size>
__host__ __device__ T multiply_elements(const Array<T, size>& arr)
{
	T mul = 1;

	for (uint8_t i = 0; i < size; i++)
	{
		mul *= arr[i];
	}

	return (T)mul;
}


template<uint8_t size>
__host__ __device__ uint32_t multiply_integers(const UInt16Array<size>& arr)
{
	uint32_t mul = 1;

	for (uint8_t i = 0; i < size; i++)
	{
		mul *= arr[i];
	}

	return (uint32_t)mul;
}

template<uint8_t size>
__host__ __device__ uint16_t multiply_integers(const UInt8Array<size>& arr)
{
	uint16_t mul = 1;

	for (uint8_t i = 0; i < size; i++)
	{
		mul *= arr[i];
	}

	return (uint16_t)mul;
}
