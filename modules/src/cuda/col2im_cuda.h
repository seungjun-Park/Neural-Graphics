#pragma once

#include <torch/extension.h>
#include <sm_60_atomic_functions.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <interpolation.h>
#include <array_utils.h>
#include <type_utils.h>
#include <type_traits>


///////////////////     Implementation      ///////////////////////

template<typename T, int8_t dim>
__global__
typename std::enable_if<(dim > IMPLEMENTED_DIM), void>::type
col2im_nd_cuda(
    const T* data_im,
    const T* data_col,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int32_t sub_batch,
    const int32_t channels,
    const IntArray<dim> input_size,
    const IntArray<dim> output_size,
    const IntArray<dim> kernel_size,
    const IntArray<dim> stride,
    const IntArray<dim> padding,
    const IntArray<dim> dilation,
    const int32_t groups,
    const int32_t deformable_groups,
    mapped_type<T>* data_grad_im,
    mapped_type<T>* data_grad_offset_field,
    mapped_type<T>* data_grad_attn_mask) {

    int32_t ch_mul = (channels + blockDim.x - 1) / blockDim.x;

    if (threadIdx.x + (blockDim.x * (blockIdx.x % ch_mul)) >= channels)
    {
        return;
    }

    int32_t input_sizes = multiply_integers<dim>(input_size);
    int32_t output_sizes = multiply_integers<dim>(output_size);
    int32_t kernel_sizes = multiply_integers<dim>(kernel_size);

    int32_t ch = threadIdx.x + (blockDim.x * (blockIdx.x % ch_mul));
    int32_t col = blockIdx.x / (ch_mul) % output_sizes;
    int32_t b = blockIdx.x / (output_sizes * ch_mul) % sub_batch;
    int32_t g = blockIdx.x / (sub_batch * output_sizes * ch_mul) % groups;
    int32_t deformable_channels = channels / deformable_groups;


    data_im += ((b * groups + g) * channels + ch) * input_sizes;
    data_col += (((g * sub_batch + b) * output_sizes + col) * channels + ch) * kernel_sizes;
    data_offset_field += (((b * output_sizes + col) * groups + g) * deformable_channels + ch / deformable_groups) * kernel_sizes * dim;
    data_attn_mask += (((b * output_sizes + col) * groups + g) * deformable_channels + ch / deformable_groups) * kernel_sizes;
    
    data_grad_im += ((b * groups + g) * channels + ch) * input_sizes;
    data_grad_offset_field += (((b * output_sizes + col) * groups + g) * deformable_channels + ch / deformable_groups) * kernel_sizes * dim;
    data_grad_attn_mask += (((b * output_sizes + col) * groups + g) * deformable_channels + ch / deformable_groups) * kernel_sizes;

    int32_t current_output_size[dim];
    int32_t out_div = 1;

    for (int8_t i = dim - 1; i >= 0; i--)
    {
        current_output_size[i] = col / out_div % output_size[i];
        out_div *= output_size[i];
    }

    int32_t current_kernel_size[dim];
    Array<T, dim> coord;

    for (int32_t k = 0; k < kernel_sizes; k++)
    {
        int32_t k_div = 1;

        // compute current kernel size, output size and coord.
        for (int8_t i = dim - 1; i >= 0; i--)
        {
            current_kernel_size[i] = k / k_div % kernel_size[i];
            k_div *= kernel_size[i];

            coord[i] = current_output_size[i] * stride[i] - padding[i] + current_kernel_size[i] * dilation[i] + *(data_offset_field + i);
        }

        T val = linear_interp_nd<T, dim>(data_im, coord, input_size);
        atomicAdd(data_grad_attn_mask, (mapped_type<T>)((*data_col) * val));

        Array<T, dim> grad_coord = linear_interp_nd_grad<T, dim>(data_im, coord, input_size);

        for (int8_t i = 0; i < dim; i++)
        {
            atomicAdd(data_grad_offset_field + i, (mapped_type<T>)((*data_col) * grad_coord[i] * (*data_attn_mask)));
        }

        linear_interp_nd_weight<T, dim>(*data_col, *data_attn_mask, coord, input_size, data_grad_im);

        data_col++;
        data_offset_field += dim;
        data_attn_mask++;

        data_grad_offset_field += dim;
        data_grad_attn_mask++;
    }
}

template<typename T, int8_t dim>
__global__
typename std::enable_if<(dim == 1), void>::type
col2im_nd_cuda(
    const T* data_im,
    const T* data_col,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int32_t sub_batch,
    const int32_t channels,
    const IntArray<1> input_size,
    const IntArray<1> output_size,
    const IntArray<1> kernel_size,
    const IntArray<1> stride,
    const IntArray<1> padding,
    const IntArray<1> dilation,
    const int32_t groups,
    const int32_t deformable_groups,
    mapped_type<T>* data_grad_im,
    mapped_type<T>* data_grad_offset_field,
    mapped_type<T>* data_grad_attn_mask) {

    int32_t ch_mul = (channels + blockDim.x - 1) / blockDim.x;

    if (threadIdx.x + (blockDim.x * (blockIdx.x % ch_mul)) >= channels)
    {
        return;
    }

    int32_t input_sizes = multiply_integers<1>(input_size);
    int32_t output_sizes = multiply_integers<1>(output_size);
    int32_t kernel_sizes = multiply_integers<1>(kernel_size);

    int32_t ch = threadIdx.x + (blockDim.x * (blockIdx.x % ch_mul));
    int32_t col = blockIdx.x / (ch_mul) % output_sizes;
    int32_t b = blockIdx.x / (output_sizes * ch_mul) % sub_batch;
    int32_t g = blockIdx.x / (sub_batch * output_sizes * ch_mul) % groups;
    int32_t deformable_channels = channels / deformable_groups;

    data_im += ((b * groups + g) * channels + ch) * input_sizes;
    data_col += (((g * sub_batch + b) * output_sizes + col) * channels + ch) * kernel_sizes;
    data_offset_field += (((b * output_sizes + col) * groups + g) * deformable_channels + ch / deformable_groups) * kernel_sizes;
    data_attn_mask += (((b * output_sizes + col) * groups + g) * deformable_channels + ch / deformable_groups) * kernel_sizes;

    data_grad_im += ((b * groups + g) * channels + ch) * input_sizes;
    data_grad_offset_field += (((b * output_sizes + col) * groups + g) * deformable_channels + ch / deformable_groups) * kernel_sizes;
    data_grad_attn_mask += (((b * output_sizes + col) * groups + g) * deformable_channels + ch / deformable_groups) * kernel_sizes;

    Array<T, 1> coord;

    for (int32_t k = 0; k < kernel_size[0]; k++)
    {
        coord[0] = output_size[0] * stride[0] - padding[0] + k * dilation[0] + *(data_offset_field);
        T val = linear_interp_nd<T, 1>(data_im, coord, input_size);
        atomicAdd(data_grad_attn_mask, (mapped_type<T>)((*data_col) * val));

        Array<T, 1> grad_coord = linear_interp_nd_grad<T, 1>(data_im, coord, input_size);
        atomicAdd(data_grad_offset_field, (mapped_type<T>)((*data_col) * grad_coord[0] * (*data_attn_mask)));
    
        linear_interp_nd_weight<T, 1>(*data_col, *data_attn_mask, coord, input_size, data_grad_im);

        data_col++;
        data_offset_field++;
        data_attn_mask++;

        data_grad_offset_field++;
        data_grad_attn_mask++;
    }
}

template<typename T, int8_t dim>
__global__
typename std::enable_if<(dim == 2), void>::type
col2im_nd_cuda(
    const T* data_im,
    const T* data_col,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int32_t sub_batch,
    const int32_t channels,
    const IntArray<2> input_size,
    const IntArray<2> output_size,
    const IntArray<2> kernel_size,
    const IntArray<2> stride,
    const IntArray<2> padding,
    const IntArray<2> dilation,
    const int32_t groups,
    const int32_t deformable_groups,
    mapped_type<T>* data_grad_im,
    mapped_type<T>* data_grad_offset_field,
    mapped_type<T>* data_grad_attn_mask) {

    int32_t ch_mul = (channels + blockDim.x - 1) / blockDim.x;

    if (threadIdx.x + (blockDim.x * (blockIdx.x % ch_mul)) >= channels)
    {
        return;
    }

    int32_t input_sizes = multiply_integers<2>(input_size);
    int32_t output_sizes = multiply_integers<2>(output_size);
    int32_t kernel_sizes = multiply_integers<2>(kernel_size);

    int32_t ch = threadIdx.x + (blockDim.x * (blockIdx.x % ch_mul));
    int32_t col = blockIdx.x / (ch_mul) % output_sizes;
    int32_t b = blockIdx.x / (output_sizes * ch_mul) % sub_batch;
    int32_t g = blockIdx.x / (sub_batch * output_sizes * ch_mul) % groups;
    int32_t deformable_channels = channels / deformable_groups;

    data_im += ((b * groups + g) * channels + ch) * input_sizes;
    data_col += (((g * sub_batch + b) * output_sizes + col) * channels + ch) * kernel_sizes;
    data_offset_field += (((b * output_sizes + col) * groups + g) * deformable_channels + ch / deformable_groups) * kernel_sizes * 2;
    data_attn_mask += (((b * output_sizes + col) * groups + g) * deformable_channels + ch / deformable_groups) * kernel_sizes;

    data_grad_im += ((b * groups + g) * channels + ch) * input_sizes;
    data_grad_offset_field += (((b * output_sizes + col) * groups + g) * deformable_channels + ch / deformable_groups) * kernel_sizes * 2;
    data_grad_attn_mask += (((b * output_sizes + col) * groups + g) * deformable_channels + ch / deformable_groups) * kernel_sizes;

    Array<T, 2> coord;

    for (int32_t k_h = 0; k_h < kernel_size[0]; k_h++)
    {
        for (int32_t k_w = 0; k_w < kernel_size[1]; k_w++)
        {
            coord[0] = output_size[0] * stride[0] - padding[0] + k_h * dilation[0] + *(data_offset_field);
            coord[1] = output_size[1] * stride[1] - padding[1] + k_w * dilation[1] + *(data_offset_field + 1);

            T val = linear_interp_nd<T, 2>(data_im, coord, input_size);
            atomicAdd(data_grad_attn_mask, (mapped_type<T>)((*data_col) * val));

            Array<T, 2> grad_coord = linear_interp_nd_grad<T, 2>(data_im, coord, input_size);
            atomicAdd(data_grad_offset_field, (mapped_type<T>)((*data_col) * grad_coord[0] * (*data_attn_mask)));
            atomicAdd(data_grad_offset_field + 1, (mapped_type<T>)((*data_col) * grad_coord[1] * (*data_attn_mask)));

            linear_interp_nd_weight<T, 2>(*data_col, *data_attn_mask, coord, input_size, data_grad_im);

            data_col++;
            data_offset_field += 2;
            data_attn_mask++;

            data_grad_offset_field += 2;
            data_grad_attn_mask++;
        }
    }
}

template<typename T, int8_t dim>
__global__
typename std::enable_if<(dim == 3), void>::type
col2im_nd_cuda(
    const T* data_im,
    const T* data_col,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int32_t sub_batch,
    const int32_t channels,
    const IntArray<3> input_size,
    const IntArray<3> output_size,
    const IntArray<3> kernel_size,
    const IntArray<3> stride,
    const IntArray<3> padding,
    const IntArray<3> dilation,
    const int32_t groups,
    const int32_t deformable_groups,
    mapped_type<T>* data_grad_im,
    mapped_type<T>* data_grad_offset_field,
    mapped_type<T>* data_grad_attn_mask) {

    int32_t ch_mul = (channels + blockDim.x - 1) / blockDim.x;

    if (threadIdx.x + (blockDim.x * (blockIdx.x % ch_mul)) >= channels)
    {
        return;
    }

    int32_t input_sizes = multiply_integers<3>(input_size);
    int32_t output_sizes = multiply_integers<3>(output_size);
    int32_t kernel_sizes = multiply_integers<3>(kernel_size);

    int32_t ch = threadIdx.x + (blockDim.x * (blockIdx.x % ch_mul));
    int32_t col = blockIdx.x / (ch_mul) % output_sizes;
    int32_t b = blockIdx.x / (output_sizes * ch_mul) % sub_batch;
    int32_t g = blockIdx.x / (sub_batch * output_sizes * ch_mul) % groups;
    int32_t deformable_channels = channels / deformable_groups;

    data_im += ((b * groups + g) * channels + ch) * input_sizes;
    data_col += (((g * sub_batch + b) * output_sizes + col) * channels + ch) * kernel_sizes;
    data_offset_field += (((b * output_sizes + col) * groups + g) * deformable_channels + ch / deformable_groups) * kernel_sizes * 2;
    data_attn_mask += (((b * output_sizes + col) * groups + g) * deformable_channels + ch / deformable_groups) * kernel_sizes;

    data_grad_im += ((b * groups + g) * channels + ch) * input_sizes;
    data_grad_offset_field += (((b * output_sizes + col) * groups + g) * deformable_channels + ch / deformable_groups) * kernel_sizes * 2;
    data_grad_attn_mask += (((b * output_sizes + col) * groups + g) * deformable_channels + ch / deformable_groups) * kernel_sizes;

    Array<T, 3> coord;

    for (int32_t k_d = 0; k_d < kernel_size[0]; k_d++)
    {
        for (int32_t k_h = 0; k_h < kernel_size[1]; k_h++)
        {
            for (int32_t k_w = 0; k_w < kernel_size[2]; k_w++)
            {
                coord[0] = output_size[0] * stride[0] - padding[0] + k_d * dilation[0] + *(data_offset_field);
                coord[1] = output_size[1] * stride[1] - padding[1] + k_h * dilation[1] + *(data_offset_field + 1);
                coord[2] = output_size[2] * stride[2] - padding[2] + k_w * dilation[2] + *(data_offset_field + 2);

                T val = linear_interp_nd<T, 3>(data_im, coord, input_size);
                atomicAdd(data_grad_attn_mask, (mapped_type<T>)((*data_col) * val));

                Array<T, 3> grad_coord = linear_interp_nd_grad<T, 3>(data_im, coord, input_size);
                atomicAdd(data_grad_offset_field, (mapped_type<T>)((*data_col) * grad_coord[0] * (*data_attn_mask)));
                atomicAdd(data_grad_offset_field + 1, (mapped_type<T>)((*data_col) * grad_coord[1] * (*data_attn_mask)));
                atomicAdd(data_grad_offset_field + 2, (mapped_type<T>)((*data_col) * grad_coord[2] * (*data_attn_mask)));

                linear_interp_nd_weight<T, 3>(*data_col, *data_attn_mask, coord, input_size, data_grad_im);

                data_col++;
                data_offset_field += 3;
                data_attn_mask++;

                data_grad_offset_field += 3;
                data_grad_attn_mask++;
            }
        }
    }
}