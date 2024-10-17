#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <utils.h>
#include <interpolation.h>
#include <type_traits>

// implementation of n-dimensional im2col.
// unlike cpu version, cuda version was implemented only n-dimensional.
// because n-dimension specific version has same overhead to n-dimensional verison.

template<typename T, int64_t dim>
__global__
typename std::enable_if<(dim > 0), void>::type
im2col_nd_cuda(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int64_t sub_batch,
    const int64_t channels,
    const IntArray<dim> input_size,
    const IntArray<dim> output_size,
    const IntArray<dim> kernel_size,
    const IntArray<dim> stride,
    const IntArray<dim> padding,
    const IntArray<dim> dilation,
    const int64_t groups,
    const int64_t offset_field_channels_per_groups,
    T* data_col) {

    int64_t input_sizes = multiply_integers<dim>(input_size);
    int64_t output_sizes = multiply_integers<dim>(output_size);
    int64_t kernel_sizes = multiply_integers<dim>(kernel_size);

    int64_t num_col = groups * channels * kernel_sizes * sub_batch * output_sizes;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= num_col)
    {
        return;
    }

    int64_t col = idx % output_sizes;
    int64_t k = idx / output_sizes % kernel_sizes;
    int64_t ch = idx / (kernel_sizes * output_sizes) % channels;
    int64_t g = idx / (channels * kernel_sizes * output_sizes) % groups;
    int64_t batch_idx = idx / (groups * channels * kernel_sizes * output_sizes) % sub_batch;

    int64_t current_kernel_size[dim];
    int64_t current_output_size[dim];

    int64_t k_div = 1;
    int64_t out_div = 1;

    FloatArray<dim> coord;

    int64_t col_idx = (((g * channels + ch) * kernel_sizes + k) * sub_batch + batch_idx) * output_sizes + col;
    int64_t im_idx = ((batch_idx * groups + g) * channels + ch) * input_sizes;
    int64_t offset_field_idx = (((batch_idx * dim * groups + g) * offset_field_channels_per_groups + (ch * offset_field_channels_per_groups / channels)) * kernel_sizes + k) * output_sizes + col;
    int64_t base_offset_field_idx = groups * offset_field_channels_per_groups * kernel_sizes * output_sizes;
    int64_t attn_mask_idx = (((batch_idx * groups + g) * offset_field_channels_per_groups + (ch * offset_field_channels_per_groups / channels)) * kernel_sizes + k) * output_sizes + col;

    data_im += ((batch_idx * groups + g) * channels + ch) * input_sizes;
    data_col += col_idx;

    // compute current kernel size, output size and coord.
    for (int i = dim - 1; i >= 0; i--)
    {
        current_kernel_size[i] = k / k_div % kernel_size[i];
        current_output_size[i] = col / out_div % output_size[i];
        k_div *= kernel_size[i];
        out_div *= output_size[i];

        coord[i] = current_output_size[i] * stride[i] - padding[i] + current_kernel_size[i] * dilation[i] + (data_offset_field + base_offset_field_idx * i)[offset_field_idx];
    }

    T val = linear_interp_nd<T, dim>(data_im, coord, input_size);

    *data_col = val * data_attn_mask[attn_mask_idx];
}
