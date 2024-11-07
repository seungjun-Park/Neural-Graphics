#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <array_utils.h>
#include <interpolation.h>
#include <type_traits>

// implementation of n-dimensional im2col.
// unlike cpu version, cuda version was implemented only n-dimensional.
// because n-dimension specific version has same overhead to n-dimensional verison.

template<typename T, int8_t dim, bool is_channels_last>
__global__
typename std::enable_if <(dim > IMPLEMENTED_DIM && !is_channels_last), void>::type
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
    T* data_col) {

    int32_t ch_mul = (channels + blockDim.x - 1) / blockDim.x;
    int32_t ch = threadIdx.x + blockDim.x * (blockIdx.x % ch_mul);

    if (ch >= channels)
    {
        return;
    }

    int64_t kernel_sizes = multiply_integers<dim>(kernel_size);
    int64_t input_sizes = multiply_integers<dim>(input_size);
    int64_t output_sizes = multiply_integers<dim>(output_size);

    int64_t col = blockIdx.x / (ch_mul) % output_sizes;
    int64_t b = blockIdx.x / (ch_mul * output_sizes) % sub_batch;
    int64_t g = blockIdx.x / (ch_mul * output_sizes * sub_batch) % groups;

    extern __shared__ int8_t sharedMem[];

    T* data_offset_field_shm = reinterpret_cast<T*>(sharedMem);
    T* data_attn_mask_shm = reinterpret_cast<T*>(sharedMem) + kernel_sizes * dim;

    int64_t offset_field_idx = (b * groups + g) * kernel_sizes * dim * output_sizes + col;
    int64_t attn_mask_idx = (b * groups + g) * kernel_sizes * output_sizes + col;

    if (threadIdx.x == 0)
    {
        for (int64_t k = 0; k < kernel_sizes; k++)
        {
            for (int8_t i = 0; i < dim; i++)
            {
                data_offset_field_shm[k * dim + i] = data_offset_field[offset_field_idx + (k * dim + i) * output_sizes];
            }
            data_attn_mask_shm[k] = data_attn_mask[attn_mask_idx + k * output_sizes];
        }
    }

    __syncthreads();

    data_im += ((b  * groups + g) * channels + ch) * input_sizes;
    data_col += ((g * channels + ch) * kernel_sizes * sub_batch + b) * output_sizes + col;

    int64_t current_output_size[dim];
    int64_t current_kernel_size[dim];
    Array<T, dim> coord;

    for (int64_t k = 0; k < kernel_sizes; k++)
    {
        int64_t out_div = 1;
        int64_t k_div = 1;
        // compute current kernel size, output size and coord.
        for (int8_t i = dim - 1; i >= 0; i--)
        {
            current_kernel_size[i] = k / k_div % kernel_size[i];
            current_output_size[i] = col / out_div % output_size[i];
            out_div *= output_size[i];
            k_div *= kernel_size[i];
            coord[i] = current_output_size[i] * stride[i] - padding[i] + current_kernel_size[i] * dilation[i] + data_offset_field_shm[k * dim + i];
        }

        T val = linear_interp_nd<T, dim, is_channels_last>(data_im, coord, input_size, channels * groups);

        data_col[k * sub_batch * output_sizes] = val * data_attn_mask_shm[k];
    }

    __syncthreads();
}

template<typename T, int8_t dim, bool is_channels_last>
__global__
typename std::enable_if <(dim == 1 && !is_channels_last), void>::type
im2col_nd_cuda(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int64_t sub_batch,
    const int64_t channels,
    const IntArray<1> input_size,
    const IntArray<1> output_size,
    const IntArray<1> kernel_size,
    const IntArray<1> stride,
    const IntArray<1> padding,
    const IntArray<1> dilation,
    const int64_t groups,
    T* data_col) {

    int32_t ch_mul = (channels + blockDim.x - 1) / blockDim.x;
    int32_t ch = threadIdx.x + blockDim.x * (blockIdx.x % ch_mul);

    if (ch >= channels)
    {
        return;
    }

    int64_t kernel_sizes = multiply_integers<1>(kernel_size);
    int64_t input_sizes = multiply_integers<1>(input_size);
    int64_t output_sizes = multiply_integers<1>(output_size);

    int64_t col = blockIdx.x / (ch_mul) % output_size[0];
    int64_t b = blockIdx.x / (ch_mul * output_sizes) % sub_batch;
    int64_t g = blockIdx.x / (ch_mul * output_sizes * sub_batch) % groups;

    extern __shared__ int8_t sharedMem[];

    T* data_offset_field_shm = reinterpret_cast<T*>(sharedMem);
    T* data_attn_mask_shm = reinterpret_cast<T*>(sharedMem) + kernel_sizes;

    int64_t offset_field_idx = (b * groups + g) * kernel_sizes * output_sizes + col;
    int64_t attn_mask_idx = (b * groups + g) * kernel_sizes * output_sizes + col;

    if (threadIdx.x == 0)
    {
        for (int64_t k = 0; k < kernel_size[0]; k++)
        {
            data_offset_field_shm[k] = data_offset_field[offset_field_idx + k * output_sizes];
            data_attn_mask_shm[k] = data_attn_mask[attn_mask_idx + k * output_sizes];
        }
    }

    __syncthreads();

    data_im += ((b * groups + g) * channels + ch) * input_sizes;
    data_col += ((g * channels + ch) * kernel_sizes * sub_batch + b) * output_sizes + col;

    Array<T, 1> coord;

    for (int64_t k = 0; k < kernel_size[0]; k++)
    {
        coord[0] = col * stride[0] - padding[0] + k * dilation[0] + data_offset_field_shm[k];

        T val = linear_interp_nd<T, 1, is_channels_last>(data_im, coord, input_size, channels * groups);

        data_col[k * sub_batch * output_sizes] = val * data_attn_mask_shm[k];
    }

    __syncthreads();
}

template<typename T, int8_t dim, bool is_channels_last>
__global__
typename std::enable_if <(dim == 2 && !is_channels_last), void>::type
im2col_nd_cuda(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int64_t sub_batch,
    const int64_t channels,
    const IntArray<2> input_size,
    const IntArray<2> output_size,
    const IntArray<2> kernel_size,
    const IntArray<2> stride,
    const IntArray<2> padding,
    const IntArray<2> dilation,
    const int64_t groups,
    T* data_col) {

    int32_t ch_mul = (channels + blockDim.x - 1) / blockDim.x;
    int32_t ch = threadIdx.x + blockDim.x * (blockIdx.x % ch_mul);

    if (ch >= channels)
    {
        return;
    }

    int64_t kernel_sizes = multiply_integers<2>(kernel_size);
    int64_t input_sizes = multiply_integers<2>(input_size);
    int64_t output_sizes = multiply_integers<2>(output_size);

    int64_t w_col = blockIdx.x / (ch_mul) % output_size[1];
    int64_t h_col = blockIdx.x / (ch_mul * output_size[1]) % output_size[0];
    int64_t b = blockIdx.x / (ch_mul * output_sizes) % sub_batch;
    int64_t g = blockIdx.x / (ch_mul * output_sizes * sub_batch) % groups;

    extern __shared__ int8_t sharedMem[];

    T* data_offset_field_shm = reinterpret_cast<T*>(sharedMem);
    T* data_attn_mask_shm = reinterpret_cast<T*>(sharedMem) + kernel_sizes * 2;

    int64_t offset_field_idx = ((b * groups + g) * kernel_sizes * 2 * output_size[0] + h_col) * output_size[1] + w_col;
    int64_t attn_mask_idx = ((b * groups + g) * kernel_sizes * output_size[0] + h_col) * output_size[1] + w_col;

    if (threadIdx.x == 0)
    {
        for (int64_t h_k = 0; h_k < kernel_size[0]; h_k++)
        {
            for (int64_t w_k = 0; w_k < kernel_size[1]; w_k++)
            {
                int64_t k_idx = (h_k * kernel_size[1] + w_k);

                data_offset_field_shm[k_idx * 2] = data_offset_field[offset_field_idx + k_idx * 2 * output_sizes];
                data_offset_field_shm[k_idx * 2 + 1] = data_offset_field[offset_field_idx + (k_idx * 2 + 1) * output_sizes];
                data_attn_mask_shm[k_idx] = data_attn_mask[attn_mask_idx + k_idx * output_sizes];
            }
        }
    }

    __syncthreads();

    data_im += ((b * groups + g) * channels + ch) * input_sizes;
    data_col += (((g * channels + ch) * kernel_sizes * sub_batch + b) * output_size[0] + h_col) * output_size[1] + w_col;

    Array<T, 2> coord;

    for (int64_t h_k = 0; h_k < kernel_size[0]; h_k++)
    {
        for (int64_t w_k = 0; w_k < kernel_size[1]; w_k++)
        {
            int64_t k_idx = h_k * kernel_size[1] + w_k;

            coord[0] = h_col * stride[0] - padding[0] + h_k * dilation[0] + data_offset_field_shm[k_idx * 2];
            coord[1] = w_col * stride[1] - padding[1] + w_k * dilation[1] + data_offset_field_shm[k_idx * 2 + 1];

            T val = linear_interp_nd<T, 2, is_channels_last>(data_im, coord, input_size, channels * groups);
            
            data_col[k_idx * sub_batch * output_sizes] = val * data_attn_mask_shm[k_idx];
        }
    }
    __syncthreads();
}


template<typename T, int8_t dim, bool is_channels_last>
__global__
typename std::enable_if <(dim == 3 && !is_channels_last), void>::type
im2col_nd_cuda(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int64_t sub_batch,
    const int64_t channels,
    const IntArray<3> input_size,
    const IntArray<3> output_size,
    const IntArray<3> kernel_size,
    const IntArray<3> stride,
    const IntArray<3> padding,
    const IntArray<3> dilation,
    const int64_t groups,
    T* data_col) {

    int32_t ch_mul = (channels + blockDim.x - 1) / blockDim.x;
    int32_t ch = threadIdx.x + blockDim.x * (blockIdx.x % ch_mul);

    if (ch >= channels)
    {
        return;
    }

    int64_t kernel_sizes = multiply_integers<3>(kernel_size);
    int64_t input_sizes = multiply_integers<3>(input_size);
    int64_t output_sizes = multiply_integers<3>(output_size);

    int64_t w_col = blockIdx.x / (ch_mul) % output_size[2];
    int64_t h_col = blockIdx.x / (ch_mul * output_size[2]) % output_size[1];
    int64_t d_col = blockIdx.x / (ch_mul * output_size[1] * output_size[2]) % output_size[0];
    int64_t b = blockIdx.x / (ch_mul * output_sizes) % sub_batch;
    int64_t g = blockIdx.x / (ch_mul * output_sizes * sub_batch) % groups;

    extern __shared__ int8_t sharedMem[];

    T* data_offset_field_shm = reinterpret_cast<T*>(sharedMem);
    T* data_attn_mask_shm = reinterpret_cast<T*>(sharedMem) + kernel_sizes * 3;

    int64_t offset_field_idx = (((b * groups + g) * kernel_sizes * 3 * output_size[0] + d_col) * output_size[1] + h_col) * output_size[2] + w_col;
    int64_t attn_mask_idx = (((b * groups + g) * kernel_sizes * output_size[0] + d_col) * output_size[1] + h_col) * output_size[2] + w_col;

    if (threadIdx.x == 0)
    {
        for (int64_t d_k = 0; d_k < kernel_size[0]; d_k++)
        {
            for (int64_t h_k = 0; h_k < kernel_size[1]; h_k++)
            {
                for (int64_t w_k = 0; w_k < kernel_size[2]; w_k++)
                {
                    int64_t k_idx = ((d_k * kernel_size[1] + h_k) * kernel_size[2] + w_k);

                    data_offset_field_shm[k_idx * 3] = data_offset_field[offset_field_idx + k_idx * 3 * output_sizes];
                    data_offset_field_shm[k_idx * 3 + 1] = data_offset_field[offset_field_idx + (k_idx * 3 + 1) * output_sizes];
                    data_offset_field_shm[k_idx * 3 + 2] = data_offset_field[offset_field_idx + (k_idx * 3 + 2) * output_sizes];

                    data_attn_mask_shm[k_idx] = data_attn_mask[attn_mask_idx + k_idx * output_sizes];
                }
            }
        }
    }

    __syncthreads();

    data_im += ((b * groups + g) * channels + ch) * input_sizes;
    data_col += ((((g * channels + ch) * kernel_sizes * sub_batch + b) * output_size[0] + d_col) * output_size[1] + h_col) * output_size[2] + w_col;

    Array<T, 3> coord;

    for (int64_t d_k = 0; d_k < kernel_size[0]; d_k++)
    {
        for (int64_t h_k = 0; h_k < kernel_size[1]; h_k++)
        {
            for (int64_t w_k = 0; w_k < kernel_size[2]; w_k++)
            {
                int64_t k_idx = (d_k * kernel_size[1] + h_k) * kernel_size[1] + w_k;
                coord[0] = d_col * stride[0] - padding[0] + d_k * dilation[0] + data_offset_field_shm[k_idx * 3];
                coord[1] = h_col * stride[1] - padding[1] + h_k * dilation[1] + data_offset_field_shm[k_idx * 3 + 1];
                coord[2] = w_col * stride[2] - padding[2] + w_k * dilation[2] + data_offset_field_shm[k_idx * 3 + 2];

                T val = linear_interp_nd<T, 3, is_channels_last>(data_im, coord, input_size, channels * groups);

                data_col[k_idx * sub_batch * output_sizes] = val * data_attn_mask_shm[k_idx];
            }
        }
    }

    __syncthreads();
}



template<typename T, int8_t dim, bool is_channels_last>
__global__
typename std::enable_if <(dim > IMPLEMENTED_DIM&& is_channels_last), void>::type
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
    T* data_col) {

    int32_t ch_mul = (channels + blockDim.x - 1) / blockDim.x;
    int32_t ch = threadIdx.x + blockDim.x * (blockIdx.x % ch_mul);

    if (ch >= channels)
    {
        return;
    }

    int64_t kernel_sizes = multiply_integers<dim>(kernel_size);
    int64_t input_sizes = multiply_integers<dim>(input_size);
    int64_t output_sizes = multiply_integers<dim>(output_size);

    int64_t col = blockIdx.x / (ch_mul) % output_sizes;
    int64_t b = blockIdx.x / (ch_mul * output_sizes) % sub_batch;
    int64_t g = blockIdx.x / (ch_mul * output_sizes * sub_batch) % groups;

    extern __shared__ int8_t sharedMem[];

    T* data_offset_field_shm = reinterpret_cast<T*>(sharedMem);
    T* data_attn_mask_shm = reinterpret_cast<T*>(sharedMem) + kernel_sizes * dim;

    int64_t idx = ((b * output_sizes + col) * groups + g) * kernel_sizes;

    if (threadIdx.x == 0)
    {
        for (int64_t k = 0; k < kernel_sizes; k++)
        {
            for (int8_t i = 0; i < dim; i++)
            {
                data_offset_field_shm[k * dim + i] = data_offset_field[(idx + k) * dim + i];
            }
            data_attn_mask_shm[k] = data_attn_mask[idx + k];
        }
    }

    __syncthreads();

    data_im += (b * input_sizes * groups + g) * channels + ch;
    data_col += (((g * sub_batch + b) * output_sizes + col) * channels + ch) * kernel_sizes;

    int64_t current_output_size[dim];
    int64_t current_kernel_size[dim];
    Array<T, dim> coord;

    for (int64_t k = 0; k < kernel_sizes; k++)
    {
        int64_t out_div = 1;
        int64_t k_div = 1;
        // compute current kernel size, output size and coord.
        for (int8_t i = dim - 1; i >= 0; i--)
        {
            current_kernel_size[i] = k / k_div % kernel_size[i];
            current_output_size[i] = col / out_div % output_size[i];
            out_div *= output_size[i];
            k_div *= kernel_size[i];
            coord[i] = current_output_size[i] * stride[i] - padding[i] + current_kernel_size[i] * dilation[i] + data_offset_field_shm[k * dim + i];
        }

        T val = linear_interp_nd<T, dim, is_channels_last>(data_im, coord, input_size, channels * groups);

        data_col[k] = val * data_attn_mask_shm[k];
    }

    __syncthreads();
}

template<typename T, int8_t dim, bool is_channels_last>
__global__
typename std::enable_if <(dim == 1 && is_channels_last), void>::type
im2col_nd_cuda(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int64_t sub_batch,
    const int64_t channels,
    const IntArray<1> input_size,
    const IntArray<1> output_size,
    const IntArray<1> kernel_size,
    const IntArray<1> stride,
    const IntArray<1> padding,
    const IntArray<1> dilation,
    const int64_t groups,
    T* data_col) {

    int32_t ch_mul = (channels + blockDim.x - 1) / blockDim.x;
    int32_t ch = threadIdx.x + blockDim.x * (blockIdx.x % ch_mul);

    if (ch >= channels)
    {
        return;
    }

    int64_t kernel_sizes = multiply_integers<1>(kernel_size);
    int64_t input_sizes = multiply_integers<1>(input_size);
    int64_t output_sizes = multiply_integers<1>(output_size);

    int64_t col = blockIdx.x / (ch_mul) % output_size[0];
    int64_t b = blockIdx.x / (ch_mul * output_sizes) % sub_batch;
    int64_t g = blockIdx.x / (ch_mul * output_sizes * sub_batch) % groups;

    extern __shared__ int8_t sharedMem[];

    T* data_offset_field_shm = reinterpret_cast<T*>(sharedMem);
    T* data_attn_mask_shm = reinterpret_cast<T*>(sharedMem) + kernel_sizes;

    int64_t idx = ((b * output_size[0] + col) * groups + g) * kernel_sizes;

    if (threadIdx.x == 0)
    {
        for (int64_t k = 0; k < kernel_size[0]; k++)
        {
            data_offset_field_shm[k] = data_offset_field[idx + k];
            data_attn_mask_shm[k] = data_attn_mask[idx + k];
        }
    }

    __syncthreads();

    data_im += (b * input_sizes * groups + g) * channels + ch;
    data_col += (((g * sub_batch + b) * output_sizes + col) * channels + ch) * kernel_sizes;

    Array<T, 1> coord;

    for (int64_t k = 0; k < kernel_size[0]; k++)
    {
        coord[0] = col * stride[0] - padding[0] + k * dilation[0] + data_offset_field_shm[k];

        T val = linear_interp_nd<T, 1, is_channels_last>(data_im, coord, input_size, channels * groups);

        data_col[k] = val * data_attn_mask_shm[k];
    }

    __syncthreads();
}

template<typename T, int8_t dim, bool is_channels_last>
__global__
typename std::enable_if <(dim == 2 && is_channels_last), void>::type
im2col_nd_cuda(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int64_t sub_batch,
    const int64_t channels,
    const IntArray<2> input_size,
    const IntArray<2> output_size,
    const IntArray<2> kernel_size,
    const IntArray<2> stride,
    const IntArray<2> padding,
    const IntArray<2> dilation,
    const int64_t groups,
    T* data_col) {

    int32_t ch_mul = (channels + blockDim.x - 1) / blockDim.x;
    int32_t ch = threadIdx.x + blockDim.x * (blockIdx.x % ch_mul);

    if (ch >= channels)
    {
        return;
    }

    int64_t kernel_sizes = multiply_integers<2>(kernel_size);
    int64_t input_sizes = multiply_integers<2>(input_size);
    int64_t output_sizes = multiply_integers<2>(output_size);

    int64_t w_col = blockIdx.x / (ch_mul) % output_size[1];
    int64_t h_col = blockIdx.x / (ch_mul * output_size[1]) % output_size[0];
    int64_t b = blockIdx.x / (ch_mul * output_sizes) % sub_batch;
    int64_t g = blockIdx.x / (ch_mul * output_sizes * sub_batch) % groups;

    extern __shared__ int8_t sharedMem[];

    T* data_offset_field_shm = reinterpret_cast<T*>(sharedMem);
    T* data_attn_mask_shm = reinterpret_cast<T*>(sharedMem) + kernel_sizes * 2;

    int64_t idx = (((b * output_size[0] + h_col) * output_size[1] + w_col) * groups + g) * kernel_sizes;

    if (threadIdx.x == 0)
    {
        for (int64_t h_k = 0; h_k < kernel_size[0]; h_k++)
        {
            for (int64_t w_k = 0; w_k < kernel_size[1]; w_k++)
            {
                int64_t k_idx = (h_k * kernel_size[1] + w_k);

                data_offset_field_shm[k_idx * 2] = data_offset_field[(idx + k_idx) * 2];
                data_offset_field_shm[k_idx * 2 + 1] = data_offset_field[(idx + k_idx) * 2 + 1];
                data_attn_mask_shm[k_idx] = data_attn_mask[idx + k_idx];
            }
        }
    }

    __syncthreads();

    data_im += (b * input_sizes * groups + g) * channels + ch;
    data_col += ((((g * sub_batch + b) * output_size[0] + h_col) * output_size[1] + w_col) * channels + ch) * kernel_sizes;

    Array<T, 2> coord;

    for (int64_t h_k = 0; h_k < kernel_size[0]; h_k++)
    {
        for (int64_t w_k = 0; w_k < kernel_size[1]; w_k++)
        {
            idx = h_k * kernel_size[1] + w_k;

            coord[0] = h_col * stride[0] - padding[0] + h_k * dilation[0] + data_offset_field_shm[idx * 2];
            coord[1] = w_col * stride[1] - padding[1] + w_k * dilation[1] + data_offset_field_shm[idx * 2 + 1];

            T val = linear_interp_nd<T, 2, is_channels_last>(data_im, coord, input_size, channels * groups);

            data_col[idx] = val * data_attn_mask_shm[idx];
        }
    }

    __syncthreads();
}


template<typename T, int8_t dim, bool is_channels_last>
__global__
typename std::enable_if <(dim == 3 && is_channels_last), void>::type
im2col_nd_cuda(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int64_t sub_batch,
    const int64_t channels,
    const IntArray<3> input_size,
    const IntArray<3> output_size,
    const IntArray<3> kernel_size,
    const IntArray<3> stride,
    const IntArray<3> padding,
    const IntArray<3> dilation,
    const int64_t groups,
    T* data_col) {

    int32_t ch_mul = (channels + blockDim.x - 1) / blockDim.x;
    int32_t ch = threadIdx.x + blockDim.x * (blockIdx.x % ch_mul);

    if (ch >= channels)
    {
        return;
    }

    int64_t kernel_sizes = multiply_integers<3>(kernel_size);
    int64_t input_sizes = multiply_integers<3>(input_size);
    int64_t output_sizes = multiply_integers<3>(output_size);

    int64_t w_col = blockIdx.x / (ch_mul) % output_size[2];
    int64_t h_col = blockIdx.x / (ch_mul * output_size[2]) % output_size[1];
    int64_t d_col = blockIdx.x / (ch_mul * output_size[1] * output_size[2]) % output_size[0];
    int64_t b = blockIdx.x / (ch_mul * output_sizes) % sub_batch;
    int64_t g = blockIdx.x / (ch_mul * output_sizes * sub_batch) % groups;

    extern __shared__ int8_t sharedMem[];

    T* data_offset_field_shm = reinterpret_cast<T*>(sharedMem);
    T* data_attn_mask_shm = reinterpret_cast<T*>(sharedMem) + kernel_sizes * 3;

    int64_t idx = ((((b * output_size[0] + d_col) * output_size[1] + h_col) * output_size[2] + w_col) * groups + g) * kernel_sizes;

    if (threadIdx.x == 0)
    {
        for (int64_t d_k = 0; d_k < kernel_size[0]; d_k++)
        {
            for (int64_t h_k = 0; h_k < kernel_size[1]; h_k++)
            {
                for (int64_t w_k = 0; w_k < kernel_size[2]; w_k++)
                {
                    int64_t k_idx = ((d_k * kernel_size[1] + h_k) * kernel_size[2] + w_k);

                    data_offset_field_shm[k_idx * 3] = data_offset_field[(idx + k_idx) * 3];
                    data_offset_field_shm[k_idx * 3 + 1] = data_offset_field[(idx + k_idx) * 3 + 1];
                    data_offset_field_shm[k_idx * 3 + 2] = data_offset_field[(idx + k_idx) * 3 + 2];

                    data_attn_mask_shm[k_idx] = data_attn_mask[idx + k_idx];
                }
            }
        }
    }

    __syncthreads();

    data_im += (b * input_sizes * groups + g) * channels + ch;
    data_col += (((((g * sub_batch + b) * output_size[0] + d_col) * output_size[1] + h_col) * output_size[2] + w_col) * channels + ch) * kernel_sizes;

    Array<T, 3> coord;

    for (int64_t d_k = 0; d_k < kernel_size[0]; d_k++)
    {
        for (int64_t h_k = 0; h_k < kernel_size[1]; h_k++)
        {
            for (int64_t w_k = 0; w_k < kernel_size[2]; w_k++)
            {
                idx = (d_k * kernel_size[1] + h_k) * kernel_size[1] + w_k;
                coord[0] = d_col * stride[0] - padding[0] + d_k * dilation[0] + data_offset_field_shm[idx * 3];
                coord[1] = h_col * stride[1] - padding[1] + h_k * dilation[1] + data_offset_field_shm[idx * 3 + 1];
                coord[2] = w_col * stride[2] - padding[2] + w_k * dilation[2] + data_offset_field_shm[idx * 3 + 2];

                T val = linear_interp_nd<T, 3, is_channels_last>(data_im, coord, input_size, channels * groups);

                data_col[idx] = val * data_attn_mask_shm[idx];
            }
        }
    }

    __syncthreads();
}