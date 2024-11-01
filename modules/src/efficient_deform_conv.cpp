#include <torch/extension.h>
#include <efficient_deform_conv_utils.h>
#include <array_utils.h>
#include <type_utils.h>

#include <ATen/native/utils/ParamUtils.h>
#include <ATen/autocast_mode.h>


template<int8_t dim>
at::Tensor efficient_deform_conv_forward(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset_field,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride,
	at::IntArrayRef padding,
	at::IntArrayRef dilation,
	const int64_t groups,
	const at::Tensor& bias
)
{
	bool is_batched = input.dim() == dim + 2;
	const at::Tensor batched_input = is_batched ? input.contiguous() : input.contiguous().unsqueeze(0);

	TORCH_CHECK(dim > 0, "weight should have at least three dimensions");
	TORCH_CHECK(groups > 0, "non-positive groups is not supported");

	// the function is located torch::ops::custom_op namespace.
	std::string func_name = "custom_op::efficient_deform_conv";
	func_name += std::to_string(dim) + "d_forward";

	static auto op = torch::Dispatcher::singleton()
		.findSchemaOrThrow(func_name.c_str(), "")
		.typed<decltype(efficient_deform_conv_forward<dim>)>();
 
	at::Tensor output;

	if (!torch::GradMode::is_enabled())
	{
		torch::NoGradGuard no_grad;
		output = op.call(
			input,
			weight,
			offset_field,
			at::native::expand_param_if_needed(kernel_size, "kernel_size", dim),
			at::native::expand_param_if_needed(stride, "stride", dim),
			at::native::expand_param_if_needed(padding, "padding", dim),
			at::native::expand_param_if_needed(dilation, "dilation", dim),
			groups,
			bias
		);
	}
	else
	{
		output = op.call(
			input,
			weight,
			offset_field,
			at::native::expand_param_if_needed(kernel_size, "kernel_size", dim),
			at::native::expand_param_if_needed(stride, "stride", dim),
			at::native::expand_param_if_needed(padding, "padding", dim),
			at::native::expand_param_if_needed(dilation, "dilation", dim),
			groups,
			bias
		);
	}

	return (is_batched) ? output : output.squeeze(0);
}

template<int8_t dim>
torch::autograd::tensor_list efficient_deform_conv_backward(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset_field,
	const at::Tensor& grad_output,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride,
	at::IntArrayRef padding,
	at::IntArrayRef dilation,
	const int64_t groups,
	const at::Tensor& bias
)
{
	bool is_batched = input.dim() == dim + 2;
	const at::Tensor batched_input = is_batched ? input.contiguous() : input.contiguous().unsqueeze(0);

	TORCH_CHECK(dim > 0, "weight should have at least three dimensions");
	TORCH_CHECK(groups > 0, "non-positive groups is not supported");

	// the function is located torch::ops::custom_op namespace.
	std::string func_name = "custom_op::efficient_deform_conv";
	func_name += std::to_string(dim) + "d_backward";

	static auto op = torch::Dispatcher::singleton()
		.findSchemaOrThrow(func_name.c_str(), "")
		.typed<decltype(efficient_deform_conv_backward<dim>)>();

	torch::autograd::tensor_list grads;

	if (!torch::GradMode::is_enabled())
	{
		torch::NoGradGuard no_grad;
		grads = std::move(op.call(
			input,
			weight,
			offset_field,
			grad_output,
			at::native::expand_param_if_needed(kernel_size, "kernel_size", dim),
			at::native::expand_param_if_needed(stride, "stride", dim),
			at::native::expand_param_if_needed(padding, "padding", dim),
			at::native::expand_param_if_needed(dilation, "dilation", dim),
			groups,
			bias
		));
	}
	else
	{
		grads = std::move(op.call(
			input,
			weight,
			offset_field,
			grad_output,
			at::native::expand_param_if_needed(kernel_size, "kernel_size", dim),
			at::native::expand_param_if_needed(stride, "stride", dim),
			at::native::expand_param_if_needed(padding, "padding", dim),
			at::native::expand_param_if_needed(dilation, "dilation", dim),
			groups,
			bias
		));
	}

	if (!is_batched)
	{
		grads[0] = grads[0].squeeze(0);
		grads[2] = grads[2].squeeze(0);
	}

	return grads;
}


template<int8_t dim>// To supoort autograd.
class EfficientDeformConvNdFunction : public torch::autograd::Function<EfficientDeformConvNdFunction<dim>>
{
public:
	static at::Tensor forward(
		torch::autograd::AutogradContext* ctx,
		const at::Tensor& input,
		const at::Tensor& weight,
		const at::Tensor& offset_field,
		at::IntArrayRef kernel_size,
		at::IntArrayRef stride,
		at::IntArrayRef padding,
		at::IntArrayRef dilation,
		const int64_t groups,
		const at::Tensor& bias
	) {
		// at::AutoNonVariableTypeMode g; will deprecated 1.10 version.
		at::AutoDispatchBelowADInplaceOrView g;

		ctx->save_for_backward(
			{ input, weight, offset_field, bias }
		);

		ctx->saved_data["kernel_size"] = kernel_size;
		ctx->saved_data["stride"] = stride;
		ctx->saved_data["padding"] = padding;
		ctx->saved_data["dilation"] = dilation;
		ctx->saved_data["groups"] = groups;

		return efficient_deform_conv_forward<dim>(
			input,
			weight,
			offset_field,
			kernel_size,
			stride,
			padding,
			dilation,
			groups,
			bias
		);
	}

	static torch::autograd::variable_list backward(
		torch::autograd::AutogradContext* ctx,
		torch::autograd::tensor_list grad_outputs
	) {
		at::Tensor grad_output = grad_outputs[0];
		torch::autograd::tensor_list tensors = ctx->get_saved_variables();

		return efficient_deform_conv_backward<dim>(
			tensors[0],
			tensors[1],
			tensors[2],
			grad_output,
			ctx->saved_data["kernel_size"].toIntVector(),
			ctx->saved_data["stride"].toIntVector(),
			ctx->saved_data["padding"].toIntVector(),
			ctx->saved_data["dilation"].toIntVector(),
			ctx->saved_data["groups"].toInt(),
			tensors[3]
		);
	}
};

template<int8_t dim>
at::Tensor efficient_deform_conv_nd_autograd(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset_field,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride,
	at::IntArrayRef padding,
	at::IntArrayRef dilation,
	const int64_t groups,
	const at::Tensor& bias)
{
	return EfficientDeformConvNdFunction<dim>::apply(
		input,
		weight,
		offset_field,
		kernel_size,
		stride,
		padding,
		dilation,
		groups,
		bias
	);
}

template<int8_t dim>
at::Tensor efficient_deform_conv_nd_autocast_cpu(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset_field,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride,
	at::IntArrayRef padding,
	at::IntArrayRef dilation,
	const int64_t groups,
	const at::Tensor& bias)
{
	c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
	c10::DeviceType device = input.device().type();
	c10::ScalarType dtype = at::autocast::get_autocast_cpu_dtype();
	return efficient_deform_conv_nd_autograd<dim>(
		at::autocast::cached_cast(dtype, input, device),
		at::autocast::cached_cast(dtype, weight, device),
		at::autocast::cached_cast(dtype, offset_field, device),
		kernel_size,
		stride,
		padding,
		dilation,
		groups,
		at::autocast::cached_cast(dtype, bias, device)
		);
}

template<int8_t dim>
at::Tensor efficient_deform_conv_nd_autocast_cuda(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset_field,
	at::IntArrayRef kernel_size,
	at::IntArrayRef stride,
	at::IntArrayRef padding,
	at::IntArrayRef dilation,
	const int64_t groups,
	const at::Tensor& bias)
{
	c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
	c10::DeviceType device = input.device().type();
	c10::ScalarType dtype = at::autocast::get_autocast_gpu_dtype();
	return efficient_deform_conv_nd_autograd<dim>(
		at::autocast::cached_cast(dtype, input, device),
		at::autocast::cached_cast(dtype, weight, device),
		at::autocast::cached_cast(dtype, offset_field, device),
		kernel_size,
		stride,
		padding,
		dilation,
		groups,
		at::autocast::cached_cast(dtype, bias, device)
	);
}

TORCH_LIBRARY(custom_op, m)
{
	m.def("efficient_deform_conv1d(Tensor input, Tensor weight, Tensor offset_field, int[] kernel_size, int[] stride, int[] padding, int[] dilation, int groups, Tensor bias) -> Tensor");
	m.def("efficient_deform_conv2d(Tensor input, Tensor weight, Tensor offset_field, int[] kernel_size, int[] stride, int[] padding, int[] dilation, int groups, Tensor bias) -> Tensor");
	m.def("efficient_deform_conv3d(Tensor input, Tensor weight, Tensor offset_field, int[] kernel_size, int[] stride, int[] padding, int[] dilation, int groups, Tensor bias) -> Tensor");

	m.def("efficient_deform_conv1d_forward(Tensor input, Tensor weight, Tensor offset_field, int[] kernel_size, int[] stride, int[] padding, int[] dilation, int groups, Tensor bias) -> Tensor");
	m.def("efficient_deform_conv2d_forward(Tensor input, Tensor weight, Tensor offset_field, int[] kernel_size, int[] stride, int[] padding, int[] dilation, int groups, Tensor bias) -> Tensor");
	m.def("efficient_deform_conv3d_forward(Tensor input, Tensor weight, Tensor offset_field, int[] kernel_size, int[] stride, int[] padding, int[] dilation, int groups, Tensor bias) -> Tensor");

	m.def("efficient_deform_conv1d_backward(Tensor input, Tensor weight, Tensor offset_field, Tensor grad_output, int[] kernel_size, int[] stride, int[] padding, int[] dilation, int groups, Tensor bias) -> Tensor[]");
	m.def("efficient_deform_conv2d_backward(Tensor input, Tensor weight, Tensor offset_field, Tensor grad_output, int[] kernel_size, int[] stride, int[] padding, int[] dilation, int groups, Tensor bias) -> Tensor[]");
	m.def("efficient_deform_conv3d_backward(Tensor input, Tensor weight, Tensor offset_field, Tensor grad_output, int[] kernel_size, int[] stride, int[] padding, int[] dilation, int groups, Tensor bias) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(custom_op, Autograd, m)
{
	m.impl("efficient_deform_conv1d", &efficient_deform_conv_nd_autograd<1>);
	m.impl("efficient_deform_conv2d", &efficient_deform_conv_nd_autograd<2>);
	m.impl("efficient_deform_conv3d", &efficient_deform_conv_nd_autograd<3>);
}

TORCH_LIBRARY_IMPL(custom_op, AutocastCPU, m)
{
	m.impl("efficient_deform_conv1d", &efficient_deform_conv_nd_autocast_cpu<1>);
	m.impl("efficient_deform_conv2d", &efficient_deform_conv_nd_autocast_cpu<2>);
	m.impl("efficient_deform_conv3d", &efficient_deform_conv_nd_autocast_cpu<3>);
}

TORCH_LIBRARY_IMPL(custom_op, AutocastCUDA, m)
{
	m.impl("efficient_deform_conv1d", &efficient_deform_conv_nd_autocast_cuda<1>);
	m.impl("efficient_deform_conv2d", &efficient_deform_conv_nd_autocast_cuda<2>);
	m.impl("efficient_deform_conv3d", &efficient_deform_conv_nd_autocast_cuda<3>);
}


// Dummy to prevent python link error.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.doc() = "Pytorch implementation of deformable convolution Nd";
}