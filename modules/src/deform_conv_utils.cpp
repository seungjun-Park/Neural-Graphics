#include <deform_conv_utils.h>

bool check_is_channels_last(const at::Tensor& target)
{
	int64_t dim = target.dim();
	auto sizes = target.sizes().vec();
	auto strides = target.strides().vec();

	std::rotate(sizes.begin() + 1, sizes.begin() + 2, sizes.end());
	std::rotate(strides.begin() + 1, strides.begin() + 2, strides.end());

	int64_t stride = 1;
	for (int64_t i = dim - 1; i >= 0; i--)
	{
		if (stride != strides[i])
		{
			return false;
		}
		stride *= sizes[i];
	}

	return true;
}

void check_deform_conv_backend(
	const at::Tensor& input,
	const at::Tensor& weight,
	const at::Tensor& offset_field,
	const at::Tensor& attn_mask,
	const at::Tensor& bias,
	const at::Tensor& grad_output,
	at::Backend location)
{

	at::checkBackend("check_deform_conv_backend", { input, weight, offset_field, attn_mask }, location);

	if (bias.defined())
	{
		at::checkBackend("check_deform_conv_backend", { bias }, location);
	}
	if (grad_output.defined())
	{
		at::checkBackend("check_deform_conv_backend", { grad_output }, location);
	}
}