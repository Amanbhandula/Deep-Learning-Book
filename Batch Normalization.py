
# coding: utf-8

# In[1]:


"""Batch normalization.
  As described in http://arxiv.org/abs/1502.03167.
  Normalizes a tensor by `mean` and `variance`, and applies (optionally) a
  `scale` \\(\gamma\\) to it, as well as an `offset` \\(\beta\\):
  \\(\frac{\gamma(x-\mu)}{\sigma}+\beta\\)
  `mean`, `variance`, `offset` and `scale` are all expected to be of one of two
  shapes:
    * In all generality, they can have the same number of dimensions as the
      input `x`, with identical sizes as `x` for the dimensions that are not
      normalized over (the 'depth' dimension(s)), and dimension 1 for the
      others which are being normalized over.
      `mean` and `variance` in this case would typically be the outputs of
      `tf.nn.moments(..., keep_dims=True)` during training, or running averages
      thereof during inference.
    * In the common case where the 'depth' dimension is the last dimension in
      the input tensor `x`, they may be one dimensional tensors of the same
      size as the 'depth' dimension.
      This is the case for example for the common `[batch, depth]` layout of
      fully-connected layers, and `[batch, height, width, depth]` for
      convolutions.
      `mean` and `variance` in this case would typically be the outputs of
      `tf.nn.moments(..., keep_dims=False)` during training, or running averages
      thereof during inference.
  Args:
    x: Input `Tensor` of arbitrary dimensionality.
    mean: A mean `Tensor`.
    variance: A variance `Tensor`.
    offset: An offset `Tensor`, often denoted \\(\beta\\) in equations, or
      None. If present, will be added to the normalized tensor.
    scale: A scale `Tensor`, often denoted \\(\gamma\\) in equations, or
      `None`. If present, the scale is applied to the normalized tensor.
    variance_epsilon: A small float number to avoid dividing by 0.
    name: A name for this operation (optional).
  Returns:
    the normalized, scaled, offset tensor.
  """


# In[2]:


def batch_normalization(x,mean, variance, offset, scale, variance_epsilon, name=None):
    with ops.name_scope(name, "batchnorm", [x, mean, variance, scale, offset]):
        inv = math_ops.rsqrt(variance + variance_epsilon)
        if scale is not None:
            inv *= scale
        return x * inv + (offset - mean * inv if offset is not None else -mean * inv)


# In[3]:


def fused_batch_norm(
    x,
    scale,
    offset,  # pylint: disable=invalid-name
    mean=None,
    variance=None,
    epsilon=0.001,
    data_format="NHWC",
    is_training=True,
    name=None):
  r"""Batch normalization.
  As described in http://arxiv.org/abs/1502.03167.
  Args:
    x: Input `Tensor` of 4 dimensions.
    scale: A `Tensor` of 1 dimension for scaling.
    offset: A `Tensor` of 1 dimension for bias.
    mean: A `Tensor` of 1 dimension for population mean used for inference.
    variance: A `Tensor` of 1 dimension for population variance
              used for inference.
    epsilon: A small float number added to the variance of x.
    data_format: The data format for x. Either "NHWC" (default) or "NCHW".
    is_training: A bool value to specify if the operation is used for
                 training or inference.
    name: A name for this operation (optional).
  Returns:
    y: A 4D Tensor for the normalized, scaled, offsetted x.
    batch_mean: A 1D Tensor for the mean of x.
    batch_var: A 1D Tensor for the variance of x.
  Raises:
    ValueError: If mean or variance is not None when is_training is True.
  """
  x = ops.convert_to_tensor(x, name="input")
  scale = ops.convert_to_tensor(scale, name="scale")
  offset = ops.convert_to_tensor(offset, name="offset")
  if is_training:
    if (mean is not None) or (variance is not None):
      raise ValueError("Both 'mean' and 'variance' must be None "
                       "if is_training is True.")
  if mean is None:
    mean = constant_op.constant([])
  if variance is None:
    variance = constant_op.constant([])
  # Set a minimum epsilon to 1.001e-5, which is a requirement by CUDNN to
  # prevent exception (see cudnn.h).
  min_epsilon = 1.001e-5
  epsilon = epsilon if epsilon > min_epsilon else min_epsilon
  # TODO(reedwm): In a few weeks, switch to using the V2 version exclusively. We
  # currently only use the V2 version for float16 inputs, which is not supported
  # by the V1 version.
  if x.dtype == dtypes.float16 or x.dtype == dtypes.bfloat16:
    fused_batch_norm_func = gen_nn_ops.fused_batch_norm_v2
  else:
    fused_batch_norm_func = gen_nn_ops._fused_batch_norm  # pylint: disable=protected-access
  y, batch_mean, batch_var, _, _ = fused_batch_norm_func(
      x,
      scale,
      offset,
      mean,
      variance,
      epsilon=epsilon,
      data_format=data_format,
      is_training=is_training,
      name=name)
  return y, batch_mean, batch_var



# In[4]:




def batch_norm_with_global_normalization(t,
                                         m,
                                         v,
                                         beta,
                                         gamma,
                                         variance_epsilon,
                                         scale_after_normalization,
                                         name=None):
  """Batch normalization.
  This op is deprecated. See `tf.nn.batch_normalization`.
  Args:
    t: A 4D input Tensor.
    m: A 1D mean Tensor with size matching the last dimension of t.
      This is the first output from tf.nn.moments,
      or a saved moving average thereof.
    v: A 1D variance Tensor with size matching the last dimension of t.
      This is the second output from tf.nn.moments,
      or a saved moving average thereof.
    beta: A 1D beta Tensor with size matching the last dimension of t.
      An offset to be added to the normalized tensor.
    gamma: A 1D gamma Tensor with size matching the last dimension of t.
      If "scale_after_normalization" is true, this tensor will be multiplied
      with the normalized tensor.
    variance_epsilon: A small float number to avoid dividing by 0.
    scale_after_normalization: A bool indicating whether the resulted tensor
      needs to be multiplied with gamma.
    name: A name for this operation (optional).
  Returns:
     A batch-normalized `t`.
  """
  return batch_normalization(t, m, v, beta, gamma if scale_after_normalization
                             else None, variance_epsilon, name)


