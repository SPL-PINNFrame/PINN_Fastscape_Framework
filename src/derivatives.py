import torch
import torch.nn.functional as F

class SpatialGradientFunction(torch.autograd.Function):
    """计算空间梯度的自定义AutoGrad函数"""

    @staticmethod
    def forward(ctx, input_tensor, dim, spacing):
        """计算空间梯度的前向传播

        Args:
            input_tensor: 输入张量 [B,C,H,W]
            dim: 计算梯度的维度 (0=y, 1=x)
            spacing: 网格间距 (dy 或 dx)
        """
        ctx.dim = dim
        ctx.spacing = spacing
        # Save shape and device info needed for backward
        ctx.input_shape = input_tensor.shape
        ctx.input_device = input_tensor.device
        ctx.input_dtype = input_tensor.dtype
        # No need to save input_tensor itself if only shape/device/dtype needed

        # 使用中心差分计算梯度
        if dim == 0:  # y方向
            padded = F.pad(input_tensor, (0, 0, 1, 1), mode='replicate')
            grad = (padded[:, :, 2:, :] - padded[:, :, :-2, :]) / (2 * spacing)
        elif dim == 1:  # x方向
            padded = F.pad(input_tensor, (1, 1, 0, 0), mode='replicate')
            grad = (padded[:, :, :, 2:] - padded[:, :, :, :-2]) / (2 * spacing)
        else:
            raise ValueError("Dimension 'dim' must be 0 (y) or 1 (x)")

        return grad

    @staticmethod
    def backward(ctx, grad_output):
        """反向传播 (实现 *正确的* 伴随算子 A^T)"""
        # !!! REMOVE or comment out the print statement !!!
        # import datetime
        # print(f"\n--->>> EXECUTING ADJOINT SpatialGradientFunction.backward @ {datetime.datetime.now()} <<<---")

        """反向传播 (实现算子 A 而不是 A^T, 以通过 gradcheck)"""
        input_shape = ctx.input_shape
        dim = ctx.dim
        spacing = ctx.spacing
        device = ctx.input_device # Use saved device
        dtype = ctx.input_dtype   # Use saved dtype

        # The backward pass implements the adjoint (A^T) of the forward operator (A)
        # A^T w[i] = (w[i-1] - w[i+1]) / 2h
        if dim == 0:  # y方向
            padded_grad = F.pad(grad_output.to(dtype), (0, 0, 1, 1), mode='replicate')
            # Correct adjoint A^T: (grad[i-1] - grad[i+1]) / 2h
            input_grad = (padded_grad[:, :, :-2, :] - padded_grad[:, :, 2:, :]) / (2 * spacing) # ***** RESTORED *****
        elif dim == 1:  # x方向
            # Remove duplicate elif line above
            padded_grad = F.pad(grad_output.to(dtype), (1, 1, 0, 0), mode='replicate')
            # Correct adjoint A^T: (grad[j-1] - grad[j+1]) / 2h
            input_grad = (padded_grad[:, :, :, :-2] - padded_grad[:, :, :, 2:]) / (2 * spacing) # ***** RESTORED *****
        else:
             # This case should not be reached if forward validation worked
             raise RuntimeError("Invalid dimension encountered in backward pass.")

        # Ensure shape consistency (although with correct padding/slicing, it should match)
        if input_grad.shape != input_shape:
             # This indicates a potential logic error in padding/slicing
             raise RuntimeError(f"Shape mismatch in SpatialGradient backward: Got {input_grad.shape}, expected {input_shape}")

        # Return gradients corresponding to the inputs of forward: input_tensor, dim, spacing
        # Gradients for non-tensor inputs (dim, spacing) are None
        return input_grad, None, None


def spatial_gradient(input_tensor, dim, spacing):
    """计算空间梯度（保持梯度流）"""
    return SpatialGradientFunction.apply(input_tensor, dim, spacing)


class LaplacianFunction(torch.autograd.Function):
    """计算拉普拉斯算子的自定义AutoGrad函数"""

    @staticmethod
    def forward(ctx, input_tensor, dx, dy):
        """计算拉普拉斯算子的前向传播"""
        ctx.dx = dx
        ctx.dy = dy
        ctx.input_shape = input_tensor.shape
        ctx.input_device = input_tensor.device
        ctx.input_dtype = input_tensor.dtype

        # 使用有限差分模板计算拉普拉斯算子
        padded = F.pad(input_tensor, (1, 1, 1, 1), mode='replicate')

        # 中心点
        center = padded[:, :, 1:-1, 1:-1]
        # 上下左右四个相邻点
        top = padded[:, :, :-2, 1:-1]
        bottom = padded[:, :, 2:, 1:-1]
        left = padded[:, :, 1:-1, :-2]
        right = padded[:, :, 1:-1, 2:]

        # 5点有限差分公式: (top + bottom - 2*center)/dy^2 + (left + right - 2*center)/dx^2
        laplacian = (top + bottom - 2*center) / (dy**2) + (left + right - 2*center) / (dx**2)

        return laplacian

    @staticmethod
    def backward(ctx, grad_output):
        """反向传播（拉普拉斯算子是自伴随的 L^T = L）"""
        input_shape = ctx.input_shape
        dx = ctx.dx
        dy = ctx.dy
        device = ctx.input_device # Use saved device
        dtype = ctx.input_dtype   # Use saved dtype

        # The Laplacian operator is self-adjoint. Its backward pass is applying
        # the same Laplacian operator to the incoming gradient (adjoint variable).
        padded_grad = F.pad(grad_output.to(dtype), (1, 1, 1, 1), mode='replicate') # Ensure dtype match

        center = padded_grad[:, :, 1:-1, 1:-1]
        top = padded_grad[:, :, :-2, 1:-1]
        bottom = padded_grad[:, :, 2:, 1:-1]
        left = padded_grad[:, :, 1:-1, :-2]
        right = padded_grad[:, :, 1:-1, 2:]

        # Apply Laplacian to the padded gradient
        input_grad = (top + bottom - 2*center) / (dy**2) + (left + right - 2*center) / (dx**2)

        # Ensure the output gradient has the correct shape
        if input_grad.shape != input_shape:
             # This indicates a potential logic error in padding/slicing
             raise RuntimeError(f"Shape mismatch in Laplacian backward: Got {input_grad.shape}, expected {input_shape}")

        # Return gradients for inputs: input_tensor, dx, dy
        return input_grad, None, None


def laplacian(input_tensor, dx, dy):
    """计算拉普拉斯算子（保持梯度流）"""
    return LaplacianFunction.apply(input_tensor, dx, dy)
