import torch
import torch_npu
import torch.nn.functional as F
import numpy as np
import os

def silu_numpy_float16(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    sigmoid_x = 1 / (1 + np.exp(-x))
    result = x * sigmoid_x
    return result.astype(np.float16)

def generate_and_save_matrices():
    """
    在NPU上生成A和B矩阵, 计算 C = A @ B.T, 并将三个矩阵保存为bin文件。
    """
    # 1. 定义矩阵维度
    M = 128
    K = 2048
    N = 12288
    
    # 定义数据类型 (在NPU上使用float16效率更高)
    dtype = torch.float16
    
    # 定义输出目录
    current_script_path = os.path.abspath(__file__)
    father_dir = os.path.dirname(os.path.dirname(current_script_path))
    
    output_dir = os.path.join(father_dir, "output")
    input_dir = os.path.join(father_dir, "input")
    os.makedirs(output_dir, exist_ok=True)
    # print(f"文件将保存在 '{output_dir}' 目录下。")

    # 2. 检查NPU设备是否可用并设置
    if not torch.npu.is_available():
        print("错误：未检测到可用的 NPU 设备。请检查您的环境配置。")
        return

    device = torch.device("npu")
    print(f"正在使用设备: {torch.npu.get_device_name(0)}")
    
    try:
        # 3. 在NPU上创建输入矩阵 A 和 B
        print("\n--- 正在生成矩阵 ---")
        # 创建 [M, K] 大小的矩阵 A
        # A = torch.randint(1, 10, (M, K), dtype=dtype, device=device)
        # B = torch.randint(1, 10, (N, K), dtype=dtype, device=device)
        A = torch.randn(M, K, dtype=dtype, device=device)
        B = torch.randn(N * 2, K, dtype=dtype, device=device)
        C = torch.matmul(A, B.T)
        C = torch_npu.npu_swiglu(C)
        
        print(f"计算完成，生成矩阵 C，形状: {C.shape}, 数据类型: {C.dtype}")
        
        # 5. 将矩阵保存为 .bin 文件
        print("\n--- 正在保存文件 ---")
        
        # 定义一个辅助函数来保存张量
        def save_tensor_to_bin(tensor, filename, xput_dir):
            # 首先将张量从NPU移动到CPU
            tensor_cpu = tensor.cpu()
            # 将PyTorch张量转换为NumPy数组
            tensor_numpy = tensor_cpu.numpy()
            # 将NumPy数组的数据以二进制格式写入文件
            filepath = os.path.join(xput_dir, filename)
            tensor_numpy.tofile(filepath)
            print(f"已成功保存 '{filepath}'")

        # 保存 A, B, C
        save_tensor_to_bin(A, "x1_gm.bin", input_dir)
        save_tensor_to_bin(B, "x2_gm.bin", input_dir)
        save_tensor_to_bin(C, "golden.bin", output_dir)

        # print("\n所有操作已成功完成！")

    except Exception as e:
        print(f"\n在执行过程中发生错误: {e}")


if __name__ == "__main__":
    generate_and_save_matrices()