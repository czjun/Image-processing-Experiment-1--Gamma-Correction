import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 强制使用英文标题和标准字体，避免文字方块问题
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

def adaptive_gamma_correction(image_path, target=0.5, color_space='YCbCr', method='binary'):
    # 读取图像
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("图像未找到")
    else:
        # 允许直接传入图像数组
        img = image_path.copy()

    # 颜色空间转换及亮度提取
    if color_space == 'YCbCr':
        ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycbcr)
        Lmin, Lmax = 16, 235
    elif color_space == 'RGB':
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        y = np.max(img_rgb, axis=2)
        Lmin, Lmax = 0, 255
    elif color_space == 'HSV':
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, y = cv2.split(hsv)
        Lmin, Lmax = 0, 255
    elif color_space == 'Gray':
        y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        Lmin, Lmax = 0, 255
    else:
        raise ValueError("不支持的颜色空间")

    # 归一化亮度通道
    y_normalized = (y.astype(np.float32) - Lmin) / (Lmax - Lmin)
    y_normalized = np.clip(y_normalized, 0, 1)

    # 计算当前平均亮度
    avg_uni = np.mean(y_normalized)
    
    # 根据选择的方法计算gamma值
    if method == 'formula':
        # 使用公式直接计算
        gamma = calculate_gamma_formula(avg_uni, target)
    elif method == 'newton':
        # 使用牛顿迭代法
        gamma = calculate_gamma_newton(y_normalized, target)
    else:  # 默认使用二分法
        gamma = calculate_gamma_binary(y_normalized, target)

    # 应用Gamma校正
    y_corrected = (y_normalized ** gamma) * (Lmax - Lmin) + Lmin
    y_corrected = np.clip(y_corrected, 0, 255).astype(np.uint8)

    # 合并通道并转换回BGR
    if color_space == 'YCbCr':
        # 不进行色度增强，只调整亮度通道，避免紫色偏色问题
        cr_scale = 1.0
        cb_scale = 1.0
        
        cr = np.clip(cr.astype(np.float32) * cr_scale, 0, 255).astype(np.uint8)
        cb = np.clip(cb.astype(np.float32) * cb_scale, 0, 255).astype(np.uint8)
        merged = cv2.merge([y_corrected, cr, cb])
        result = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
    elif color_space == 'RGB':
        result = cv2.cvtColor(y_corrected, cv2.COLOR_GRAY2BGR)
    elif color_space == 'HSV':
        merged = cv2.merge([h, s, y_corrected])
        result = cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)
    elif color_space == 'Gray':
        result = y_corrected
    else:
        raise ValueError("颜色空间错误")

    return result, gamma

# 不同的Gamma计算方法
def calculate_gamma_formula(avg_uni, target=0.5):
    """使用公式直接计算Gamma值
    
    gamma = log(target) / log(avg_uni)
    """
    # 处理边界情况，避免log(0)错误
    eps = 1e-5
    avg_uni = max(avg_uni, eps)
    gamma = np.log(target) / np.log(avg_uni)
    return np.clip(gamma, 0.1, 10.0)  # 限制在合理范围内

def calculate_gamma_newton(y_normalized, target=0.5, max_iter=20, tol=1e-6):
    """使用牛顿迭代法计算最优Gamma值"""
    gamma = 1.0  # 初始猜测值
    for i in range(max_iter):
        # 计算当前gamma值下的平均亮度
        corrected_avg = np.mean(y_normalized ** gamma)
        # 计算函数值
        f = corrected_avg - target
        if abs(f) < tol:
            break
        # 计算导数
        df = np.mean((y_normalized ** gamma) * np.log(np.clip(y_normalized, 1e-10, 1.0)))
        # 避免除以接近0的值
        if abs(df) < 1e-10:
            break
        # 牛顿迭代更新
        gamma = gamma - f / df
        # 限制gamma范围，避免异常值
        gamma = np.clip(gamma, 0.1, 10.0)
    
    return gamma

def calculate_gamma_binary(y_normalized, target=0.5, max_iter=100, tol=1e-5):
    """使用二分法求解Gamma值"""
    avg_uni = np.mean(y_normalized)
    if np.isclose(avg_uni, target):
        return 1.0
    
    # 二分法求解Gamma值
    low, high = (0.1, 1.0) if avg_uni < target else (1.0, 10.0)
    for _ in range(max_iter):
        mid = (low + high) / 2
        corrected_avg = np.mean(y_normalized ** mid)
        error = corrected_avg - target
        if abs(error) < tol:
            break
        if error < 0:
            high = mid  # 需要更小的Gamma
        else:
            low = mid   # 需要更大的Gamma
    
    return mid

# 可视化功能
def show_comparison(original, corrected, gamma, save_path=None):
    """显示原图和校正后的图像对比"""
    plt.figure(figsize=(12, 6))
    
    # 显示原图
    plt.subplot(121)
    if len(original.shape) == 3:
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(original, cmap='gray')
    plt.title("Original", fontsize=12)
    plt.axis('off')
    
    # 显示校正后的图像
    plt.subplot(122)
    if len(corrected.shape) == 3:
        plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(corrected, cmap='gray')
    plt.title(f"Corrected (gamma={gamma:.2f})", fontsize=12)
    plt.axis('off')
    
    plt.suptitle("Gamma Correction Comparison", fontsize=14)
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_gamma_curve(gamma, save_path=None, x_range=(0, 1), points=100):
    """绘制Gamma校正曲线"""
    x = np.linspace(x_range[0], x_range[1], points)
    y = x ** gamma
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'r-', linewidth=2)
    plt.plot(x, x, 'k--', linewidth=1)  # 添加y=x参考线
    plt.title(f"Gamma Correction Curve (gamma={gamma:.2f})", fontsize=14)
    plt.xlabel("Input Brightness", fontsize=12)
    plt.ylabel("Output Brightness", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def analyze_results(original, corrected, gamma, save_dir=None):
    """分析原图和校正后图像的统计信息"""
    # 转换为灰度图进行分析
    if len(original.shape) == 3:
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        orig_gray = original.copy()
        
    if len(corrected.shape) == 3:
        corr_gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    else:
        corr_gray = corrected.copy()
    
    # 计算均值、标准差
    orig_mean = np.mean(orig_gray) / 255.0
    corr_mean = np.mean(corr_gray) / 255.0
    orig_std = np.std(orig_gray) / 255.0
    corr_std = np.std(corr_gray) / 255.0
    
    print(f"原图平均亮度: {orig_mean:.4f}, 标准差: {orig_std:.4f}")
    print(f"校正后平均亮度: {corr_mean:.4f}, 标准差: {corr_std:.4f}")
    print(f"使用的Gamma值: {gamma:.4f}")
    
    # 绘制直方图对比
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.hist(orig_gray.ravel(), 256, [0,256], color='blue', alpha=0.7)
    plt.axvline(np.mean(orig_gray), color='r', linestyle='dashed', linewidth=2)
    plt.title(f"Original Histogram (Avg: {orig_mean:.4f})", fontsize=12)
    plt.xlim([0, 256])
    
    plt.subplot(122)
    plt.hist(corr_gray.ravel(), 256, [0,256], color='green', alpha=0.7)
    plt.axvline(np.mean(corr_gray), color='r', linestyle='dashed', linewidth=2)
    plt.title(f"Corrected Histogram (Avg: {corr_mean:.4f})", fontsize=12)
    plt.xlim([0, 256])
    plt.tight_layout()
    
    # 保存图像
    if save_dir:
        save_path = os.path.join(save_dir, "histogram_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return {
        "orig_mean": orig_mean,
        "corr_mean": corr_mean,
        "orig_std": orig_std,
        "corr_std": corr_std,
        "gamma": gamma
    }

def compare_methods(image_path, target=0.5, color_space='YCbCr', save_dir=None):
    """比较不同Gamma计算方法的效果"""
    # 创建保存目录
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 读取原图
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError("图像未找到")
    
    # 使用不同方法计算Gamma值并校正
    methods = ['binary', 'formula', 'newton']
    results = {}
    
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original", fontsize=12)
    plt.axis('off')
    
    for i, method in enumerate(methods, start=2):
        corrected, gamma = adaptive_gamma_correction(original, target, color_space, method)
        results[method] = {
            "gamma": gamma,
            "image": corrected
        }
        
        plt.subplot(2, 2, i)
        plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
        plt.title(f"{method.capitalize()} Method (gamma={gamma:.2f})", fontsize=12)
        plt.axis('off')
    
    plt.suptitle(f"Comparison of Different Gamma Calculation Methods (Target: {target})", fontsize=14)
    plt.tight_layout()
    
    # 保存对比图
    if save_dir:
        plt.savefig(os.path.join(save_dir, "method_comparison.png"), dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # 返回结果
    return results

def compare_targets(image_path, targets=[0.3, 0.5, 0.7], color_space='YCbCr', method='binary', save_dir=None):
    """比较不同目标亮度值的效果"""
    # 创建保存目录
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 读取原图
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError("图像未找到")
    
    # 使用不同目标亮度值进行校正
    results = {}
    
    # 计算子图布局
    n = len(targets) + 1  # +1 for original
    rows = 2
    cols = (n + 1) // 2
    
    plt.figure(figsize=(cols * 6, rows * 5))
    
    # 显示原图
    plt.subplot(rows, cols, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original", fontsize=12)
    plt.axis('off')
    
    # 显示不同目标亮度的结果
    for i, target in enumerate(targets, start=2):
        corrected, gamma = adaptive_gamma_correction(original, target, color_space, method)
        results[target] = {
            "gamma": gamma,
            "image": corrected
        }
        
        plt.subplot(rows, cols, i)
        plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
        plt.title(f"Target: {target} (gamma={gamma:.2f})", fontsize=12)
        plt.axis('off')
    
    plt.suptitle(f"Comparison of Different Target Brightness Values (Color Space: {color_space})", fontsize=14)
    plt.tight_layout()
    
    # 保存对比图
    if save_dir:
        plt.savefig(os.path.join(save_dir, "target_comparison.png"), dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # 返回结果
    return results

def batch_process(input_dir, output_dir, target=0.5, color_space='YCbCr', method='binary'):
    """批量处理多张图像"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    input_dir = Path(input_dir)
    files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpeg'))
    
    if not files:
        print(f"在{input_dir}中未找到图像文件")
        return
    
    # 处理每张图像
    results = {}
    for file in files:
        print(f"处理: {file.name}")
        input_path = str(file)
        output_path = os.path.join(output_dir, f"corrected_{file.name}")
        
        # 进行Gamma校正
        corrected, gamma = adaptive_gamma_correction(input_path, target, color_space, method)
        
        # 保存结果
        cv2.imwrite(output_path, corrected)
        
        # 记录Gamma值
        results[file.name] = gamma
        
        print(f"  - Gamma值: {gamma:.4f}")
        print(f"  - 保存至: {output_path}")
    
    # 保存处理结果汇总
    with open(os.path.join(output_dir, "gamma_values.txt"), 'w') as f:
        f.write(f"目标亮度: {target}, 颜色空间: {color_space}, 计算方法: {method}\n\n")
        for name, gamma in results.items():
            f.write(f"{name}: gamma={gamma:.4f}\n")
    
    print(f"批量处理完成，共处理{len(files)}张图像")
    return results

# 使用示例
if __name__ == "__main__":
    input_path = 'input.jpg'
    
    # 创建结果目录
    results_dir = "gamma_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 读取原图
    original = cv2.imread(input_path)
    if original is None:
        print(f"未找到图像: {input_path}")
        exit(1)
    
    # 1. 基本Gamma校正
    corrected, gamma = adaptive_gamma_correction(input_path, target=0.5, color_space='YCbCr')
    cv2.imwrite(f"{results_dir}/output.jpg", corrected)
    
    # 2. 显示原图和校正后的图像对比
    show_comparison(original, corrected, gamma, save_path=f"{results_dir}/comparison.png")
    
    # 3. 绘制Gamma曲线
    plot_gamma_curve(gamma, save_path=f"{results_dir}/gamma_curve.png")
    
    # 4. 分析结果
    analyze_results(original, corrected, gamma, save_dir=results_dir)
    
    # 5. 比较不同Gamma计算方法
    compare_methods(input_path, target=0.5, color_space='YCbCr', save_dir=results_dir)
    
    # 6. 比较不同目标亮度值
    compare_targets(input_path, targets=[0.3, 0.5, 0.7], color_space='YCbCr', save_dir=results_dir)
    
    print(f"所有结果已保存到 {results_dir} 目录")
    print("注意：处理后的图像使用了仅调整亮度通道的方法，保持了原图的色彩平衡")
    
    # 如果需要批量处理更多图像，取消下面的注释
    # batch_process("input_images", "output_images", target=0.5, color_space='YCbCr')