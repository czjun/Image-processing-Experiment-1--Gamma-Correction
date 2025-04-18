import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 基本字体设置避免文字显示问题
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

def adaptive_gamma_correction(image_path, target=0.5, color_space='YCbCr', method='binary'):
    # 读取图像
    img = cv2.imread(image_path) if isinstance(image_path, str) else image_path.copy()
    
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
    else:  # Grayscale
        y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        Lmin, Lmax = 0, 255

    # 归一化亮度通道
    y_normalized = (y.astype(np.float32) - Lmin) / (Lmax - Lmin)
    y_normalized = np.clip(y_normalized, 0, 1)
    
    # 计算当前平均亮度
    avg_uni = np.mean(y_normalized)
    
    # 计算gamma值
    if method == 'formula':
        gamma = calculate_gamma_formula(avg_uni, target)
    elif method == 'newton':
        gamma = calculate_gamma_newton(y_normalized, target)
    else:  # 二分法
        gamma = calculate_gamma_binary(y_normalized, target)





    # 应用Gamma校正
    y_corrected = (y_normalized ** gamma) * (Lmax - Lmin) + Lmin
    y_corrected = np.clip(y_corrected, 0, 255).astype(np.uint8)

    # 合并通道并转换回BGR
    if color_space == 'YCbCr':
        merged = cv2.merge([y_corrected, cr, cb])
        result = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
    elif color_space == 'RGB':
        result = cv2.cvtColor(y_corrected, cv2.COLOR_GRAY2BGR)
    elif color_space == 'HSV':
        merged = cv2.merge([h, s, y_corrected])
        result = cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)
    else:
        result = y_corrected

    return result, gamma

def calculate_gamma_formula(avg_uni, target=0.5):
    """公式法：gamma = log(target) / log(avg_uni)"""
    eps = 1e-5
    avg_uni = max(avg_uni, eps)
    gamma = np.log(target) / np.log(avg_uni)
    return np.clip(gamma, 0.1, 10.0)

def calculate_gamma_newton(y_normalized, target=0.5, max_iter=20, tol=1e-6):
    """牛顿迭代法求解最优Gamma值"""
    gamma = 1.0
    for _ in range(max_iter):
        corrected_avg = np.mean(y_normalized ** gamma)
        f = corrected_avg - target
        if abs(f) < tol:
            break
        df = np.mean((y_normalized ** gamma) * np.log(np.clip(y_normalized, 1e-10, 1.0)))
        if abs(df) < 1e-10:
            break
        gamma = gamma - f / df
        gamma = np.clip(gamma, 0.1, 10.0)
    return gamma

def calculate_gamma_binary(y_normalized, target=0.5, max_iter=100, tol=1e-5):
    """二分法求解Gamma值"""
    avg_uni = np.mean(y_normalized)
    if np.isclose(avg_uni, target):
        return 1.0
    
    low, high = (0.1, 1.0) if avg_uni < target else (1.0, 10.0)
    for _ in range(max_iter):
        mid = (low + high) / 2
        corrected_avg = np.mean(y_normalized ** mid)
        error = corrected_avg - target
        if abs(error) < tol:
            break
        if error < 0:
            high = mid
        else:
            low = mid
    return mid

def show_results(original, corrected, gamma, save_dir):
    """显示和保存结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存处理后的图像
    cv2.imwrite(f"{save_dir}/output.jpg", corrected)
    
    # 计算统计信息
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    corr_gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    orig_mean = np.mean(orig_gray) / 255.0
    corr_mean = np.mean(corr_gray) / 255.0
    
    print(f"原图平均亮度: {orig_mean:.4f}")
    print(f"校正后平均亮度: {corr_mean:.4f}")
    print(f"使用的Gamma值: {gamma:.4f}")
    
    # 1. 对比图
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
    plt.title(f"Corrected (gamma={gamma:.2f})")
    plt.axis('off')
    plt.savefig(f"{save_dir}/comparison.png", dpi=300)
    
    # 2. Gamma曲线
    plt.figure(figsize=(8, 6))
    x = np.linspace(0, 1, 100)
    y = x ** gamma
    plt.plot(x, y, 'r-', linewidth=2)
    plt.plot(x, x, 'k--', linewidth=1)
    plt.title(f"Gamma Correction Curve (gamma={gamma:.2f})")
    plt.xlabel("Input Brightness")
    plt.ylabel("Output Brightness")
    plt.grid(True)
    plt.savefig(f"{save_dir}/gamma_curve.png", dpi=300)
    
    # 3. 直方图对比
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.hist(orig_gray.ravel(), 256, [0,256], color='blue', alpha=0.7)
    plt.axvline(np.mean(orig_gray), color='r', linestyle='dashed', linewidth=2)
    plt.title(f"Original Histogram (Avg: {orig_mean:.4f})")
    
    plt.subplot(122)
    plt.hist(corr_gray.ravel(), 256, [0,256], color='green', alpha=0.7)
    plt.axvline(np.mean(corr_gray), color='r', linestyle='dashed', linewidth=2)
    plt.title(f"Corrected Histogram (Avg: {corr_mean:.4f})")
    plt.savefig(f"{save_dir}/histogram_comparison.png", dpi=300)

def compare_methods(image_path, save_dir):
    """比较不同Gamma计算方法的效果"""
    original = cv2.imread(image_path)
    methods = ['binary', 'formula', 'newton']
    
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis('off')
    
    for i, method in enumerate(methods, start=2):
        corrected, gamma = adaptive_gamma_correction(original, 0.5, 'YCbCr', method)
        plt.subplot(2, 2, i)
        plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
        plt.title(f"{method.capitalize()} Method (gamma={gamma:.2f})")
        plt.axis('off')
    
    plt.savefig(f"{save_dir}/method_comparison.png", dpi=300)

def compare_targets(image_path, save_dir):
    """比较不同目标亮度值的效果"""
    original = cv2.imread(image_path)
    targets = [0.3, 0.5, 0.7]
    
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis('off')
    
    for i, target in enumerate(targets, start=2):
        corrected, gamma = adaptive_gamma_correction(original, target)
        plt.subplot(2, 2, i)
        plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
        plt.title(f"Target: {target} (gamma={gamma:.2f})")
        plt.axis('off')
    
    plt.savefig(f"{save_dir}/target_comparison.png", dpi=300)

if __name__ == "__main__":
    input_path = 'input.jpg'
    results_dir = "gamma_results"
    
    # 读取原图
    original = cv2.imread(input_path)
    
    # 基本Gamma校正
    corrected, gamma = adaptive_gamma_correction(input_path, target=0.5, color_space='YCbCr')
    
    # 显示结果
    show_results(original, corrected, gamma, results_dir)
    
    # 比较不同方法和目标亮度
    compare_methods(input_path, results_dir)
    compare_targets(input_path, results_dir)
    
    print(f"所有结果已保存到 {results_dir} 目录") 