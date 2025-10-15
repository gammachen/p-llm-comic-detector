import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
from PIL import Image

class ComicTextOCR:
    def __init__(self, languages=['ch_sim']):
        """
        初始化漫画文字识别器
        
        参数:
            languages: 支持的语言列表，默认简体中文
        """
        self.reader = easyocr.Reader(languages, gpu=False)  # 初始化EasyOCR
    
    # 修改为使用EasyOCR支持的中文语言代码
    def preprocess_image(self, image_path):
        """
        图像预处理：增强气泡与背景的对比度
        保存各处理步骤的中间结果以便可视化
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 保存原始图像
        cv2.imwrite("step_0_original.jpg", image)
        print("已保存原始图像: step_0_original.jpg")
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("step_1_gray.jpg", gray)
        print("已保存灰度图像: step_1_gray.jpg")
        
        # 使用高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        cv2.imwrite("step_2_blurred.jpg", blurred)
        print("已保存模糊图像: step_2_blurred.jpg")
        
        # 改进的二值化处理 - 自适应阈值
        binary = cv2.adaptiveThreshold(
            blurred, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 
            2
        )
        cv2.imwrite("step_3_binary.jpg", binary)
        print("已保存二值化图像: step_3_binary.jpg")
        
        # 形态学操作：闭操作填充小空洞，开操作移除小物体
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite("step_4_closed.jpg", closed)
        print("已保存闭操作后图像: step_4_closed.jpg")
        
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        cv2.imwrite("step_5_opened.jpg", opened)
        print("已保存开操作后图像: step_5_opened.jpg")
        
        return image, opened
    
    def detect_speech_bubbles(self, binary_image, original_image, min_area=300):
        """
        检测漫画中的气泡区域，改进版本
        """
        # 寻找轮廓，包括内部轮廓
        contours, hierarchy = cv2.findContours(
            binary_image, 
            cv2.RETR_TREE, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        bubbles = []
        bubble_coordinates = []
        
        for i, contour in enumerate(contours):
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 过滤掉太小的区域，但降低阈值以捕获小气泡
            if area < min_area:
                continue
                
            # 获取轮廓的边界矩形
            x, y, w, h = cv2.boundingRect(contour)
            
            # 计算宽高比，气泡通常不会太窄或太宽
            aspect_ratio = float(w) / h
            if aspect_ratio > 5 or aspect_ratio < 0.2:
                continue
                
            # 计算轮廓的圆形度
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # 计算填充度（轮廓面积与边界矩形面积的比率）
            rect_area = w * h
            fill_ratio = area / rect_area
            
            # 更灵活的筛选条件：降低圆形度要求，增加填充度考虑
            # 气泡通常是浅色背景，检查ROI内的像素均值
            roi = original_image[y:y+h, x:x+w]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(roi_gray)
            
            # 气泡通常是亮色区域，或者具有特定的轮廓特征
            if ((circularity > 0.15 and fill_ratio > 0.2) or 
                (circularity > 0.1 and fill_ratio > 0.3 and mean_brightness > 100)):
                
                # 检查是否是嵌套轮廓（气泡内的文字区域不应被当作独立气泡）
                if hierarchy[0][i][3] != -1:  # 有父轮廓
                    continue
                    
                bubbles.append({
                    'roi': roi,
                    'coordinates': (x, y, w, h),
                    'contour': contour,
                    'brightness': mean_brightness
                })
                bubble_coordinates.append((x, y, w, h))
        
        return bubbles, bubble_coordinates
    
    def extract_text_from_bubble(self, bubble_roi):
        """
        从气泡区域提取文字，优化版本
        """
        try:
            # 对气泡区域进行额外预处理以提高OCR准确率
            # 转换为灰度
            roi_gray = cv2.cvtColor(bubble_roi, cv2.COLOR_BGR2GRAY)
            
            # 应用自适应阈值
            roi_binary = cv2.adaptiveThreshold(
                roi_gray, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                11, 
                2
            )
            
            # 使用EasyOCR识别文字，调整参数以提高准确率
            results = self.reader.readtext(
                roi_binary, 
                detail=1,
                paragraph=False,
                min_size=10,
                contrast_ths=0.1,
                adjust_contrast=0.5
            )
            return results
        except Exception as e:
            print(f"文字识别错误: {e}")
            return []
    
    def process_comic_image(self, image_path, output_path=None):
        """
        处理漫画图像：检测气泡并识别文字
        """
        # 图像预处理
        original_image, binary_image = self.preprocess_image(image_path)
        result_image = original_image.copy()
        
        # 检测气泡
        bubbles, bubble_coordinates = self.detect_speech_bubbles(binary_image, original_image)
        
        # 对于漫画，我们还可以尝试另一种方法：检测对话框的边框线
        # 创建一个副本用于边缘检测
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 检测直线（对话框的边框）
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        
        # 寻找可能的对话框矩形区域
        if lines is not None:
            # 这里简化处理，实际应用可能需要更复杂的矩形检测算法
            pass  # 可以扩展此部分以结合直线检测和轮廓检测的结果
        
        print(f"检测到 {len(bubbles)} 个气泡")
        
        all_results = []
        
        # 处理每个气泡
        for i, bubble in enumerate(bubbles):
            x, y, w, h = bubble['coordinates']
            
            # 提取文字 - 优化OCR参数
            text_results = self.extract_text_from_bubble(bubble['roi'])
            
            # 在结果图像上绘制气泡边界框和轮廓
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.drawContours(result_image, [bubble['contour']], -1, (0, 0, 255), 2)
            
            bubble_texts = []
            # 处理识别到的文字
            for text_info in text_results:
                bbox, text, confidence = text_info
                
                if confidence > 0.1:  # 置信度阈值
                    # 转换文字坐标到原图坐标系
                    original_bbox = []
                    for point in bbox:
                        px, py = point
                        original_bbox.append([px + x, py + y])
                    
                    # 绘制文字边界框
                    pts = np.array(original_bbox, np.int32)
                    cv2.polylines(result_image, [pts], True, (255, 0, 0), 2)
                    
                    # 添加文字标签
                    cv2.putText(result_image, text, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    bubble_texts.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': original_bbox
                    })
                    
                    print(f"气泡 {i+1}: 文字='{text}', 置信度={confidence:.2f}")
            
            all_results.append({
                'bubble_id': i+1,
                'coordinates': (x, y, w, h),
                'texts': bubble_texts
            })
        
        # 保存或显示结果
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"结果图像已保存: {output_path}")
        
        # 显示结果
        self.display_result(original_image, result_image)
        
        # 创建所有处理步骤的可视化组合图（确保结果图像已保存后再生成）
        self.visualize_processing_steps()
        
        return all_results, result_image
    
    def display_result(self, original_image, result_image):
        """
        显示原始图像和结果图像
        """
        plt.figure(figsize=(15, 10))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title('原始图像')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title('检测结果')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 保存组合显示图
        plt.savefig("processing_comparison.jpg")
        print("已保存处理比较图: processing_comparison.jpg")
    
    def visualize_processing_steps(self):
        """
        可视化所有处理步骤的中间结果，包括最终检测结果
        """
        try:
            # 读取所有中间步骤图像和最终结果图像
            step_files = [
                "step_0_original.jpg",
                "step_1_gray.jpg",
                "step_2_blurred.jpg",
                "step_3_binary.jpg",
                "step_4_closed.jpg",
                "step_5_opened.jpg",
                "detected_result.png"  # 添加最终检测结果图像
            ]
            
            step_titles = [
                "Original Image",
                "Gray Conversion",
                "Gaussian Blur",
                "Binary Thresholding",
                "Closing Operation",
                "Opening Operation",
                "Final Detection Result"  # 最终结果标题
            ]
            
            # 创建一个大图展示所有步骤，调整布局为3行3列以容纳最终结果
            plt.figure(figsize=(24, 20))
            
            for i, (file, title) in enumerate(zip(step_files, step_titles)):
                try:
                    img = cv2.imread(file)
                    if img is not None:
                        plt.subplot(3, 3, i+1)
                        if i == 0 or i == 6:  # 原始图像和最终结果使用RGB显示
                            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        else:
                            # 对于灰度图，使用灰度颜色映射
                            if i == 1 or i == 2:
                                plt.imshow(img, cmap='gray')
                            else:
                                # 对于二值化和形态学操作结果，使用热力图颜色映射
                                plt.imshow(img, cmap='hot')
                        plt.title(title, fontsize=14)
                        plt.axis('off')
                except Exception as e:
                    print(f"读取步骤 {i+1} 图像时出错: {e}")
            
            plt.tight_layout()
            plt.savefig("all_processing_steps.jpg")
            print("已保存所有处理步骤的组合图: all_processing_steps.jpg")
            plt.show()
            
        except Exception as e:
            print(f"可视化处理步骤时出错: {e}")

# 修复主程序块的缩进问题
# 使用示例
if __name__ == "__main__":
    # 初始化OCR处理器 - 调整语言设置以适应EasyOCR的限制
    # 中文简体只能与英语组合，不能同时与日语组合
    comic_ocr = ComicTextOCR(languages=['ch_sim', 'en'])  # 简体中文和英语
    
    # 处理漫画图像
    image_path = "comic_image.jpg"  # 替换为你的图像路径
    output_path = "detected_result.png"

    try:
        results, result_image = comic_ocr.process_comic_image(image_path, output_path)
        
        # 打印详细结果
        print("\n" + "="*50)
        print("识别结果汇总:")
        print("="*50)
        
        for result in results:
            print(f"气泡 {result['bubble_id']}:")
            print(f"  坐标: (x={result['coordinates'][0]}, y={result['coordinates'][1]}, "
                  f"宽={result['coordinates'][2]}, 高={result['coordinates'][3]})")
            if result['texts']:
                for text_info in result['texts']:
                    print(f"  文字: '{text_info['text']}' (置信度: {text_info['confidence']:.2f})")
            else:
                print("  文字: 未识别到文字")
            print()
            
    except Exception as e:
        print(f"处理错误: {e}")