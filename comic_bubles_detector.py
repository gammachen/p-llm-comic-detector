import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
from PIL import Image

class ComicTextOCR:
    # def __init__(self, languages=['zh', 'ja', 'en']):
    def __init__(self, languages=['zh']):
        """
        初始化漫画文字识别器
        
        参数:
            languages: 支持的语言列表，默认中文
        """
        self.reader = easyocr.Reader(languages, gpu=False)  # 初始化EasyOCR:cite[5]:cite[7]
    
    def preprocess_image(self, image_path):
        """
        图像预处理：增强气泡与背景的对比度
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 二值化处理 - 使用Otsu自适应阈值
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return image, binary
    
    def detect_speech_bubbles(self, binary_image, original_image, min_area=1000):
        """
        检测漫画中的气泡区域
        """
        # 寻找轮廓:cite[4]:cite[8]
        contours, hierarchy = cv2.findContours(
            binary_image, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        bubbles = []
        bubble_coordinates = []
        
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 过滤掉太小的区域
            if area < min_area:
                continue
                
            # 获取轮廓的边界矩形
            x, y, w, h = cv2.boundingRect(contour)
            
            # 计算轮廓的圆形度（判断是否接近气泡形状）
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # 基于面积和圆形度筛选气泡
            if area > min_area and circularity > 0.3:
                bubble_roi = original_image[y:y+h, x:x+w]
                bubbles.append({
                    'roi': bubble_roi,
                    'coordinates': (x, y, w, h),
                    'contour': contour
                })
                bubble_coordinates.append((x, y, w, h))
        
        return bubbles, bubble_coordinates
    
    def extract_text_from_bubble(self, bubble_roi):
        """
        从气泡区域提取文字
        """
        try:
            # 使用EasyOCR识别文字:cite[5]:cite[7]
            results = self.reader.readtext(bubble_roi, detail=1)
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
        
        print(f"检测到 {len(bubbles)} 个气泡")
        
        all_results = []
        
        # 处理每个气泡
        for i, bubble in enumerate(bubbles):
            x, y, w, h = bubble['coordinates']
            
            # 提取文字
            text_results = self.extract_text_from_bubble(bubble['roi'])
            
            # 在结果图像上绘制气泡边界框
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            bubble_texts = []
            # 处理识别到的文字
            for text_info in text_results:
                bbox, text, confidence = text_info
                
                if confidence > 0.3:  # 置信度阈值
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

# 使用示例
if __name__ == "__main__":
    # 初始化OCR处理器
    comic_ocr = ComicTextOCR(languages=['ja', 'en'])  # 日语和英语
    
    # 处理漫画图像
    image_path = "comic_image.jpg"  # 替换为你的图像路径
    output_path = "result_image.jpg"
    
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