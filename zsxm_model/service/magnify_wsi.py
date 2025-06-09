import io
import json
import base64
import threading
import os.path as osp
from PIL import Image
from flask import Flask, request, jsonify
from openslide import OpenSlide
from kfb.kfbreader import KFBSlide


# --- 配置 ---
WSI_MAPPING_FILE = 'wsi_mapping.json' # WSI 映射文件的路径
CROP_COOR_FILE = 'crop_coordinate.json' # 从WSI局部获取的缩略图的crop坐标文件
DEFAULT_PATCH_LEVEL = 1 # 默认读取的 WSI 层级，从0开始
POINT_PATCH_SIZE = (28*18, 504) # 返回的 patch 的最大尺寸 (宽度, 高度)
NORMALIZATION_RANGE = 1000 # 输入坐标的归一化范围 (0-1000)

app = Flask(__name__)


# 加载 WSI 缩略图与原始图路径的映射字典
class WSIMapping():
    def __init__(self, json_path=WSI_MAPPING_FILE):
        with open(json_path, 'r') as f:
            self.wsi_mapping = json.load(f)
        for wsi in self.wsi_mapping.values():
            if not osp.exists(wsi):
                raise FileNotFoundError(f"WSI 文件不存在: {wsi}")
        self._lock = threading.Lock() # 添加锁
            
    def __contains__(self, key):
        return key in self.wsi_mapping
            
    def __getitem__(self, key):
        slide_tuple = self.wsi_mapping[key] # 先在锁外快速检查，减少锁竞争
        if isinstance(slide_tuple, str):
            # 如果是路径，需要获取锁进行加载检查和更新
            with self._lock:
                # 在锁内再次检查，防止等待锁时其他线程已加载 (Double-Checked Locking)
                slide_tuple = self.wsi_mapping[key]
                if isinstance(slide_tuple, str):
                    slide_object = get_slide(slide_tuple)
                    app.logger.info(f'线程{threading.get_ident()}从缩略图"{key}"加载了WSI: {self.wsi_mapping[key]}')
                    self.wsi_mapping[key] = slide_tuple = (slide_tuple, slide_object)
        return slide_tuple

wsi_mapping = WSIMapping()  # 启动时加载映射表

with open(CROP_COOR_FILE, 'r') as f:
    crop_coor_map = json.load(f)

# --- WSI 读取辅助函数 ---
def get_slide(wsi_path):
    ext = osp.splitext(wsi_path)[1].lower()
    if ext in ['.svs', '.tif', '.tiff', '.mrxs']:
        slide = OpenSlide(wsi_path)
    elif ext in ['.kfb']:
        slide = KFBSlide(wsi_path)
    else:
        raise ValueError(f'不支持的WSI后缀名: {wsi_path}')
    return slide

def read_region(slide, location, level, size, zero_level_loc=False) -> Image.Image:
    ratio = slide.level_downsamples[level] / slide.level_downsamples[0]
    if isinstance(slide, KFBSlide):
        if zero_level_loc:
            return Image.fromarray(slide.read_region((round(location[0]/ratio), round(location[1]/ratio)), level, size))
        return Image.fromarray(slide.read_region(location, level, size))
    elif isinstance(slide, OpenSlide):
        if zero_level_loc:
            return slide.read_region(location, level, size)
        return slide.read_region((round(location[0]*ratio), round(location[1]*ratio)), level, size)
    else:
        raise ValueError(f'不支持的slide类型: {type(slide)}')


# --- 图像编码和坐标转换 ---
def encode_image_to_base64(image: Image.Image, format: str = "png") -> str:
    """将 PIL Image 对象编码为 Base64 字符串，并清除 ICC/iCCP 元信息"""
    buffer = io.BytesIO()
    
    # 创建一个不带 profile 的干净图像
    clean_img = Image.new("RGB", image.size)
    clean_img.paste(image)

    # 保存成无 metadata 的图像
    clean_img.save(buffer, format=format.upper())

    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def convert_normalized_to_level(coord, level_dimension):
    """
    将 0-1000 范围的归一化坐标转换为 WSI level 的坐标
    考虑到WSI缩略图都被处理成了正方形，而原始WSI是长方形，因此需要调整偏移量
    同时也需要注意，在取训练用的缩略图数据时，要避免取那种从原始WSI中取一部分作为缩略图的样本！
    """
    max_edge = max(level_dimension)
    x_offset, y_offset = (max_edge-level_dimension[0])//2, (max_edge-level_dimension[1])//2
    return [round(c/NORMALIZATION_RANGE*max_edge - (y_offset if i%2 else x_offset)) for i, c in enumerate(coord)]


# --- API 函数处理逻辑 ---
def handle_highres_request(arguments, input_images, mode):
    """处理 'get_highres_by_point' 和 'get_highres_by_bbox' 的通用逻辑"""
    # 输入验证
    if not isinstance(arguments, dict):
        return {"status": "error", "content": f"The function parameter must be a dictionary, not: {type(arguments)}"}, 200
    if isinstance(input_images, (str, dict)):
        input_images = [input_images]
    try:
        wsi_index = arguments['wsi_index']
    except KeyError:
        return {"status": "error", "content": "The parameter 'wsi_index' is required."}, 200
    if not isinstance(wsi_index, int) or wsi_index < 0:
        return {"status": "error", "content": f"The parameter 'wsi_index' must be a non-negative integer, not: {wsi_index}"}, 200
    
    valid_wsi_thumbnails = []
    for img in input_images:
        if isinstance(img, dict):
            img = img['path']
        if img in wsi_mapping:
            valid_wsi_thumbnails.append(img)
    if len(valid_wsi_thumbnails) == 0:
        # 当提供了输入中的图像，但是没有找到任何有效的 WSI 缩略图对应的原始WSI。
        # 这里假设所有的WSI缩略图都事先另外验证了配对性，如果是WSI缩略图无法从映射中找到，则返回此结果是错误的，因为这是代码的异常而不是模型生成结果的异常。
        if len(input_images) > 0: 
            return {"status": "error", "content": "None of the input images are valid WSI thumbnails, you should not call this function."}, 200
        else:
            return {"status": "error", "content": "Your input contains no images. This function cannot be called without image."}, 200
    if wsi_index >= len(valid_wsi_thumbnails):
        return {"status": "error", "content": f"There are {len(valid_wsi_thumbnails)} WSI thumbnails available, but wsi_index={wsi_index} is out of range."}, 200
    
    if mode == 'point':
        coords_list = arguments.get('point_list')
        if not isinstance(coords_list, list) or not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in coords_list):
            return {"status": "error", "content": "The parameter 'point_list' must be a list of [x, y] coordinate pairs."}, 200
    elif mode == 'bbox':
        coords_list = arguments.get('bbox_list')
        if not isinstance(coords_list, list) or not all(isinstance(b, (list, tuple)) and len(b) == 4 for b in coords_list):
            return {"status": "error", "content": "The parameter 'bbox_list' must be a list of [x1, y1, x2, y2] bounding boxes."}, 200
    else:
        raise ValueError("内部错误：无效的处理模式：{mode}")
    
    if not coords_list:
        return {"status": "error", "content": f"The parameter '{mode}_list' must not be an empty list."}, 200
    elif not all(isinstance(c, (float, int)) for co in coords_list for c in co):
        return {"status": "error", "content": f"The coordinates of '{mode}_list' must be numbers."}, 200
    elif not all(0 <= c <= NORMALIZATION_RANGE for co in coords_list for c in co):
        return {"status": "error", "content": f"The coordinates of '{mode}_list' must be in the range [0, {NORMALIZATION_RANGE}]"}, 200

    # 获取WSI对象
    thumbnail_id = valid_wsi_thumbnails[wsi_index]
    slide_path, slide = wsi_mapping[thumbnail_id] # 直接从映射中获取 WSI 对象
    
    # 读取Patch
    base64_images, content = [], ""
    for i, coord in enumerate(coords_list):
        # 最多5张图
        if i >= 4:
            content = "Too many patches requested, only the first 4 will be returned: "
            break
        if mode == 'point':
            if slide_path not in crop_coor_map:
                cx, cy = convert_normalized_to_level(coord, slide.level_dimensions[DEFAULT_PATCH_LEVEL])
            else:
                crop_x1, crop_y1, crop_x2, crop_y2 = crop_coor_map[slide_path]
                coor_ratio = slide.level_downsamples[DEFAULT_PATCH_LEVEL] / slide.level_downsamples[0]
                crop_x1, crop_y1, crop_x2, crop_y2 = int(crop_x1/coor_ratio), int(crop_y1/coor_ratio), int(crop_x2/coor_ratio), int(crop_y2/coor_ratio)
                cx, cy = convert_normalized_to_level(coord, (crop_x2-crop_x1, crop_y2-crop_y1))
                cx, cy = cx + crop_x1, cy + crop_y1
            x, y = cx - POINT_PATCH_SIZE[0] // 2, cy - POINT_PATCH_SIZE[1] // 2
            location = (min(max(0, x), slide.level_dimensions[DEFAULT_PATCH_LEVEL][0] - POINT_PATCH_SIZE[0]),
                        min(max(0, y), slide.level_dimensions[DEFAULT_PATCH_LEVEL][1] - POINT_PATCH_SIZE[1])) # 基本边界检查
            patch_image = read_region(slide, location, DEFAULT_PATCH_LEVEL, POINT_PATCH_SIZE, zero_level_loc=False)
        if mode == 'bbox':
            if slide_path not in crop_coor_map:
                x1_0, y1_0, x2_0, y2_0 = convert_normalized_to_level(coord, slide.dimensions)
            else:
                crop_x1, crop_y1, crop_x2, crop_y2 = crop_coor_map[slide_path]
                x1_0, y1_0, x2_0, y2_0 = convert_normalized_to_level(coord, (crop_x2-crop_x1, crop_y2-crop_y1))
                x1_0, y1_0, x2_0, y2_0 = x1_0 + crop_x1, y1_0 + crop_y1, x2_0 + crop_x1, y2_0 + crop_y1
            if x2_0 <= x1_0 or y2_0 <= y1_0:
                return {"status": "error", "content": f"Invalid bounding box coordinates: {coord}"}, 200
            ratio = max((x2_0-x1_0)/POINT_PATCH_SIZE[0], (y2_0-y1_0)/POINT_PATCH_SIZE[1])
            best_level = slide.get_best_level_for_downsample(ratio)
            if slide_path not in crop_coor_map:
                x1, y1, x2, y2 = convert_normalized_to_level(coord, slide.level_dimensions[best_level])
            else:
                crop_x1, crop_y1, crop_x2, crop_y2 = crop_coor_map[slide_path]
                coor_ratio = slide.level_downsamples[best_level] / slide.level_downsamples[0]
                crop_x1, crop_y1, crop_x2, crop_y2 = int(crop_x1/coor_ratio), int(crop_y1/coor_ratio), int(crop_x2/coor_ratio), int(crop_y2/coor_ratio)
                x1, y1, x2, y2 = convert_normalized_to_level(coord, (crop_x2-crop_x1, crop_y2-crop_y1))
                x1, y1, x2, y2 = x1 + crop_x1, y1 + crop_y1, x2 + crop_x1, y2 + crop_y1
            x1 = min(max(0, x1), slide.level_dimensions[best_level][0]-2)
            y1 = min(max(0, y1), slide.level_dimensions[best_level][1]-2)
            x2 = min(max(0, x2), slide.level_dimensions[best_level][0]-1)
            y2 = min(max(0, y2), slide.level_dimensions[best_level][1]-1)
            patch_image = read_region(slide, (x1, y1), best_level, (x2-x1, y2-y1), zero_level_loc=False)
            patch_image.thumbnail(POINT_PATCH_SIZE, Image.Resampling.LANCZOS) # 缩放到指定大小

        # 编码图像
        base64_img = encode_image_to_base64(patch_image, format="png")
        base64_images.append(base64_img)

    return {
        "status": "success",
        "content": content + "\n".join(["<image>"] * len(base64_images)),
        "images": base64_images,
        "image_format": "png"
    }, 200


# --- 注册允许的函数及其处理逻辑 ---
ALLOWED_FUNCTIONS = {
    "get_highres_by_bbox": lambda args, imgs: handle_highres_request(args, imgs, mode='bbox'),
    "get_highres_by_point": lambda args, imgs: handle_highres_request(args, imgs, mode='point'),
}


# --- Flask 路由 ---
@app.route('/magnifywsi', methods=['POST'])
def handle_api_call():
    if not request.is_json:
        return jsonify({"status": "exception", "content": "请求必须是 JSON 格式"}), 400

    data = request.get_json()
    function_name = data.get('function_name')
    arguments = data.get('arguments')
    input_images = data.get('input_images')

    # 基本的输入检查
    if not function_name or not isinstance(function_name, str):
         return jsonify({"status": "exception", "content": f"缺少或无效的 {function_name = }"}), 400
    if arguments is None: # arguments 可以是空字典 {}
         return jsonify({"status": "exception", "content": "缺少 'arguments'"}), 400
    if input_images is None: # input_images 可以是空列表 []，但在函数逻辑中需要处理
         return jsonify({"status": "exception", "content": "缺少 'input_images'"}), 400


    # 检查函数是否被允许
    if function_name not in ALLOWED_FUNCTIONS:
        return jsonify({"status": "error", "content": f"The function name you provided is invalid: {function_name}"}), 200

    # 获取处理函数
    target_function = ALLOWED_FUNCTIONS[function_name]

    # 执行函数并获取结果
    # 函数内部会处理所有逻辑和错误，并返回符合格式的字典和状态码
    result, http_status_code = target_function(arguments, input_images)

    return jsonify(result), http_status_code


if __name__ == '__main__':
    # 启动服务器
    app.run(debug=False, host='0.0.0.0', port=18888, threaded=True)