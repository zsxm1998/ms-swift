import re
import ast
import math
import os.path as osp
import xml.etree.ElementTree as ET
from typing import List
from shapely.geometry import Polygon
from shapely.ops import unary_union


from swift.plugin import ORM, orms


def think_format_with_suffix(text):
    # 匹配“<think>xxx</think>yyy<answer>zzz</answer>suffix”，其中xxx、zzz不可为空，yyy可为空，suffix可空可有但不能包含标签
    full_pattern = re.compile(
        r'^<think>((?:(?!<think>|</think>|<answer>|</answer>).)+)</think>'   # think 内容：不能有嵌套标签
        r'(.*?)'                                                             # think 和 answer 之间任意内容
        r'<answer>((?:(?!<think>|</think>|<answer>|</answer>).)+)</answer>'  # answer 内容不能包含任何标签符号
        r'(?!.*<think>|.*<answer>|.*</think>|.*</answer>)'                   # 后缀不能再包含标签
        , re.DOTALL
    )

    match = full_pattern.match(text)
    if not match:
        return False

    # 确保 think 和 answer 内容非空
    think_content = match.group(1).strip()
    answer_content = match.group(3).strip()
    if not think_content or not answer_content:
        return False

    # 用于统计是否只出现一次这种结构
    loose_pattern = re.compile(
        r'<think>((?:(?!<think>|</think>|<answer>|</answer>).)+)</think>'
        r'(.*?)'
        r'<answer>((?:(?!<think>|</think>|<answer>|</answer>).)+)</answer>',
        re.DOTALL
    )

    # 确保格式只出现一次
    all_matches = loose_pattern.findall(text)
    if len(all_matches) != 1:
        return False

    return True


def think_format_no_suffix(text):
    # 匹配“<think>xxx</think>yyy<answer>zzz</answer>”，其中xxx、zzz不可为空，yyy可为空
    full_pattern = re.compile(
        r'^<think>((?:(?!<think>|</think>|<answer>|</answer>).)+)</think>'      # think 内容：不能有嵌套标签
        r'(.*?)'                                                                # 任意中间内容
        r'<answer>((?:(?!<think>|</think>|<answer>|</answer>).)+)</answer>$',   # answer 内容：不能有嵌套标签
        re.DOTALL
    )

    match = full_pattern.match(text)
    if not match:
        return False

    # 确保 think 和 answer 内容非空
    think_content = match.group(1).strip()
    answer_content = match.group(3).strip()
    if not think_content or not answer_content:
        return False

    # 用于统计是否只出现一次这种结构
    loose_pattern = re.compile(
        r'<think>((?:(?!<think>|</think>|<answer>|</answer>).)+)</think>'
        r'(.*?)'
        r'<answer>((?:(?!<think>|</think>|<answer>|</answer>).)+)</answer>',
        re.DOTALL
    )

    # 确保格式只出现一次
    all_matches = loose_pattern.findall(text)
    if len(all_matches) != 1:
        return False

    return True


def check_negative_exist(text, fragment_list=None):
    if fragment_list is None:
        fragment_list = [
            # 英文否定
            'no.', 'no,', 'no ', 'not ', 'none', 'nobody', 'nothing', 'neither', 'nor ',
            'cannot', "can't", "don't", "doesn't", "didn't", "won't", "wouldn't",
            "isn't", "aren't", "wasn't", "weren't", 'without',
            # 中文否定
            '未', '无', '没', '否', '不', '缺', '排除', '拒绝', '阴性',
        ]

    # 小写处理用于英文匹配
    text_lower = text.lower()
    
    for fragment in fragment_list:
        if fragment in text or fragment in text_lower:
            return True
    return False


def build_all_choice_prefixes(): # 构造所有可能的选项标记
    # 英文字母
    letter_choices = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + \
                     [chr(i) for i in range(ord('a'), ord('z') + 1)]

    # 数字
    digit_choices = list(map(str, range(1, 100)))

    # 罗马数字
    roman_numerals = [
        "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
        "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
        "XXI", "XXII", "XXIII", "XXIV", "XXV", "XXVI", "XXVII", "XXVIII", "XXIX", "XXX",
        "XXXI", "XXXII", "XXXIII", "XXXIV", "XXXV", "XXXVI", "XXXVII", "XXXVIII", "XXXIX", "XL",
        "XLI", "XLII", "XLIII", "XLIV", "XLV", "XLVI", "XLVII", "XLVIII", "XLIX", "L"
    ]
    roman_numerals += [r.lower() for r in roman_numerals]

    labels = letter_choices + digit_choices + roman_numerals

    extra_patterns = []
    for label in labels:
        escaped = re.escape(label)
        extra_patterns.extend([
            rf"^\(?{escaped}\)?[\.:)\s]",   # A. A) (A) A:
            rf"^Option\s+{escaped}:",       # Option 1:
            rf"^{escaped}\)$",              # 1)
            rf"^{escaped}$"                 # just the label
        ])
    return labels, extra_patterns


def is_choice_format(s): # 判断是否是选择项
    s = s.strip()
    _, patterns = build_all_choice_prefixes()
    combined = re.compile("|".join(patterns))
    return bool(combined.match(s))


def extract_choice_label(s): # 提取选项标签
    s = s.strip()
    if not is_choice_format(s):
        return False
    label_match = re.match(r"^(?:Option\s+)?\(?([A-Za-z0-9]+)\)?(?=[\.:)\s])?", s)
    return label_match.group(1) if label_match else False


def extract_between_tags(text, start_tag, end_tag, include_tags=False, return_origin=False):
    try:
        start_index = text.index(start_tag)
        end_index = text.index(end_tag, start_index + len(start_tag))
    except ValueError:
        if return_origin:
            return text
        return ""
    
    return text[start_index:end_index + len(end_tag)] if include_tags else text[start_index + len(start_tag):end_index]


def check_other_task_tag_exist(text, stag_set, exclude_tags=[]):
    """测试是否存在其他任务的标签"""
    if isinstance(exclude_tags, str):
        exclude_tags = [exclude_tags]
    for stag in stag_set - set(exclude_tags):
        if stag in text:
            return True
    return False


def has_chinese(text: str) -> bool:
    """判断字符串中是否包含中文字符。"""
    return re.search(r'[\u4e00-\u9fff]', text) is not None


def parse_bbox_string(bbox_str, key_word='box'):
    """
    从形如 <bbox_list><box>160, 312, 254, 414</box>...</bbox_list> 中提取bbox列表
    返回: list of [x1, y1, x2, y2]
    """
    try:
        box_strs = re.findall(rf"<{key_word}>(.*?)</{key_word}>", bbox_str)
        boxes = []
        for box_str in box_strs:
            parts = [int(p.strip()) for p in box_str.strip().split(",")]
            if len(parts) != 4:
                raise ValueError(f"Invalid box: {box_str}")
            if not all(0 <= x <= 1000 for x in parts):
                raise ValueError(f"Coordinates out of range in box: {box_str}")
            boxes.append(parts)
        return boxes
    except Exception as e:
        raise ValueError(f"Failed to parse bbox string: {e}")


def compute_iou(boxA, boxB):
    """计算两个框之间的IoU"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = areaA + areaB - interArea
    if unionArea == 0:
        return 0.0
    return interArea / unionArea


def compute_detection_reward(gt_boxes, pred_boxes):
    """计算基于F1 × avg_IoU 的 reward 值"""
    if not gt_boxes and not pred_boxes:
        return 1.0  # 两者都为空时，reward设为1

    matched_gt = set()
    matched_pred = set()
    match_scores = []

    # 计算所有IoU，并按IoU降序排列
    iou_pairs = []
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            iou = compute_iou(gt, pred)
            if iou >= 0.5:
                iou_pairs.append((iou, i, j))
    iou_pairs.sort(reverse=True)

    for iou, i, j in iou_pairs:
        if i not in matched_gt and j not in matched_pred:
            matched_gt.add(i)
            matched_pred.add(j)
            match_scores.append(iou)

    TP = len(match_scores)
    FP = len(pred_boxes) - TP
    FN = len(gt_boxes) - TP

    if TP == 0:
        return 0.0

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    avg_iou = sum(match_scores) / TP
    reward = f1 * avg_iou
    return reward


def det_no_class_reward(gt_res, content_res):
    """计算不带类别检测任务的reward"""
    if check_negative_exist(gt_res):
        gt_boxes = []
    else:
        gt_boxes = parse_bbox_string(gt_res)
    if check_negative_exist(content_res):
        pred_boxes = []
    else:
        pred_boxes = parse_bbox_string(content_res)
    return compute_detection_reward(gt_boxes, pred_boxes)


def extract_bbox_categories(xml_data):
    """
    提取出每类的 bbox 列表。
    返回字典：{类别: [bbox1, bbox2, ...]}
    """
    # Parse the XML-like structure in the answer field to extract categories and their corresponding bounding boxes
    xml_root = ET.fromstring(xml_data)
    # Extract each category and their bounding boxes
    category_boxes = {}
    for bbox_list in xml_root:
        category = bbox_list.attrib['class']
        boxes = [list(map(int, bbox.text.split(','))) for bbox in bbox_list]
        category_boxes[category] = boxes

    return category_boxes


def det_with_class_reward(gt_res, content_res, alpha=0.8):
    """
    多类别目标检测任务的reward计算，支持混合权重方式。
    
    参数:
        gt_res: 真实的多类别预测结果字符串
        content_res: 模型预测的多类别预测结果字符串
        alpha: float, ∈ [0, 1]，类别均值与数量加权的平衡参数，为1时为类别均值，为0时为数量加权
    
    返回:
        final_reward: float, ∈ [0, 1]
    """
    assert 0.0 <= alpha <= 1.0, "alpha must be between 0 and 1"

    if check_negative_exist(gt_res):
        gt_dict = {}
    else:
        gt_dict = extract_bbox_categories(gt_res)
    if check_negative_exist(content_res):
        pred_dict = {}
    else:
        pred_dict = extract_bbox_categories(content_res)
    
    all_categories = set(gt_dict.keys()) | set(pred_dict.keys())
    if len(all_categories) == 0:
        return 1.0
    reward_per_class = {}
    weight_per_class = {}

    total_gt_boxes = 0

    for category in all_categories:
        gt_boxes = gt_dict.get(category, [])
        pred_boxes = pred_dict.get(category, [])

        reward = compute_detection_reward(gt_boxes, pred_boxes)
        reward_per_class[category] = reward
        weight_per_class[category] = max(len(gt_boxes), 1)  # 至少为1，确保即使gt中无该类，仍有FP惩罚
        total_gt_boxes += len(gt_boxes)

    # 类别平均得分
    mean_reward = sum(reward_per_class.values()) / len(all_categories)

    # 数量加权得分
    total_weight = sum(weight_per_class.values())
    weighted_reward = sum(reward_per_class[c] * weight_per_class[c] for c in all_categories) / total_weight

    # 混合方式
    final_reward = alpha * mean_reward + (1 - alpha) * weighted_reward
    return final_reward


def parse_polygon_string(poly_str):
    """
    解析形如 <contour_list><polygon>[x1, y1], [x2, y2], ...</polygon>...</contour_list>
    的字符串，返回多边形列表
    """
    try:
        # 提取所有 <polygon>...</polygon>
        polygons_raw = re.findall(r"<polygon>(.*?)</polygon>", poly_str)
        polygons = []
        for poly_text in polygons_raw:
            points = ast.literal_eval("[" + poly_text + "]")  # 转为 list of [x, y]
            if not all(isinstance(pt, (list, tuple)) and len(pt) == 2
                       and isinstance(pt[0], int) and isinstance(pt[1], int) for pt in points):
                raise ValueError(f"Invalid polygon points: {poly_text}")
            try:
                polygon = Polygon(points)
                if not polygon.is_valid or polygon.area == 0:
                    continue  # 忽略无效或空区域
            except:
                continue  # 忽略不能构成多边形的点集
            polygons.append(points)

        return polygons

    except Exception as e:
        raise ValueError(f"Failed to parse polygons: {e}")


def seg_reward(gt_res, content_res, beta=0.5):
    """
    输入GT和预测的contour字符串，输出最终reward值（IoU × complexity_penalty）
    """
    if check_negative_exist(gt_res):
        gt_polygons = []
    else:
        gt_polygons = parse_polygon_string(gt_res)
    if check_negative_exist(content_res):
        pred_polygons = []
    else:
        pred_polygons = parse_polygon_string(content_res)
    
    # 构造 shapely MultiPolygon 对象
    gt_shapes = [Polygon(p) for p in gt_polygons]
    pred_shapes = [Polygon(p) for p in pred_polygons]

    if not gt_shapes and not pred_shapes:
        return 1.0  # 都为空，认为是完美匹配
    if not gt_shapes or not pred_shapes:
        return 0.0  # 有一边为空，完全不匹配

    gt_union = unary_union(gt_shapes)
    pred_union = unary_union(pred_shapes)

    intersection = gt_union.intersection(pred_union).area
    union = gt_union.union(pred_union).area
    if union == 0:
        iou = 0.0
    else:
        iou = intersection / union

    # 计算复杂度惩罚
    gt_points = sum(len(p) for p in gt_polygons)
    pred_points = sum(len(p) for p in pred_polygons)
    if pred_points <= gt_points or gt_points == 0:
        penalty = 1.0
    else:
        ratio = pred_points / gt_points
        penalty = math.exp(-beta * (ratio - 1))

    reward = iou * penalty
    return reward


image_organ_map = {'HCC_grading': 'liver', 'ICC_subtype': 'liver', 'liver_cancer': 'liver', 'liverWSI': 'liver',
                   'Lung1000': 'lung', 'Stomach1000': 'stomach', 'RCC': 'kidney', 'XiaJiabinWeiAi': 'stomach',}


def infer_organ_from_text(text: str) -> str:
    organ_keywords = {
        "liver": {
            "en": [
                "liver", "hepatic", "hepatocellular", "hcc", "hepatic lesion",
                "cholangiocarcinoma", "intrahepatic", "icc", "chcc-cca",
                "macrotrabecular", "pseudoglandular", "scirrhous",
                "large bile duct", "small bile duct", "combined hepatocellular-cholangiocarcinoma"
            ],
            "zh": [
                "肝", "粗梁型", "粗梁团块型", "假腺管型", "硬化型", "小胆管型", "大胆管型", "混合型肝癌"
            ]
        },
        "lung": {
            "en": [
                "lung", "pulmonary", "nsclc", "sclc", "bronchial", "adenocarcinoma of the lung",
                "keratinizing squamous", "non-keratinizing squamous", "basaloid carcinoma of lung"
            ],
            "zh": [
                "肺", "肺腺癌", "非小细胞", "小细胞", "鳞癌", "角化型", "非角化型", "基底样肺癌"
            ]
        },
        "stomach": {
            "en": [
                "stomach", "gastric", "gastroesophageal", "gastric cancer", "lauren classification",
                "intestinal type", "diffuse type", "mixed type"
            ],
            "zh": [
                "胃", "胃腺癌", "弥漫型", "肠型", "混合型", "lauren分型"
            ]
        },
        "kidney": {
            "en": [
                "kidney", "renal", "renal cell", "rcc",
                "clear cell", "ccrcc", "papillary renal cell", "prcc", "chromophobe", "chrcc", "collecting duct carcinoma"
            ],
            "zh": [
                "肾", "透明细胞癌", "乳头状肾癌", "嫌色细胞癌", "集合管癌"
            ]
        }
    }

    text = text.lower()
    matched_organs = []

    for organ, kw_dict in organ_keywords.items():
        for keyword in kw_dict["en"]:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                matched_organs.append(organ)
                break
        else:
            for keyword in kw_dict["zh"]:
                if keyword in text:
                    matched_organs.append(organ)
                    break

    if len(matched_organs) == 0:
        return "unknown"
    elif len(matched_organs) == 1:
        return matched_organs[0]
    else:
        return "unsure"


class PathORM(ORM):
    def __init__(self,
                 tokenizer=None,
                 cosine_min_len_value_wrong: float = 0.0,
                 cosine_max_len_value_wrong: float = 0.1,
                 cosine_min_len_value_correct: float = 1.0,
                 cosine_max_len_value_correct: float = 1.0,
                 cosine_max_len: int = 4096):
        self.tokenizer = tokenizer
        self.min_len_value_wrong = cosine_min_len_value_wrong
        self.max_len_value_wrong = cosine_max_len_value_wrong
        self.min_len_value_correct = cosine_min_len_value_correct
        self.max_len_value_correct = cosine_max_len_value_correct
        self.max_len = cosine_max_len
        self.task_setags = {
            'seg': ('<contour_list>', '</contour_list>'),
            'det_no_class': ('<bbox_list>', '</bbox_list>'),
            'det_with_class': ('<detection_result>', '</detection_result>')
        }
        self.stag_set = set(token for token, _ in self.task_setags.values())

    @staticmethod
    def cosfn(t, T, max_len_value, min_len_value):
        import math
        return min_len_value + (max_len_value - min_len_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions, solution, task, messages, **kwargs) -> List[float]:
        rewards = []
        for content, gt, task_type, msgs in zip(completions, solution, task, messages):
            stag, etag = self.task_setags.get(task_type, (None, None))
            format_reward, acc_reward = 0., 0.

            # 计算各个类别的format_reward
            if task_type in ['choice']:
                if think_format_with_suffix(content):
                    format_reward = 0.8
                    if think_format_no_suffix(content):
                        format_reward = 1.0
                else:
                    if content.count('<think>') == 1 and content.count('</think>') == 1 \
                        and content.index('<think>') < content.index('</think>'):
                        format_reward += 0.3
                    if content.count('<answer>') == 1 and content.count('</answer>') == 1 \
                        and content.index('<answer>') < content.index('</answer>'):
                        format_reward += 0.4
            elif task_type in ['seg', 'det_no_class', 'det_with_class']:
                format_reward = 0.1
                if content.count(stag) == 1 and content.count(etag) == 1 \
                    and content.index(stag) < content.index(etag):
                    format_reward = 1.0
                elif check_negative_exist(content):
                    format_reward = 1.0
                elif check_other_task_tag_exist(content, self.stag_set, stag):
                    format_reward = 0.0
            else:
                raise ValueError(f'task "{task_type}" not supported')
            
            # 计算各个类别的acc_reward
            try:
                if format_reward > 0.5: # 只有格式符合基本要求才计算acc_reward
                    if task_type in ['choice']:
                        # 提取gt和content中的选项标签
                        gt_choice = extract_choice_label(gt) # 选项序号或False
                        content_answer = extract_between_tags(content, '<answer>', '</answer>')
                        content_choice = extract_choice_label(content_answer) # 选项序号或False

                        # 若标签符合则赋予1.0的离散reward
                        is_correct = False
                        if gt_choice and content_choice and gt_choice == content_choice:
                            acc_reward = 1.0
                            is_correct = True
                        elif not gt_choice and check_negative_exist(gt) \
                            and not content_choice and check_negative_exist(content_answer):
                            acc_reward = 1.0
                            is_correct = True
                        
                        # 若tokenzier存在则计算长度reward
                        if self.tokenizer is not None:
                            if is_correct:
                                # Swap min/max for correct answers
                                max_len_value = self.max_len_value_correct
                                min_len_value = self.min_len_value_correct
                            else:
                                max_len_value = self.max_len_value_wrong
                                min_len_value = self.min_len_value_wrong
                            gen_len = len(self.tokenizer.encode(extract_between_tags(content, '<think>', '</think>')))
                            acc_reward = self.cosfn(gen_len, self.max_len, max_len_value, min_len_value)
                        
                        # 考虑思考语言对reward的影响
                        # language_inconsistency = False
                        # # 如果确定gt有中文，则content中的思考部分必须有中文
                        # if has_chinese(gt):
                        #     language_inconsistency = not has_chinese(extract_between_tags(content, '<think>', '</think>'))
                        # # 排除solution只有选项的情况，其他情况下若solution中没有中文则可以完全确认是英文
                        # elif not gt_choice or gt_choice != gt:
                        #     language_inconsistency = has_chinese(extract_between_tags(content, '<think>', '</think>'))
                        user_last_query = [m['content'] for m in msgs if m['role'] == 'user'][-1]
                        language_inconsistency = has_chinese(user_last_query) \
                            != has_chinese(content) #has_chinese(extract_between_tags(content, '<think>', '</think>'))
                        
                        # 如果存在语言不一致
                        if language_inconsistency:
                            if is_correct:
                                if self.tokenizer is not None:
                                    acc_reward = min(self.min_len_value_correct, self.max_len_value_correct) - 0.1
                                else:
                                    acc_reward = 0.9
                            else:
                                if self.tokenizer is not None:
                                    acc_reward = min(self.min_len_value_wrong, self.max_len_value_wrong) - 0.1
                                else:
                                    acc_reward = -0.1
                    elif task_type in ['seg', 'det_no_class', 'det_with_class']:
                        reward_func = globals().get(f"{task_type}_reward")
                        acc_reward = reward_func(
                            extract_between_tags(gt, stag, etag, include_tags=True, return_origin=True),
                            extract_between_tags(content, stag, etag, include_tags=True, return_origin=True)
                        )
            except:
                acc_reward = 0.0

            rewards.append(format_reward + acc_reward)
        
        return rewards
    

class PathORM_Organ(PathORM):
    """在PathORM的基础上，对选择任务，如果能确定数据的器官来源，则回答正确时检测思考过程中对图片器官的判断，若错误则acc reward减去0.5"""

    def __call__(self, completions, solution, task, messages, images, **kwargs) -> List[float]:
        rewards = []
        for content, gt, task_type, msgs, imgs in zip(completions, solution, task, messages, images):
            stag, etag = self.task_setags.get(task_type, (None, None))
            format_reward, acc_reward = 0., 0.

            # 计算各个类别的format_reward
            if task_type in ['choice']:
                if think_format_with_suffix(content):
                    format_reward = 0.8
                    if think_format_no_suffix(content):
                        format_reward = 1.0
                else:
                    if content.count('<think>') == 1 and content.count('</think>') == 1 \
                        and content.index('<think>') < content.index('</think>'):
                        format_reward += 0.3
                    if content.count('<answer>') == 1 and content.count('</answer>') == 1 \
                        and content.index('<answer>') < content.index('</answer>'):
                        format_reward += 0.4
            elif task_type in ['seg', 'det_no_class', 'det_with_class']:
                format_reward = 0.1
                if content.count(stag) == 1 and content.count(etag) == 1 \
                    and content.index(stag) < content.index(etag):
                    format_reward = 1.0
                elif check_negative_exist(content):
                    format_reward = 1.0
                elif check_other_task_tag_exist(content, self.stag_set, stag):
                    format_reward = 0.0
            else:
                raise ValueError(f'task "{task_type}" not supported')
            
            # 计算各个类别的acc_reward
            try:
                if format_reward > 0.5: # 只有格式符合基本要求才计算acc_reward
                    if task_type in ['choice']:
                        # 提取gt和content中的选项标签
                        gt_choice = extract_choice_label(gt) # 选项序号或False
                        content_answer = extract_between_tags(content, '<answer>', '</answer>')
                        content_choice = extract_choice_label(content_answer) # 选项序号或False

                        # 若标签符合则赋予1.0的离散reward
                        is_correct = False
                        if gt_choice and content_choice and gt_choice == content_choice:
                            acc_reward = 1.0
                            is_correct = True
                        elif not gt_choice and check_negative_exist(gt) \
                            and not content_choice and check_negative_exist(content_answer):
                            acc_reward = 1.0
                            is_correct = True
                        
                        # 若tokenzier存在则计算长度reward
                        if self.tokenizer is not None:
                            if is_correct:
                                # Swap min/max for correct answers
                                max_len_value = self.max_len_value_correct
                                min_len_value = self.min_len_value_correct
                            else:
                                max_len_value = self.max_len_value_wrong
                                min_len_value = self.min_len_value_wrong
                            gen_len = len(self.tokenizer.encode(extract_between_tags(content, '<think>', '</think>')))
                            acc_reward = self.cosfn(gen_len, self.max_len, max_len_value, min_len_value)
                        
                        # 考虑思考语言对reward的影响
                        user_last_query = [m['content'] for m in msgs if m['role'] == 'user'][-1]
                        language_inconsistency = has_chinese(user_last_query) \
                            != has_chinese(content) #has_chinese(extract_between_tags(content, '<think>', '</think>'))
                        
                        # 如果存在语言不一致
                        if language_inconsistency:
                            if is_correct:
                                if self.tokenizer is not None:
                                    acc_reward = min(self.min_len_value_correct, self.max_len_value_correct) - 0.1
                                else:
                                    acc_reward = 0.9
                            else:
                                if self.tokenizer is not None:
                                    acc_reward = min(self.min_len_value_wrong, self.max_len_value_wrong) - 0.1
                                else:
                                    acc_reward = -0.1

                        # 如果回答正确，且器官判断错误，则惩罚0.5
                        if is_correct:
                            gt_organ, img_root = 'unknown', osp.dirname(imgs[0]['path'])
                            for img_keyword, organ in image_organ_map.items():
                                if img_keyword in img_root:
                                    gt_organ = organ
                                    break
                            think_organ = infer_organ_from_text(extract_between_tags(content, '<think>', '</think>'))
                            if think_organ=='unsure' or (gt_organ!=think_organ and 'unknown' not in [gt_organ, think_organ]):
                                acc_reward -= 0.5

                    elif task_type in ['seg', 'det_no_class', 'det_with_class']:
                        reward_func = globals().get(f"{task_type}_reward")
                        acc_reward = reward_func(
                            extract_between_tags(gt, stag, etag, include_tags=True, return_origin=True),
                            extract_between_tags(content, stag, etag, include_tags=True, return_origin=True)
                        )
            except:
                acc_reward = 0.0

            rewards.append(format_reward + acc_reward)
        
        return rewards


orms['pathorm'] = PathORM
orms['pathorm_organ'] = PathORM_Organ