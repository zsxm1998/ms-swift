import os.path as osp
import json
import argparse
import inspect
from tqdm import tqdm
from typing import List, Dict
from dataclasses import asdict
from copy import deepcopy

from swift.llm import InferClient, VllmEngine, InferRequest, RequestConfig, AdapterRequest
from swift.llm.infer.protocol import ChatCompletionResponse
from swift.plugin.multi_turn import magnify_wsi


def load_json(path):
    with open(path, 'r') as jf:
        data = json.load(jf)
    return data


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def preprocess_data(args):
    question_file, image_folder = args.question_file, args.image_folder
    if question_file.endswith(".json"):
        questions = load_json(question_file)
        for i, data in enumerate(questions):
            if 'image' in data:
                data['images'] = data.pop('image')
            if 'images' in data and isinstance(data['images'], str):
                data['images'] = [data['images']]
            if data['messages'][-1]['role'] in ['assistant', 'gpt', 'bot']:
                data['answer'] = data['messages'].pop()['content']
            elif 'solution' in data:
                data['answer'] = data.pop('solution')
            data['question_id'] = data.get('question_id', i)
    else:
        ori_questions = load_jsonl(question_file)
        questions = []
        for i, data in enumerate(ori_questions):
            qdata = {}
            if 'image' in data:
                qdata['images'] = data['image']
            if 'images' in data:
                qdata['images'] = data['images']
            if 'images' in qdata and isinstance(qdata['images'], str):
                qdata['images'] = [qdata['images']]
            qdata['messages'] = [{'role': 'user', 'content': data['text']}]
            if 'answer' in data:
                qdata['answer'] = data['answer']
            qdata['question_id'] = data.get('question_id', i)
            questions.append(qdata)
    for data in questions:
        if 'images' in data:
            for i, image in enumerate(data['images']):
                if not image.startswith("http") and not osp.exists(image):
                    if image_folder and not osp.isabs(image):
                        data['images'][i] = osp.join(image_folder, image)
                    else:
                        raise ValueError(f'Image "{image}" not found.')
            user_text = '\n'.join([x['content'] for x in data['messages'] if x['role'] in ['user', 'human']])
            user_image_count = user_text.count('<image>')
            if user_image_count < len(data['images']):
                data['messages'][-1]['content'] = ''.join(['<image>\n']*(len(data['images'])-user_image_count)) + data['messages'][-1]['content']
        if data['messages'][0]['role'] not in ['system', 'system_prompt']:
            system_str = args.system or ''
            if args.think:
                system_str = system_str + '\n\n' + args.think_system if system_str else args.think_system
            if args.func:
                system_str = system_str + '\n\n' + args.func_system if system_str else args.func_system
            if system_str:
                data['messages'].insert(0, {'role': 'system', 'content': system_str})
    return questions

def main(args):
    # 处理数据，将不同格式的输入进行统一，并处理图片路径问题，返回messages格式的数据
    dataset = preprocess_data(args)

    # 根据条件创建推理后端或确定vllm后端
    if args.model_path: # 使用模型创建vllm推理后端
        engine = VllmEngine(
            model_id_or_path=args.model_path,
            gpu_memory_utilization=0.8,
            tensor_parallel_size=args.tensor_parallel_size,
            enable_lora=args.lora_path is not None,
            max_lora_rank=16,
            use_async_engine=False,
            max_num_seqs=args.batch_size if args.batch_size > 1 else inspect.signature(VllmEngine.__init__).parameters['max_num_seqs'].default,
            limit_mm_per_prompt={"image": 7, "video": 1},
        )
    else: # 使用现成的vllm后端
        engine = InferClient(host=args.vllm_host, port=args.vllm_port, timeout=3600)
        vllm_models = engine.models
        if args.vllm_model and args.vllm_model not in vllm_models:
            raise ValueError(f'Model "{args.vllm_model}" not found in vllm models: {vllm_models}')
        elif not args.vllm_model:
            args.vllm_model = vllm_models[0]
        if args.batch_size > 1:
            engine.llm_max_batch_size = args.batch_size
            engine.mllm_max_batch_size = args.batch_size

    # 准备推理
    infer_kwargs = {'request_config': RequestConfig(max_tokens=args.max_tokens, temperature=args.temperature)}
    infer_kwargs['use_tqdm'] = False #True if args.batch_size > 1 else False
    if args.lora_path:
        infer_kwargs['adapter_request'] = AdapterRequest('lora1', args.lora_path)
    if args.vllm_model:
        infer_kwargs['model'] = args.vllm_model

    # 推理
    ans_file = open(args.answers_file, "w")
    if args.batch_size > 1:
        for i in tqdm(range(0, len(dataset), args.batch_size)):
            inputs_slice = [InferRequest(messages=data['messages'], images=data['images']) for data in dataset[i:i + args.batch_size]]
            resp_list: List[ChatCompletionResponse] = engine.infer(inputs_slice, **infer_kwargs)
            if args.func: # 处理函数调用导致的多轮对话
                messages_list = [None] * args.batch_size
                remove_response = True # 这个flag控制从第二轮开始如果messages最后是assistant则concat回复内容
                while len(inputs_slice) > 0:
                    inputs = []
                    for j, output in enumerate(resp_list):
                        choice = output.choices[0]
                        _input: Dict = asdict(inputs_slice[j]) if isinstance(inputs_slice[j], InferRequest) else deepcopy(inputs_slice[j])
                        if remove_response or _input['messages'][-1]['role'] != 'assistant' or not \
                                _input['messages'][-1]['content']:
                            InferRequest.remove_response(_input['messages'])
                            _input['messages'].append({'role': 'assistant', 'content': choice.message.content})
                        else:
                            _input['messages'][-1]['content'] += choice.message.content
                        if 'index' not in _input:
                            _input['index'] = j
                        _input['finish_reason'] = choice.finish_reason
                        inputs.append(_input)
                    results: List[Dict] = magnify_wsi(inputs)
                    inputs_slice = [r for r in results if not r['finished']]
                    for r in results:
                        if r['finished'] or r['finish_reason'] == 'length':
                            messages_list[r['index']] = (r['messages'], r['finish_reason'])
                    if len(inputs_slice) > 0:
                        _input_std = [InferRequest.from_dict(_input) for _input in inputs_slice]
                        resp_list = engine.infer(infer_requests=_input_std, **infer_kwargs)
                    remove_response = False # concat responses from the second loop
                
                for (mess, finish_reason), data in zip(messages_list, dataset[i:i + args.batch_size]):
                    if mess[0]['role'] == 'system':
                        mess.pop(0)  # Remove system message if exists
                    response = mess[-1]['content']
                    ans_file.write(json.dumps({
                        "question_id": data['question_id'],
                        "images": data['images'],
                        "prompt": data['messages'][-1]['content'],
                        "model_response": response,
                        "gt_answer": data.get('answer', None),
                        "messages": mess,
                        "finish_reason": finish_reason,
                    }, ensure_ascii=False) + "\n")
                    ans_file.flush()
            else:
                for resp, data in zip(resp_list, dataset[i:i + args.batch_size]):
                    response = resp.choices[0].message.content
                    ans_file.write(json.dumps({
                        "question_id": data['question_id'],
                        "images": data['images'],
                        "prompt": data['messages'][-1]['content'],
                        "model_response": response,
                        "gt_answer": data.get('answer', None),
                    }, ensure_ascii=False) + "\n")
                    ans_file.flush()
    else:
        if args.func:
            raise NotImplementedError("Function calling mode is not supported for batch size 1.")
        for data in tqdm(dataset):
            infer_requests = [InferRequest(messages=data['messages'], images=data['images'])]
            response = engine.infer(infer_requests, **infer_kwargs)[0].choices[0].message.content
            ans_file.write(json.dumps({
                "question_id": data['question_id'],
                "images": data['images'],
                "prompt": data['messages'][-1]['content'],
                "model_response": response,
                "gt_answer": data.get('answer', None),
            }, ensure_ascii=False) + "\n")
            ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据相关参数
    parser.add_argument("--question-file", type=str, required=True, help="问题文件路径")
    parser.add_argument("--answers-file", type=str, required=True, help="答案文件路径")
    parser.add_argument("--image-folder", type=str, default=None, help="图片文件夹路径，当图片不是绝对路径时使用")

    # 如果使用模型推理，则需要在脚本内部署后端
    parser.add_argument("--model-path", type=str, default=None, help="模型路径")
    parser.add_argument("--lora-path", type=str, default=None, help="LoRA 权重路径，可选")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1, help="张量并行大小")

    # 使用现成的vllm客户端推理，不需要传模型参数，只需要传vllm接口
    parser.add_argument("--vllm-host", type=str, default='127.0.0.1', help="vllm接口地址")
    parser.add_argument("--vllm-port", type=int, default=8000, help="vllm接口端口")
    parser.add_argument("--vllm-model", type=str, default=None, help="vllm模型名称")

    # 推理相关参数
    parser.add_argument("--batch-size", type=int, default=64, help="批处理大小")
    parser.add_argument("--max-tokens", type=int, default=4096, help="最大token数，None则为 max_model_len - num_tokens")
    parser.add_argument("--temperature", type=float, default=1, help="温度参数，取值0~2，0表示贪心搜索，越大越随机")
    parser.add_argument("--system", type=str, default="你是由浙江大学VIPA实验室开发的OmniPT多模态智能助手，用于辅助病理医生进行专业、准确、高效的诊断。你能够根据用户输入的图片和文字指令或问题，给出相应的回答。", help="系统提示词，传入则设置为该值")
    parser.add_argument("--think", action='store_true', help="是否使用思考模式，传入则设置为True")
    parser.add_argument("--think-system", type=str, default="在给出回答之前，你**必须**先在心里进行思考。将思考内容放在<think>和</think>之间，在</think>之后进行正式回答。对于有确定性答案的问题，如选择题和填空题等，在回答的最后，将选项或答案放在<answer>和</answer>之间。例如：<think>思考内容</think>正式回答内容和分析<answer>选项或回答短语</answer>", help="思考模式下的额外系统提示词，传入则设置为该值")
    parser.add_argument("--func", action='store_true', help="是否使用函数调用模式，传入则设置为True")
    parser.add_argument("--func-system", type=str, default="下面是你可以调用的工具函数列表：\n```\n[{\n    \"name\": \"get_highres_by_point\",\n    \"description\": \"基于WSI缩略图中的选定坐标获取以该点为中心的固定大小高倍率图像，以完成缩略图无法完成的任务。\",\n    \"parameters\": {\n        \"wsi_index\": {\n            \"type\": \"integer\",\n            \"description\": \"用户提供的WSI缩略图索引，从0开始。\",\n            \"required\": true\n        },\n        \"point_list\": {\n            \"type\": \"array\",\n            \"description\": \"需要放大的点列表。每个点表示为[x, y]，其中x和y为归一化坐标，所有坐标均为0-1000之间的整数。\",\n            \"items\": {\n                \"type\": \"array\",\n                \"description\": \"一个放大点，格式为[x, y]，所有值均为0-1000之间的整数。\",\n                \"minItems\": 2,\n                \"maxItems\": 2,\n                \"items\": {\n                    \"type\": \"integer\",\n                    \"minimum\": 0,\n                    \"maximum\": 1000,\n                    \"examples\": [600, 700]\n                }\n            },\n            \"required\": true\n        }\n    },\n    \"returns\": {\n        \"type\": \"images\",\n        \"description\": \"返回对应坐标点中心的固定大小高倍率图像列表。\"\n    }\n}]\n```\n\n说明：\n- 调用上述工具函数来应对需要进一步处理或包含额外信息检索的用户请求。\n- 如果选择调用函数，仅能按照以下格式回复：\n```\n<function name=\"{function_name}\">{parameters}</function>\n```\n其中：parameters => 一个JSON字典，键为函数参数名，值为相应的参数值。\n\n以下是调用示例：\n```\n<function name=\"example_function_name\">{\"argument_name\": argument_value}</function>\n```\n\n提醒：\n- 函数调用必须遵循指定格式。\n- 必须提供所有必需参数。\n- 每次只能调用一个函数。\n- 整个函数调用必须放在一行内。\n- 函数调用必须放在回复的最后，后面不能有任何其他内容，即整条回复以`</function>`结尾。\n- 如果使用搜索结果回答用户请求，必须添加来源信息。", help="函数调用模式下的额外系统提示词，传入则设置为该值")

    args = parser.parse_args()
    main(args)
