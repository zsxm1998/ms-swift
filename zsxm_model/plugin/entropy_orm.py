import copy
import torch
from typing import List

from swift.llm import to_device
from swift.plugin import ORM, orms
from zsxm_model.plugin.path_orm import think_format_no_suffix


class NoImageDeltaRatio(ORM):
    def __init__(self,
                 min_reward=0.0,
                 valid_tasks=['choice', 'choice_func'],
        ):
        self.min_reward = min_reward
        self.valid_tasks = valid_tasks

    @torch.inference_mode
    def __call__(self, completions, task, messages, images, model, template, **kwargs) -> List[float]:
        rewards = []
        for content, task_type, msgs, imgs in zip(completions, task, messages, images):
            if task_type in self.valid_tasks:
                if task_type in ['choice', 'choice_func'] and not think_format_no_suffix(content):
                    rewards.append(0.0)
                    continue
                
                msgs = copy.deepcopy(msgs)

                input_data = template.encode({'messages': msgs, 'images': imgs})
                input_data['input_ids'] = torch.tensor(input_data['input_ids']).unsqueeze(0)
                input_data['labels'] = torch.tensor(input_data['labels']).unsqueeze(0)
                withimage_output = model(**to_device(input_data, model.device))
                withimage_ce = withimage_output.loss.item()

                for m in msgs:
                    if m['role'] in ['user', 'tool'] and '<image>' in m['content']:
                        m['content'] = m['content'].replace('<image>', '').strip()
                
                input_data = template.encode({'messages': msgs})
                input_data['input_ids'] = torch.tensor(input_data['input_ids']).unsqueeze(0)
                input_data['labels'] = torch.tensor(input_data['labels']).unsqueeze(0)
                noimage_output = model(**to_device(input_data, model.device))
                noimage_ce = noimage_output.loss.item()
                
                reward = 1 - (withimage_ce + 1e-3) / (noimage_ce + 1e-3)
                rewards.append(max(reward, self.min_reward))
            else:
                rewards.append(0.0)
        
        return rewards
    

orms['noimage_delta_ratio'] = NoImageDeltaRatio