import os
from typing import Union

import torch.cuda
from pigmento import pnt


class GPU:
    @classmethod
    def distributed_device(cls, torch_format=False):
        local_rank = os.environ.get('LOCAL_RANK')
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        if local_rank is None or world_size <= 1:
            return None
        if not torch.cuda.is_available():
            pnt('distributed launch detected without CUDA; using CPU backend')
            return 'cpu' if torch_format else -1
        local_rank = int(local_rank)
        pnt(f'distributed launch detected world_size={world_size} local_rank={local_rank}')
        if torch_format:
            return f'cuda:{local_rank}'
        return local_rank

    @classmethod
    def parse_gpu_info(cls, line, args):
        def to_number(v):
            return float(v.upper().strip().replace('MIB', '').replace('W', ''))

        def processor(k, v):
            return (int(to_number(v)) if 'Not Support' not in v else 1) if k in params else v.strip()

        params = ['memory.free', 'memory.total', 'power.draw', 'power.limit']
        return {k: processor(k, v) for k, v in zip(args, line.strip().split(','))}

    @classmethod
    def get_gpus(cls):
        args = ['index', 'gpu_name', 'memory.free', 'memory.total', 'power.draw', 'power.limit']
        cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(args))
        results = os.popen(cmd).readlines()
        return [cls.parse_gpu_info(line, args) for line in results]

    @classmethod
    def auto_choose(cls, torch_format=False):
        distributed_device = cls.distributed_device(torch_format=torch_format)
        if distributed_device is not None:
            return distributed_device
        if not torch.cuda.is_available():
            pnt('system does not support CUDA')
            if torch_format:
                pnt('auto switching to CPU device')
                return "cpu"
            return -1

        gpus = cls.get_gpus()
        chosen_gpu = sorted(gpus, key=lambda d: d['memory.free'], reverse=True)[0]
        pnt(f'choosing {chosen_gpu["index"]}-th GPU with {chosen_gpu["memory.free"]} / {chosen_gpu["memory.total"]} MB')
        if torch_format:
            return "cuda:" + str(chosen_gpu['index'])
        return int(chosen_gpu['index'])


if __name__ == '__main__':
    GPU.auto_choose()
