import os
import cv2
import sys
import torch
import numpy as np
from torch import nn
from PIL import Image
from tqdm import tqdm
from typing import Callable
from termcolor import colored
from torch.nn import functional as F
from multiprocessing.pool import ThreadPool

from typing import Mapping, TypeVar, Union, Iterable, Callable
# these are generic type vars to tell mapping to accept any type vars when creating a type
KT = TypeVar("KT")  # key type
VT = TypeVar("VT")  # value type


class dotdict(dict, Mapping[KT, VT]):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = make_dotdict() or d = make_dotdict{'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """

    def update(self, dct=None, **kwargs):
        if dct is None:
            dct = kwargs
        elif isinstance(dct, Mapping):
            dct.update(kwargs)
        else:
            raise TypeError("dct must be a mapping")
        for k, v in dct.items():
            if k in self:
                target_type = type(self[k])
                if not isinstance(v, target_type):
                    # NOTE: bool('False') will be True
                    if target_type == bool and isinstance(v, str):
                        dct[k] = v == 'True'
                    else:
                        dct[k] = target_type(v)
        dict.update(self, dct)

    # def __hash__(self):
    #     # return hash(''.join([str(self.values().__hash__())]))
    #     return super(dotdict, self).__hash__()

    # def __init__(self, *args, **kwargs):
    #     super(dotdict, self).__init__(*args, **kwargs)

    """
    Uncomment following lines and 
    comment out __getattr__ = dict.__getitem__ to get feature:
    
    returns empty numpy array for undefined keys, so that you can easily copy things around
    TODO: potential caveat, harder to trace where this is set to np.array([], dtype=np.float32)
    """

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError as e:
            raise AttributeError(e)
    # MARK: Might encounter exception in newer version of pytorch
    # Traceback (most recent call last):
    #   File "/home/xuzhen/miniconda3/envs/torch/lib/python3.9/multiprocessing/queues.py", line 245, in _feed
    #     obj = _ForkingPickler.dumps(obj)
    #   File "/home/xuzhen/miniconda3/envs/torch/lib/python3.9/multiprocessing/reduction.py", line 51, in dumps
    #     cls(buf, protocol).dump(obj)
    # KeyError: '__getstate__'
    # MARK: Because you allow your __getattr__() implementation to raise the wrong kind of exception.
    # FIXME: not working typing hinting code
    __getattr__: Callable[..., 'torch.Tensor'] = __getitem__  # type: ignore # overidden dict.__getitem__
    __getattribute__: Callable[..., 'torch.Tensor']  # type: ignore
    # __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class default_dotdict(dotdict):
    def __init__(self, type=object, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        dict.__setattr__(self, 'type', type)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except (AttributeError, KeyError) as e:
            super().__setitem__(key, dict.__getattribute__(self, 'type')())
            return super().__getitem__(key)


def parallel_execution(*args, action: Callable, num_processes=16, print_progress=False, sequential=False, **kwargs):
    # NOTE: we expect first arg / or kwargs to be distributed
    # NOTE: print_progress arg is reserved

    def get_valid_arg(args, kwargs): return args[0] if isinstance(args[0], list) else next(iter(kwargs.values()))  # TODO: search through them all

    def get_action_args(valid_arg, args, kwargs, i):
        action_args = [(arg[i] if isinstance(arg, list) and len(arg) == len(valid_arg) else arg) for arg in args]
        action_kwargs = {key: (kwargs[key][i] if isinstance(kwargs[key], list) and len(kwargs[key]) == len(valid_arg) else kwargs[key]) for key in kwargs}
        return action_args, action_kwargs

    def maybe_tqdm(x): return tqdm(x) if print_progress else x

    if not sequential:
        # Create ThreadPool
        pool = ThreadPool(processes=num_processes)

        # Spawn threads
        results = []
        asyncs = []
        valid_arg = get_valid_arg(args, kwargs)
        for i in range(len(valid_arg)):
            action_args, action_kwargs = get_action_args(valid_arg, args, kwargs, i)
            async_result = pool.apply_async(action, action_args, action_kwargs)
            asyncs.append(async_result)

        # Join threads and get return values
        for async_result in maybe_tqdm(asyncs):
            results.append(async_result.get())  # will sync the corresponding thread
        pool.close()
        pool.join()
        return results
    else:
        results = []
        valid_arg = get_valid_arg(args, kwargs)
        for i in maybe_tqdm(range(len(valid_arg))):
            action_args, action_kwargs = get_action_args(valid_arg, args, kwargs, i)
            async_result = action(*action_args, **action_kwargs)
            results.append(async_result)
        return results


def run(cmd, quite=False):
    if isinstance(cmd, list):
        cmd = ' '.join(list(map(str, cmd)))
    func = sys._getframe(1).f_code.co_name
    if not quite:
        cmd_color = 'blue' if not cmd.startswith('rm') else 'red'
        log(colored(func, 'yellow') + ": " + colored(cmd, cmd_color))
    code = os.system(cmd)
    if code != 0:
        log(colored(str(code), 'red') + " <- " + colored(func, 'yellow') + ": " + colored(cmd, 'red'))
        raise RuntimeError(f'{code} <- {func}: {cmd}')


def log(msg, color=None, attrs=None):
    func = sys._getframe(1).f_code.co_name
    frame = sys._getframe(1)
    module = frame.f_globals['__name__'] if frame is not None else ''
    tqdm.write(colored(module, 'blue') + " -> " + colored(func, 'green') + ": " + colored(str(msg), color, attrs))  # be compatible with existing tqdm loops


def load_image(img_path: str, ratio=1.0):
    if img_path.endswith('.jpg'):
        im = Image.open(img_path)
        im.draft('RGB', (int(im.width * ratio), int(im.height * ratio)))
        return np.asarray(im).astype(np.float32) / 255
    else:
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image.ndim >= 3 and image.shape[-1] >= 3:
            image[..., :3] = image[..., [2, 1, 0]]
        elif image.ndim == 2:
            image = image[..., None]
        image = image.astype(np.float32) / 255  # BGR to RGB
        height, width = image.shape[:2]
        if ratio != 1.0:
            image = cv2.resize(image, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_AREA)
        return image


def load_unchanged(img_path: str, ratio=1.0):
    if img_path.endswith('.jpg'):
        im = Image.open(img_path)
        im.draft('RGB', (int(im.width * ratio), int(im.height * ratio)))
        return np.asarray(im)
    else:
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image.shape[-1] >= 3:
            image[..., :3] = image[..., [2, 1, 0]]
        height, width = image.shape[:2]
        if ratio != 1.0:
            image = cv2.resize(image, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_NEAREST)
        return image


def load_mask(img_path: str, ratio=1.0):
    if img_path.endswith('.jpg'):
        im = Image.open(img_path)
        im.draft('L', (int(im.width * ratio), int(im.height * ratio)))
        return (np.asarray(im)[..., None] > 128).astype(np.uint8)
    else:
        mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)[..., None] > 128  # BGR to RGB
        height, width = mask.shape[:2]
        if ratio != 1.0:
            mask = cv2.resize(mask.astype(np.uint8), (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_NEAREST)[..., None]
            # WTF: https://stackoverflow.com/questions/68502581/image-channel-missing-after-resizing-image-with-opencv
        return mask


def save_unchanged(img_path: str, img: np.ndarray, quality=100):
    if img_path.endswith('.hdr'):
        return cv2.imwrite(img_path, img)  # nothing to say about hdr
    if img.shape[-1] >= 3:
        img[..., :3] = img[..., [2, 1, 0]]
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    return cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])


def save_image(img_path: str, img: np.ndarray, quality=100):
    if img_path.endswith('.hdr'):
        return cv2.imwrite(img_path, img)  # nothing to say about hdr
    if img.shape[-1] >= 3:
        img[..., :3] = img[..., [2, 1, 0]]
    img = (img * 255).clip(0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    return cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])


def save_mask(msk_path: str, msk: np.ndarray, quality=100):
    os.makedirs(os.path.dirname(msk_path), exist_ok=True)
    if msk.ndim == 2:
        msk = msk[..., None]
    return cv2.imwrite(msk_path, msk[..., 0] * 255, [cv2.IMWRITE_JPEG_QUALITY, quality])


def reflect(ray_d: torch.Tensor, norm: torch.Tensor):
    dot = (ray_d * norm).sum(dim=-1, keepdim=True)
    return 2 * (norm * dot) - ray_d


def read_hdr(path):
    # TODO: will this support openexr? could not find valid openexr python binding
    # TODO: implement saving in hdr format
    """Reads an HDR map from disk.

    Args:
        path (str): Path to the .hdr file.

    Returns:
        numpy.ndarray: Loaded (float) HDR map with RGB channels in order.
    """
    with open(path, 'rb') as h:
        buffer_ = np.fromstring(h.read(), np.uint8)
    bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32)


def take_gradient(output: torch.Tensor,
                  input: torch.Tensor,
                  d_out: torch.Tensor = None,
                  create_graph: bool = True,
                  retain_graph: bool = True,
                  is_grads_batched: bool = False,
                  ):
    if d_out is not None:
        d_output = d_out
    elif isinstance(output, torch.Tensor):
        d_output = torch.ones_like(output, requires_grad=False)
    else:
        d_output = [torch.ones_like(o, requires_grad=False) for o in output]
    grads = torch.autograd.grad(inputs=input,
                                outputs=output,
                                grad_outputs=d_output,
                                create_graph=create_graph,
                                retain_graph=retain_graph,
                                only_inputs=True,
                                is_grads_batched=is_grads_batched,
                                )
    if len(grads) == 1:
        return grads[0]  # return the gradient directly
    else:
        return grads  # to be expanded


def get_max_mem():
    return torch.cuda.max_memory_allocated() / 2 ** 20


def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # channel last: normalization
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def normalize_sum(x: torch.Tensor, eps: float = 1e-8):
    return x / (x.sum(dim=-1, keepdim=True) + eps)


class GradModule(nn.Module):
    # GradModule is a module that takes gradient based on whether we're in training mode or not
    # Avoiding the high memory cost of retaining graph of *not needed* back porpagation
    def __init__(self):
        super(GradModule, self).__init__()

    def take_gradient(self, output: torch.Tensor, input: torch.Tensor, d_out: torch.Tensor = None, create_graph: bool = False, retain_graph: bool = False) -> torch.Tensor:
        return take_gradient(output, input, d_out, self.training or create_graph, self.training or retain_graph)

    def jacobian(self, output: torch.Tensor, input: torch.Tensor):
        with torch.enable_grad():
            outputs = output.split(1, dim=-1)
        grads = [self.take_gradient(o, input, retain_graph=(i < len(outputs))) for i, o in enumerate(outputs)]
        jac = torch.stack(grads, dim=-1)
        return jac


class MLP(GradModule):
    def __init__(self, input_ch=32, W=256, D=8, out_ch=257, skips=[4], actvn=nn.ReLU(), out_actvn=nn.Identity()):
        super(MLP, self).__init__()
        self.skips = skips
        self.linears = []
        for i in range(D + 1):
            I, O = W, W
            if i == 0:
                I = input_ch
            if i in skips:
                I = input_ch + W
            if i == D:
                O = out_ch
            self.linears.append(nn.Linear(I, O))
        self.linears = nn.ModuleList(self.linears)
        self.actvn = actvn
        self.out_actvn = out_actvn
        for l in self.linears:
            nn.init.kaiming_uniform_(l.weight, nonlinearity='relu')

    def forward(self, input: torch.Tensor):
        x = input
        for i, l in enumerate(self.linears):
            if i in self.skips:
                x = torch.cat([x, input], dim=-1)
            if i == len(self.linears) - 1:
                a = self.out_actvn
            else:
                a = self.actvn
            x = a(l(x))  # actual forward
        return x


def number_of_params(network: nn.Module):
    return sum([p.numel() for p in network.parameters() if p.requires_grad])


def make_params(params: torch.Tensor):
    return nn.Parameter(params, requires_grad=True)


def make_buffer(params: torch.Tensor):
    return nn.Parameter(params, requires_grad=False)


def mse(x: torch.Tensor, y: torch.Tensor):
    return ((x.float() - y.float())**2).mean()


def list_to_numpy(x: list): return np.stack(x).transpose(0, 3, 1, 2)


def list_to_tensor(x: list, device='cuda'): return torch.from_numpy(list_to_numpy(x)).to(device, non_blocking=True)  # convert list of numpy arrays of HWC to BCHW
def numpy_to_list(x: np.ndarray): return [y for y in x.transpose(0, 2, 3, 1)]
def tensor_to_list(x: torch.Tensor): return numpy_to_list(x.detach().cpu().numpy())  # convert tensor of BCHW to list of numpy arrays of HWC
