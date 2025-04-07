import random
import re
import collections
import paddle
import numpy
import math
import json
import paddle
import inspect
import torch
import copy

USE_CACHED_NUMPY = False
cached_numpy = {}

not_zero_apis = [
    "paddle.Tensor.__div__",
    "paddle.Tensor.__floordiv__",
    "paddle.Tensor.__rdiv__",
    "paddle.Tensor.__rfloordiv__",
    "paddle.Tensor.__rtruediv__",
    "paddle.Tensor.__truediv__",
    "paddle.Tensor.divide",
    "paddle.Tensor.floor_divide",
    "paddle.divide",
    "paddle.floor_divide",
    "paddle.nn.functional.kl_div",
    "paddle.sparse.divide",
    "paddle.Tensor.__mod__",
    "paddle.Tensor.__rmod__",
    "paddle.Tensor.floor_mod",
    "paddle.Tensor.mod",
    "paddle.floor_mod",
    "paddle.mod",
]

class TensorConfig:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.numpy_tensor = None
        self.paddle_tensor = None
        self.torch_tensor = None
        self.index = -2
        self.key = "null"
        self.fix = False
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.shape = copy.deepcopy(self.shape)
        result.dtype = copy.deepcopy(self.dtype)
        return result

    def __str__(self):
        return "Tensor("+str(self.shape)+",\""+self.dtype+"\")"
    def __repr__(self):
        return "Tensor("+str(self.shape)+",\""+self.dtype+"\")"

    def convert_dtype_to_torch_type(self, dtype):
        if dtype in ["float32", numpy.float32]:
            return torch.float32
        elif dtype in ['float16', numpy.float16]:
            return torch.float16
        elif dtype in ['float64', numpy.float64]:
            return torch.float64
        elif dtype in ['int16', numpy.int16]:
            return torch.int16
        elif dtype in ['int8', numpy.int8]:
            return torch.int8
        elif dtype in ['bool', numpy.bool_]:
            return torch.bool
        elif dtype in ['bfloat16', numpy.uint16]:
            return torch.bfloat16
        elif dtype in ['uint8', numpy.uint8]:
            return torch.uint8
        elif dtype in ['int32', numpy.int32]:
            return torch.int32
        elif dtype in ['int64', numpy.int64]:
            return torch.int64
        elif dtype in ['complex64', numpy.complex64]:
            return torch.complex64
        elif dtype in ['complex128', numpy.complex128]:
            return torch.complex128
        else:
            raise ValueError(f'Unsupport dtype: {dtype}')

    def numel(self):
        numel = 1
        for i in self.shape:
            numel = numel * i
        return numel
    

    def get_cached_numpy(self, dtype, shape):
        numel = 1
        for i in shape:
            numel = numel * i

        if dtype in cached_numpy:
            tensor = cached_numpy[dtype][:numel].reshape(shape)
        else:
            if "int" in dtype:
                cached_numpy[dtype] = numpy.random.randint(-65535, 65535, size=4300000000, dtype="int64").astype(dtype)
                tensor = cached_numpy[dtype][:numel].reshape(shape)
            else:
                cached_numpy[dtype] = (numpy.random.random([4300000000]) - 0.5).astype(dtype)
                tensor = cached_numpy[dtype][:numel].reshape(shape)
        return tensor
    
    def generate_random_axes(self, api_config):
        if "x" in api_config.kwargs:
            max_dim = len(api_config.kwargs["x"].shape)
        else:
            max_dim = len(api_config.args[0].shape)
        
        if max_dim == 0:
            max_dim = 1 # scalar
            
        shape_len = len(self.shape)
        if shape_len == 0:
            dim = random.randint(0, max_dim - 1)
            if random.choice([True, False]):
                dim -= max_dim
            return numpy.array(dim, dtype=self.dtype)
        elif shape_len == 1:
            all_dims = list(range(max_dim))
            random_dims = random.sample(all_dims, self.shape[0])
            final_dims = []
            for dim in random_dims:
                if random.choice([True, False]):
                    dim -= max_dim
                final_dims.append(dim)
            return numpy.array(final_dims, dtype=self.dtype)
        else:
            raise ValueError(
                f"Invalid shape for 'axis' Tensor in {api_config.api_name}. "
                f"Expected a 0-D or 1-D Tensor, but got shape {self.shape}."
            )
    

    def get_numpy_tensor(self, api_config):
        index = self.index
        if self.dtype in ["float8_e5m2", "float8_e4m3fn"]:
            print("Warning ", self.dtype, "not supported")
            return
        if self.numpy_tensor is None:
            if api_config.api_name in not_zero_apis:
                if "int" in self.dtype:
                    self.numpy_tensor = (numpy.random.randint(1, 65535, size=self.shape)).astype(self.dtype)
                else:
                    dtype = "float32" if self.dtype == "bfloat16" else self.dtype
                    self.numpy_tensor = (numpy.random.random(self.shape) + 0.5).astype(dtype)
            # a
            elif api_config.api_name in ["paddle.arange"]:
                tensor_num = 0
                for arg in api_config.args:
                    if "Tensor" in str(arg):
                        tensor_num += 1
                if tensor_num == 3 or "step" in api_config.kwargs:
                    if self.check_arg(api_config,2,"step"):
                        if "int" in self.dtype:
                            x = numpy.random.random()
                            if x > 0.5:
                                self.numpy_tensor = (numpy.random.randint(1, 65535, size=self.shape)).astype(self.dtype)
                            else:
                                self.numpy_tensor = (numpy.random.randint(-65536, -1, size=self.shape)).astype(self.dtype)
                        else:
                            dtype = "float32" if self.dtype == "bfloat16" else self.dtype
                            x = numpy.random.random()
                            if x > 0.5:
                                self.numpy_tensor = (numpy.random.random(self.shape) + 1.0).astype(dtype)
                            else:
                                self.numpy_tensor = (numpy.random.random(self.shape) - 2.0).astype(dtype)                               
            
            elif api_config.api_name in ["paddle.argmax","paddle.argmin"]:  
                if  self.check_arg(api_config, 1, "axis"):
                    arr=self.get_arg(api_config,0,'x')                
                    min_dim = min(arr.shape)
                    indices = (numpy.random.randint(0, min_dim-1, size=self.numel())).astype("int64")
                    self.numpy_tensor = indices.reshape(self.shape)
                    self.dtype = "int64"
                

            elif api_config.api_name in ["paddle.atan2"]:
                s1=self.get_arg(api_config,0)
                s2=self.get_arg(api_config,1)
                s1=s1.shape
                s2=s2.shape
                if max(s1) == 0 and max(s2) == 0:
                    while len(s1)>len(s2):
                        s2.append(0)
                    while len(s2)>len(s1):
                        s1.append(0)
                self.numpy_tensor=numpy.random.random(s1)
            # b
            # c
            elif api_config.api_name in ["paddle.chunk"]:
                if self.check_arg(api_config, 2, "axis"):
                    x_tensor = None
                    if "x" in api_config.kwargs:
                        x_tensor = api_config.kwargs["x"]
                    else:
                        x_tensor = api_config.args[0]
                    chunks = None
                    if "chunks" in api_config.kwargs:
                        chunks = api_config.kwargs["chunks"]
                    else:
                        chunks = api_config.args[1]
                    valid_axes = []
                    for i, dim_size in enumerate(x_tensor.shape):
                        if dim_size % chunks == 0:
                            valid_axes.append(i)
                    if not valid_axes:
                        valid_axes = [0]
                    chosen_axis = random.choice(valid_axes)
                    if len(self.shape) == 0:  
                        self.numpy_tensor = numpy.array(chosen_axis, dtype=self.dtype)
                    elif len(self.shape) == 1:  
                        if self.shape[0] == 1:
                            self.numpy_tensor = numpy.array([chosen_axis], dtype=self.dtype)
                        else:
                            raise ValueError(
                                f"Invalid shape for 'axis' Tensor in paddle.chunk. "
                                f"Expected a 0-D or 1-D Tensor with 1 element, but got shape {self.shape}."
                            )
                    else:
                        raise ValueError(
                            f"Invalid shape for 'axis' Tensor in paddle.chunk. "
                            f"Expected a 0-D or 1-D Tensor, but got shape {self.shape}."
                        )
            elif api_config.api_name in ["paddle.cumsum"] and self.check_arg(api_config, 1, "axis"):
                # special args[1] tensor init, for the rest reuse default initialization logic
                x_tensor_config = self.get_arg(api_config, 0, "x")
                len_shape = len(x_tensor_config.shape)
                self.numpy_tensor = numpy.random.randint(-len_shape, len_shape, size=self.shape)
                return self.numpy_tensor
            # d
            # e
            elif api_config.api_name in ["paddle.eye"]:
                self.numpy_tensor = numpy.random.randint(0, 2048, size = self.shape)

            elif api_config.api_name in ["paddle.expand"]:
                if index>0:
                    if 'x' in api_config.kwargs:
                        d=self.get_arg(api_config,arg_name='x')
                    else:   
                        d=self.get_arg(api_config,0)
                    s=d.shape
                    if len(s)==0 or s[index-1]==1:
                        self.numpy_tensor = (numpy.random.randint(1, 127, size=self.shape)).astype(self.dtype)
                    else:
                        self.numpy_tensor = numpy.array(s[index-1])

            elif api_config.api_name in ["paddle.expand_as"]:
                if self.dtype=='float16':
                    self.dtype='float32'
                    

            # f
            elif api_config.api_name in ["paddle.full"]:
                if self.check_arg(api_config, 1, "fill_value"):
                    if "int" in self.dtype:
                        self.numpy_tensor = (numpy.random.randint(1, 65535, size=self.shape)).astype(self.dtype)
                    else:
                        dtype = "float32" if self.dtype == "bfloat16" else self.dtype
                        self.numpy_tensor = (numpy.random.random(self.shape) + 0.5).astype(dtype)
                else:
                    self.numpy_tensor = (numpy.random.randint(0, 2048, size=self.shape)).astype(self.dtype)
            # g
            # h
            # i
            # j
            # k
            # l
            elif api_config.api_name in ["paddle.logspace"]:
                if self.check_arg(api_config, 2, "num"):
                    self.numpy_tensor = numpy.random.randint(1, 65535, size=self.shape)
            elif api_config.api_name in ["paddle.linalg.cholesky"]:
                if self.check_arg(api_config, 0, "x"):
                    if len(self.shape) < 2 or self.shape[-1] != self.shape[-2]:
                        raise ValueError("Shape must have at least 2 dimensions and last two dimensions must be equal")
                    batch_dims = self.shape[:-2]
                    matrix_dim = self.shape[-1]
                    A = numpy.random.random(batch_dims + [matrix_dim, matrix_dim]).astype(self.dtype)
                    if len(batch_dims) > 0:
                        tensor = numpy.einsum('...ij,...kj->...ik', A, A)
                    else:
                        tensor = numpy.dot(A, A.T)
                    tensor += numpy.eye(matrix_dim, dtype=self.dtype) * 1e-6
                    print("cholesky tensor", tensor)
                    self.numpy_tensor = tensor
            elif api_config.api_name in ["paddle.linalg.cov"]:
                if self.check_arg(api_config, 0, "x"):
                    if len(self.shape) < 1 or len(self.shape) > 2:
                        raise ValueError("Shape must have 1 or 2 dimensions for covariance input")
                    tensor = numpy.random.random(self.shape).astype(self.dtype)
                    tensor += numpy.random.random(self.shape).astype(self.dtype) * 1e-6
                    self.numpy_tensor = tensor
                elif self.check_arg(api_config, 3, "fweights"):
                    x_shape = self.get_arg(api_config, 0, "x").shape
                    rowvar = self.get_arg(api_config, 1, "rowvar")
                    if rowvar is None:
                        rowvar = True
                    n_observations = (x_shape[1] if rowvar else x_shape[0]) if len(x_shape) > 1 else x_shape[0]
                    self.numpy_tensor = numpy.random.randint(1, 11, size=(n_observations,)).astype(self.dtype)
                elif self.check_arg(api_config, 4, "aweights"):
                    x_shape = self.get_arg(api_config, 0, "x").shape
                    rowvar = self.get_arg(api_config, 1, "rowvar")
                    if rowvar is None:
                        rowvar = True
                    n_observations = (x_shape[1] if rowvar else x_shape[0]) if len(x_shape) > 1 else x_shape[0]
                    if self.dtype in ["float32", "float64"]:
                        self.numpy_tensor = numpy.random.uniform(0.1, 1.0, size=(n_observations,)).astype(self.dtype)
                    else:
                        self.numpy_tensor = numpy.random.randint(1, 11, size=(n_observations,)).astype(self.dtype)
            elif api_config.api_name in ["paddle.linalg.eigh"]:
                if self.check_arg(api_config, 0, "x"):
                    if len(self.shape) < 2 or self.shape[-1] != self.shape[-2]:
                        raise ValueError("Shape must have at least 2 dimensions and last two dimensions must be equal")
                    batch_dims = self.shape[:-2]
                    matrix_dim = self.shape[-1]
                    A = numpy.random.random(batch_dims + [matrix_dim, matrix_dim]).astype(self.dtype)
                    if self.dtype in ['complex64', 'complex128']:
                        A = A + 1j * numpy.random.random(batch_dims + [matrix_dim, matrix_dim]).astype(self.dtype)
                        tensor = A + A.swapaxes(-1, -2).conj()  # A + A^H
                    else:
                        if len(batch_dims) > 0:
                            tensor = numpy.einsum('...ij,...kj->...ik', A, A)
                        else:
                            tensor = numpy.dot(A, A.T)
                    tensor += numpy.eye(matrix_dim, dtype=self.dtype) * 1e-6
                    self.numpy_tensor = tensor
            elif api_config.api_name in ["paddle.linalg.lstsq"]:
                if self.check_arg(api_config, 0, "x") or self.check_arg(api_config, 1, "y"):
                    if len(self.shape) < 2:
                        raise ValueError("Shape must have at least 2 dimensions for lstsq x")
                    batch_dims = self.shape[:-2]
                    M, N = self.shape[-2], self.shape[-1]
                    self.numpy_tensor = numpy.random.random(batch_dims + [M, N]).astype(self.dtype)
            # m
            elif api_config.api_name in ["paddle.mean", "paddle.max", "paddle.min"]:
                if self.check_arg(api_config, 1, "axis"):
                    self.numpy_tensor = self.generate_random_axes(api_config)

            elif api_config.api_name in ["paddle.multiplex"]:
                if 'inputs' in api_config.kwargs:
                    s=self.get_arg(api_config,arg_name='inputs')
                    if index==len(s):
                        self.numpy_tensor = (numpy.random.randint(0,len(s), size=self.shape)).astype(self.dtype)

            # n
            # o
            elif api_config.api_name in ["paddle.ones"]:
                if api_config.api_name == "paddle.ones" and len(self.shape) == 0:
                    self.numpy_tensor = numpy.array(random.randint(1, 2048), dtype=self.dtype)
                else:
                    self.numpy_tensor = numpy.random.randint(1, 65535, size=self.shape).astype(self.dtype)
            # p
            elif api_config.api_name in ["paddle.prod"]:
                if self.check_arg(api_config, 1, "axis"):
                    self.numpy_tensor = self.generate_random_axes(api_config)
            # q
            elif api_config.api_name in ["paddle.quantile"]:
                if index==1:
                    self.numpy_tensor = numpy.random.rand(1).astype(self.dtype)
            # r
            elif api_config.api_name in ["paddle.Tensor.reshape","paddle.reshape"]:
                if self.index == 0 or "x" == self.key:
                    api_config.maxvalue = self.numel() // api_config.intnum
                elif self.index == 1 or "shape" == self.key:
                    flag = True
                    if "shape" == self.key:
                        if "Tensor" in str(type((api_config.kwargs["shape"]))):
                            flag = False
                    else:
                        if "Tensor" in str(type(api_config.args[1])):
                            flag = False
                    self.dtype = "int32"
                    if flag:
                        if self.fix:
                            if self.shape == []:
                                self.numpy_tensor = api_config.maxvalue
                            else:
                                self.numpy_tensor = [api_config.maxvalue]
                        else:
                            self.numpy_tensor = (numpy.random.randint(1, api_config.maxvalue+1, size=self.shape)).astype(self.dtype)
                            if self.shape == []:
                                while api_config.maxvalue % self.numpy_tensor:
                                    self.numpy_tensor = (numpy.random.randint(1, api_config.maxvalue+1, size=self.shape)).astype(self.dtype)
                                api_config.maxvalue = api_config.maxvalue // self.numpy_tensor
                            else:
                                while api_config.maxvalue % self.numpy_tensor[0]:
                                    self.numpy_tensor = (numpy.random.randint(1, api_config.maxvalue+1, size=self.shape)).astype(self.dtype)
                                api_config.maxvalue = api_config.maxvalue // self.numpy_tensor[0]
                    else:
                        self.numpy_tensor= numpy.empty(self.shape)
                        num = self.numel()
                        for i in range(num - 1):
                            temp = numpy.random.randint(1, api_config.maxvalue+1)
                            while api_config.maxvalue % temp:
                               temp = numpy.random.randint(1, api_config.maxvalue+1)
                            api_config.maxvalue = api_config.maxvalue // temp
                            self.numpy_tensor[i] = temp
                        self.numpy_tensor[num-1]=api_config.maxvalue
                        self.numpy_tensor = self.numpy_tensor.astype(self.dtype)
            # s
            elif api_config.api_name in ["paddle.sum", "paddle.squeeze"]:
                if self.check_arg(api_config, 1, "axis"):
                    self.numpy_tensor = self.generate_random_axes(api_config)
            elif api_config.api_name in ["paddle.split"]:
                if self.check_arg(api_config, 2, "axis"):
                    x_shape = self.get_arg(api_config, 0, "x").shape
                    num_or_sections = self.get_arg(api_config, 1, "num_or_sections")
                    if isinstance(num_or_sections, (list, tuple)):
                        neg_one_count = sum(1 for x in num_or_sections if x == -1)
                        if neg_one_count > 1:
                            raise ValueError(
                                f"num_or_sections can contain at most one -1, but got {num_or_sections}"
                            )
                        num_splits = len(num_or_sections)
                        known_size = sum(num_or_sections) + neg_one_count
                    elif isinstance(num_or_sections, int):
                        num_splits = num_or_sections
                        known_size = None
                    else:
                        raise ValueError(
                            f"num_or_sections must be an int, list, or tuple, but got {type(num_or_sections)}"
                        )
                    
                    target_dim = None
                    max_dim = len(x_shape)
                    if max_dim == 0:
                        target_dim = numpy.random.randint(-1, 0)
                    else:
                        for dim in range(max_dim):
                            dim_size = x_shape[dim]
                            if isinstance(num_or_sections, int) and dim_size % num_splits == 0:
                                target_dim = dim
                            elif isinstance(num_or_sections, (list, tuple)):
                                if neg_one_count == 0 and dim_size == known_size:
                                    target_dim = dim
                                elif neg_one_count == 1 and dim_size > known_size:
                                    target_dim = dim
                    if target_dim is None:
                        raise ValueError(
                            f"No valid axis found for paddle.split with x.shape={x_shape} and num_or_sections={num_or_sections}"
                        )
                        
                    shape_len = len(self.shape)
                    if shape_len == 0:
                        self.numpy_tensor = numpy.array(target_dim, dtype=self.dtype)
                    elif shape_len == 1 and self.shape[0] == 1:
                        self.numpy_tensor = numpy.array([target_dim], dtype=self.dtype)
                    else:
                        raise ValueError(
                            f"Invalid shape for 'axis' Tensor in paddle.split. "
                            f"Expected a 0-D or 1-D Tensor, but got shape {self.shape}."
                        )
            # t
            elif api_config.api_name in ["paddle.Tensor.take_along_axis", "paddle.take_along_axis"]:
                if self.check_arg(api_config, 1, "indices"):
                    arr = self.get_arg(api_config, 0, "arr")
                    min_dim = min(arr.shape)
                    indices = (numpy.random.randint(0, min_dim-1, size=self.numel())).astype("int64")
                    self.numpy_tensor = indices.reshape(self.shape)
                    self.dtype = "int64"
                    
            elif api_config.api_name in ["paddle.Tensor.clip"]:
                self.numpy_tensor=numpy.random.random()-0.5
                if index==2:
                    pre=self.get_arg(api_config,1)
                    self.numpy_tensor=numpy.clip(self.numpy_tensor,pre.numpy_tensor,None)
                     
            elif api_config.api_name in ["paddle.Tensor.expand"]:
                if index>0:
                    d=self.get_arg(api_config,0)
                    s=d.shape
                    if len(s)==0 or s[index-1]==1:
                        self.numpy_tensor = (numpy.random.randint(1, 127, size=self.shape)).astype(self.dtype)
                    else:
                        self.numpy_tensor = numpy.array(s[index-1])

            elif api_config.api_name in ['paddle.Tensor.gather',"paddle.gather"]:
                if index==1:
                    if 'x' in api_config.kwargs:
                        s=self.get_arg(api_config,arg_name='x')
                    else:
                        s=self.get_arg(api_config,0)

                    if 'axis' in api_config.kwargs:
                        tmp=self.get_arg(api_config,arg_name='axis')
                        if isinstance(tmp,TensorConfig):
                            tmp=tmp.shape
                            tmp=tmp[0]
                    else:
                        tmp=0
                    self.numpy_tensor = (numpy.random.randint(0,s.shape[tmp], size=self.shape)).astype(self.dtype)
                elif index==2:
                    self.numpy_tensor = (numpy.random.randint(0,2, size=self.shape)).astype(self.dtype)

            elif api_config.api_name in ["paddle.Tensor.gather_nd","paddle.gather_nd"]:
                if index==1:
                    if 'x' in api_config.kwargs:
                        org=self.get_arg(api_config,arg_name='x')
                    else:
                        org=self.get_arg(api_config,0)
                    org=org.shape
                    if 'index' in api_config.kwargs:
                        s=self.get_arg(api_config,arg_name='index')
                    else:
                        s=self.get_arg(api_config,1)
                    s=s.shape
                    self.numpy_tensor=numpy.zeros(s)
                    for i in range(s[-1]):
                        self.numpy_tensor[...,i]=(numpy.random.randint(0,org[i], size=self.numpy_tensor[...,i].shape)).astype(self.dtype)



            # u
            elif api_config.api_name in ["paddle.unsqueeze"]:
                if self.check_arg(api_config, 1, "axis"):
                    max_dim = len(self.get_arg(api_config, 0, "x").shape)
                    max_dim += 1

                    shape_len = len(self.shape)
                    if shape_len == 0:
                        dim = random.randint(0, max_dim - 1)
                        if random.choice([True, False]):
                            dim -= max_dim
                        self.numpy_tensor = numpy.array(dim, dtype=self.dtype)
                    elif shape_len == 1:
                        all_dims = list(range(max_dim))
                        random_dims = random.sample(all_dims, self.shape[0])
                        final_dims = []
                        for dim in random_dims:
                            if random.choice([True, False]):
                                dim -= max_dim
                            final_dims.append(dim)
                        self.numpy_tensor = numpy.array(final_dims, dtype=self.dtype)
                    else:
                        raise ValueError(
                            f"Invalid shape for 'axis' Tensor in paddle.unsqueeze. "
                            f"Expected a 0-D or 1-D Tensor, but got shape {self.shape}."
                        )
            # v
            # w
            # x
            # y
            # z
            elif api_config.api_name in ["paddle.zeros"]:
                self.numpy_tensor = numpy.random.randint(0, 2048, size = self.shape)
            # _
            elif api_config.api_name in ["paddle.Tensor.__getitem__","paddle.Tensor.__setitem__"] and (len(api_config.args) > 1 and str(api_config.args[1]) == str(self) or str(api_config.args[0]) != str(self)):
                arr = self.get_arg(api_config, 0, "arr")
                min_dim = min(arr.shape)
                indices = (numpy.random.randint(0, min_dim, size=self.numel())).astype("int64")
                self.numpy_tensor = indices.reshape(self.shape)
            if self.numpy_tensor is None:
                if USE_CACHED_NUMPY:
                    dtype = "float32" if self.dtype == "bfloat16" else self.dtype
                    self.numpy_tensor = self.get_cached_numpy(dtype, self.shape)
                else:
                    if "int" in self.dtype:
                        self.numpy_tensor = (numpy.random.randint(-65535, 65535, size=self.shape)).astype(self.dtype)
                    else:
                        dtype = "float32" if self.dtype == "bfloat16" else self.dtype
                        self.numpy_tensor = (numpy.random.random(self.shape) - 0.5).astype(dtype)

        return self.numpy_tensor, api_config
    
    def set_index(self,index):
        self.index = index
    
    def set_key(self,key):
        self.key = key

    def set_fix(self, fix):
        self.fix = fix
        

    def get_paddle_tensor(self, api_config):
        if self.dtype in ["float8_e5m2", "float8_e4m3fn"]:
            print("Warning ", self.dtype, "not supported")
            return

        if self.paddle_tensor is None:
            tensor, api_config = self.get_numpy_tensor(api_config)
            self.paddle_tensor = paddle.to_tensor(
                tensor,
                dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            )
            self.paddle_tensor.stop_gradient = True
            if self.dtype in ['float32', 'float64', 'float16', 'complex64', 'complex128', 'bfloat16']:
                if self.dtype == "bfloat16":
                    self.paddle_tensor = paddle.cast(self.paddle_tensor, dtype="uint16")
                self.paddle_tensor.stop_gradient = False
        
        return self.paddle_tensor, api_config
    
    def get_torch_tensor(self, api_config):
        if self.dtype in ["float8_e5m2", "float8_e4m3fn"]:
            print("Warning ", self.dtype, "not supported")
            return

        device = torch.device("cuda:0")
        torch.set_default_device(device)
        tensor, api_config = self.get_numpy_tensor(api_config)
        if self.torch_tensor is None:
            self.torch_tensor = torch.tensor(
                tensor,
                dtype=self.convert_dtype_to_torch_type(self.dtype)
                if self.dtype != 'bfloat16'
                else torch.float32,
                requires_grad=True if self.dtype in ['float32', 'float64', 'float16', 'complex64', 'complex128', 'bfloat16'] else False,
            )
            if self.dtype == "bfloat16":
                self.torch_tensor = self.torch_tensor.to(dtype=torch.bfloat16)
        return self.torch_tensor, api_config
    
    def clear_tensor(self):
        self.torch_tensor = None
        self.paddle_tensor = None
        self.numpy_tensor = None
        torch.cuda.empty_cache()
        paddle.device.cuda.empty_cache()

    def clear_paddle_tensor(self):
        del self.paddle_tensor
        self.paddle_tensor = None
        paddle.device.cuda.empty_cache()
    
    def clear_numpy_tensors(self):
        del self.numpy_tensor
        self.numpy_tensor = None

    def clear_torch_tensor(self):
        del self.torch_tensor
        self.torch_tensor = None
        torch.cuda.empty_cache()

    def fill_numpy_tensor(self, full_value):
        self.numpy_tensor = numpy.full(shape=self.shape, fill_value=full_value, dtype=self.dtype)

    def check_arg(self, api_config, arg_pos=-1, arg_name=None):
        """Checks if the argument in api_config matches this instance"""
        if arg_pos is not None and 0 <= arg_pos < len(api_config.args):
            return str(api_config.args[arg_pos]) == str(self)
        if arg_name and arg_name in api_config.kwargs:
            return str(api_config.kwargs[arg_name]) == str(self)        
        elif arg_pos > -1 and len(api_config.args) > arg_pos:
            return str(api_config.args[arg_pos]) == str(self)
        return False
    
    def get_arg(self, api_config, arg_pos=-1, arg_name=None):
        """Get the argument value from the api_config"""
        if arg_name and arg_name in api_config.kwargs:
            return api_config.kwargs[arg_name]
        elif arg_pos > -1 and len(api_config.args) > arg_pos:
            return api_config.args[arg_pos]

        return None

class APIConfig:
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.args = copy.deepcopy(self.args)
        result.kwargs = copy.deepcopy(self.kwargs)
        result.api_name = self.api_name
        return result

    def __init__(self, config):
        config = config.replace("\n", "")
        self.config = config
        self.args = []
        self.kwargs = collections.OrderedDict()
        self.maxvalue = -1
        self.fix = False
        self.intnum = 1
        self.firstshape = None
        config = config.replace("Tensor(", "TensorConfig(")

        self.api_name, offset = self.get_api(config)

        if self.api_name == "paddle.einsum":
            tmp = config[config.index("\"") + 1:]
            value = tmp[:tmp.index("\"")]
            offset = config.index("\"") + 1 + tmp.index("\"")
            if "equation" in config:
                self.append_kwargs("equation", value)
            else:
                self.append_args(value)

        index = 0
        while(True):
            tocken, offset = self.get_tocken(config, offset)
            if offset is None:
                return

            key = "null"
            is_kwarg = config[offset] == '='
            if is_kwarg:
                key = tocken
                tocken, offset = self.get_tocken(config, offset+1)

            value, offset = self.get_one_arg(tocken, config, offset, index ,key)
            
            if offset is None:
                return

            if is_kwarg:
                self.append_kwargs(key, value)
            else:
                self.append_args(value)
            index += 1

    def append_args(self, arg):
        self.args.append(arg)
        
    def append_kwargs(self, name, arg):
        self.kwargs[name] = arg

    def dump_item_str(self, item):
        type_mapping = {
            numpy.int16: int,
            numpy.int32: int,
            numpy.int64: int,
            numpy.float16: float,
            numpy.float32: float,
            numpy.float64: float,
            numpy.integer: int,
            numpy.floating: float,
            numpy.bool_: bool,
            numpy.complexfloating: complex,
            numpy.str_: str,
            numpy.bytes_: bytes,
            # numpy.unicode_: str,
        }
        for numpy_type, builtin_type in type_mapping.items():
            if isinstance(item, numpy_type):
                item = builtin_type(item)
                break

        if isinstance(item, TensorConfig):
            return str(item)
        elif isinstance(item, paddle.base.core.DataType):
            return "Dtype(" + str(item)[7:] + ")"
        elif isinstance(item, paddle.base.core.VarDesc.VarType):
            return "VarType(" + str(item)[7:] + ")"
        elif isinstance(item, list):
            result = "list["
            for sub_item in item:
                tmp = self.dump_item_str(sub_item)
                if tmp == "":
                    return ""
                result = result + tmp + ","
            result = result + "]"
            return result
        elif isinstance(item, tuple):
            result = "tuple("
            for sub_item in item:
                tmp = self.dump_item_str(sub_item)
                if tmp == "":
                    return ""
                result = result + tmp + ","
            result = result + ")"
            return result
        elif isinstance(item, slice):
            return (
                "slice("
                + str(item.start)
                + ","
                + str(item.stop)
                + ","
                + str(item.step)
                + ")"
            )
        elif isinstance(item, complex):
            return (
                "complex("
                + self.dump_item_str(item.real)
                + ","
                + self.dump_item_str(item.imag)
                + ")"
            )
        elif item is None:
            return "None"
        elif isinstance(
            item, (paddle.base.Variable, paddle.base.libpaddle.pir.Value)
        ):
            return ""
        elif item == math.inf:
            return "math.inf"
        elif item == -math.inf:
            return "-math.inf"
        elif item == math.nan:
            return "math.nan"
        elif item == -math.nan:
            return "-math.nan"
        elif isinstance(item, (bool, int, float)):
            return str(item)
        elif isinstance(item, str):
            return '"' + item + '"'
        elif isinstance(item, type):
            return (
                "type("
                + str(item)[str(item).index("'") + 1 : str(item).rindex("'")]
                + ")"
            )
        else:
            return str(item)


    def __str__(self):
        result = ""
        result = result + self.api_name + "("
        for arg in self.args:
            result = result + self.dump_item_str(arg) + ", "
        
        for key, value in self.kwargs.items():
            result = result + key + "=" + self.dump_item_str(value) + ", "

        result = result + ")"
        return result

    # def get_tocken(self, config, offset):
    #     def is_int(tocken):
    #         try:
    #             int(tocken)
    #             return True
    #         except Exception as err:
    #             return False
    #     pattern = r'\b[A-Za-z0-9._+-]+\b|-[A-Za-z0-9._+-]+\b'
    #     match = re.search(pattern, config[offset:])
    #     if match:
    #         if is_int(match.group()) and config[offset + match.start() + len(match.group())] == ".":
    #             return match.group()+".", offset + match.start() + len(match.group()) + 1
    #         return match.group(), offset + match.start() + len(match.group())
    #     return None, None

    def get_tocken(self,config, offset):
        def is_int(token):
            try:
                int(token)
                return True
            except Exception as err:
                return False

        # Modified pattern to handle decimal numbers starting with dot
        pattern = r'\b[A-Za-z0-9._+-]+\b|-[A-Za-z0-9._+-]+\b|\.[0-9]+'
        match = re.search(pattern, config[offset:])
        if match:
            token = match.group()
            # Handle the case where token starts with dot followed by digits
            if token.startswith('.') and token[1:].isdigit():
                return token, offset + match.start() + len(token)

            if is_int(token) and offset + match.start() + len(token) < len(config) and config[
                offset + match.start() + len(token)] == ".":
                return token + ".", offset + match.start() + len(token) + 1
            return token, offset + match.start() + len(token)
        return None, None


    def get_api(self, config):
        return config[0:config.index("(")], len(config[0:config.index("(")])

    def get_tensor(self, config, offset, index, key):
        config = config[offset:]
        tensor_str = config[config.index("TensorConfig"):config.index(")")+1]
        tensor = eval(tensor_str)
        tensor.set_index(index)
        tensor.set_key(key)
        if not "Tensor" in config[1:]:
            tensor.set_fix(True) 
        if self.firstshape == None:
            self.firstshape = tensor.shape
        return tensor, offset + len(tensor_str)

    def get_dtype(self, config, offset):
        tocken, offset = self.get_tocken(config, offset)
        return paddle.pir.core.convert_np_dtype_to_dtype_(tocken), offset

    def get_vartype(self, config, offset):
        tocken, offset = self.get_tocken(config, offset)
        return paddle.base.framework.convert_np_dtype_to_proto_type(tocken), offset

    def get_list(self, config, offset, index, key):
        result = []
        tmp = 0
        last_index = offset
        for i in range(offset, len(config)):
            if config[i] == "[":
                tmp = tmp + 1
            if config[i] == "]":
                tmp = tmp - 1
            if tmp == 0:
                last_index = i
                break
        
        list_str = config[offset: last_index+1]
        if "TensorConfig" not in list_str:
            list_str = list_str.replace(",", " ")

        offset = 1
        indice = 0
        while(True):
            tocken, offset = self.get_tocken(list_str, offset)
            if offset is None:
                break

            value, offset = self.get_one_arg(tocken, list_str, offset, index, key, indice=indice)

            if offset is None:
                break

            result.append(value)
            indice += 1

        return result, last_index+1

    def get_tuple(self, config, offset, index, key):
        result = []
        tmp = 0
        last_index = offset
        for i in range(offset, len(config)):
            if config[i] == "(":
                tmp = tmp + 1
            if config[i] == ")":
                tmp = tmp - 1
            if tmp == 0:
                last_index = i
                break
        
        tuple_str = config[offset: last_index+1]

        tuple_str = tuple_str.replace(",", " , ")
        
        indice = 0
        offset = 1
        while(True):
            tocken, offset = self.get_tocken(tuple_str, offset)
            if offset is None:
                break
            value, offset = self.get_one_arg(tocken, tuple_str, offset, index, key, indice=indice)

            if offset is None:
                break

            result.append(value)
            indice += 1

        return tuple(result), last_index+1

    def get_slice(self, config, offset):
        config = config[offset:]
        slice_str = config[config.index("("):config.index(")")+1]
        return eval("slice"+slice_str), offset+len(slice_str)

    def get_complex(self, config, offset):
        config = config[offset:]
        complex_str = config[config.index("("):config.index(")")+1]
        if "nan" in complex_str and complex_str[complex_str.index('nan')-1] != ".":
            complex_str = complex_str.replace("nan", "float('nan')")
        return eval("complex"+complex_str), offset+len(complex_str)

    def get_numpy_type(self, config, offset):
        config = config[offset:]
        numpy_type_str = config[config.index("(")+1:config.index(")")]
        if numpy_type_str == "numpy.bool":
            return numpy.bool_, offset+len(numpy_type_str)+2
        return eval(numpy_type_str), offset+len(numpy_type_str)+2

    def get_one_arg(self, tocken, config, offset, index, key, next = True, indice = None):
        if tocken == "TensorConfig":
            value, offset = self.get_tensor(config, offset-len(tocken), index, key)
        elif tocken == "Dtype":
            value, offset = self.get_dtype(config, offset)
        elif tocken == "VarType":
            value, offset = self.get_vartype(config, offset)
        elif tocken == "list":
            value, offset = self.get_list(config, offset, index, key)
        elif tocken == "tuple":
            value, offset = self.get_tuple(config, offset, index, key)
        elif tocken == "slice":
            value, offset = self.get_slice(config, offset)
        elif tocken == "complex":
            value, offset = self.get_complex(config, offset)
        elif tocken == "type":
            value, offset = self.get_numpy_type(config, offset)
        elif tocken == "nan":
            value = float('nan')
        elif tocken is not None and config[offset - len(tocken) - 1] == "\"":
            # fix tocken is not correct in str with spaces
            next_quote_idx  = config.index("\"", offset)
            value = config[offset - len(tocken):next_quote_idx]
            offset = next_quote_idx
        elif tocken is None:
            return None, None
        else:
            if tocken[0]=='.':
                tocken='0'+tocken
            value = eval(tocken)
            if self.api_name in ["paddle.Tensor.reshape","paddle.reshape"]:
                if "int" in str(type(value)):
                    if value == 0:
                        self.intnum = self.intnum * self.firstshape[indice]
                    elif not value == -1:
                        self.intnum = self.intnum * value
        return value, offset


def analyse_configs(config_path):
    with open(config_path, "r") as f:
        configs = f.readlines()
        f.close()

    api_configs = []
    for config in configs:
        # print(config)
        api_config = APIConfig(config)
        api_configs.append(api_config)
    return api_configs
