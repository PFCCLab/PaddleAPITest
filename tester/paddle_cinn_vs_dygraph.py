from __future__ import annotations

import paddle
from paddle.jit import to_static

from .api_config.log_writer import write_to_log
from .base import CUDA_ERROR, CUDA_OOM, APITestBase


class APITestCINNVSDygraph(APITestBase):
    def __init__(self, api_config, **kwargs):
        super().__init__(api_config)
        self.test_amp = kwargs.get("test_amp", False)
        self.test_backward = kwargs.get("test_backward", False)

    def test(self):
        if self.need_skip():
            print("[Skip]", flush=True)
            return

        if not self.ana_paddle_api_info():
            print("ana_paddle_api_info failed", flush=True)
            return

        try:
            if not self.gen_numpy_input():
                print("gen_numpy_input failed", flush=True)
                return
        except Exception as err:
            print(f"[numpy error] {self.api_config.config}\n{err!s}", flush=True)
            write_to_log("numpy_error", self.api_config.config)
            return

        if not self.gen_paddle_input():
            print("gen_paddle_input failed", flush=True)
            return

        try:

            def func(args, kwargs):
                """Forward function"""
                if self.api_config.api_name.startswith("paddle.Tensor."):
                    api_name = self.api_config.api_name.split(".")[-1]
                    api = getattr(args[0], api_name)
                    return api(*args[1:], **kwargs)
                return self.paddle_api(*args, **kwargs)

            def func_backward(outputs_list, inputs_list, grads_input_list):
                """Backward function"""
                return paddle.grad(
                    outputs_list,
                    inputs_list,
                    grad_outputs=grads_input_list,
                    allow_unused=True,
                )

            if self.test_amp:
                with paddle.amp.auto_cast():
                    dynamic_fwd_output = func(self.paddle_args, self.paddle_kwargs)
            else:
                dynamic_fwd_output = func(self.paddle_args, self.paddle_kwargs)
        except Exception as err:
            if self.should_ignore_paddle_error(str(err)):
                print(f"[Pass] {self.api_config.config}", flush=True)
                write_to_log("pass", self.api_config.config)
                return
            if any(cuda_err in str(err) for cuda_err in CUDA_ERROR):
                print(
                    f"[cuda error] dynamic forward {self.api_config.config}\n{err!s}",
                )
                write_to_log("cuda_error", self.api_config.config)
                raise
            if any(cuda_err in str(err) for cuda_err in CUDA_OOM):
                print(
                    f"[oom] dynamic forward {self.api_config.config}\n{err!s}",
                )
                write_to_log("oom", self.api_config.config)
                raise
            print(
                f"[paddle error] dynamic forward {self.api_config.config}\n{err!s}",
                flush=True,
            )
            write_to_log("paddle_error", self.api_config.config)
            return

        try:
            paddle.base.core.eager._for_test_check_cuda_error()
        except Exception as err:
            print(
                f"[cuda error] dynamic forward {self.api_config.config}\n{err!s}",
                flush=True,
            )
            write_to_log("cuda_error", self.api_config.config)
            raise

        need_check_grad = self.test_backward and self.need_check_grad()
        if need_check_grad:
            try:
                dynamic_bwd_output = None
                dynamic_inputs_list = self.get_paddle_input_list()
                dynamic_outputs_list, dynamic_grads_input_list = (
                    self.gen_paddle_output_and_output_grad(dynamic_fwd_output)
                )
                if (
                    not dynamic_inputs_list
                    or not dynamic_outputs_list
                    or not dynamic_grads_input_list
                ):
                    need_check_grad = False
                else:
                    dynamic_bwd_output = func_backward(
                        dynamic_outputs_list,
                        dynamic_inputs_list,
                        dynamic_grads_input_list,
                    )
            except Exception as err:
                if str(err).startswith("Too large tensor to get cached numpy: "):
                    print(
                        f"[numpy error] dynamic backward {self.api_config.config}\n{err!s}",
                        flush=True,
                    )
                    write_to_log("numpy_error", self.api_config.config)
                    return
                if self.should_ignore_paddle_error(str(err)):
                    print(f"[Pass] {self.api_config.config}", flush=True)
                    write_to_log("pass", self.api_config.config)
                    return
                if any(cuda_err in str(err) for cuda_err in CUDA_ERROR):
                    print(
                        f"[cuda error] dynamic backward {self.api_config.config}\n{err!s}",
                        flush=True,
                    )
                    write_to_log("cuda_error", self.api_config.config)
                    raise
                if any(cuda_err in str(err) for cuda_err in CUDA_OOM):
                    print(
                        f"[oom] dynamic backward {self.api_config.config}\n{err!s}",
                        flush=True,
                    )
                    write_to_log("oom", self.api_config.config)
                    raise
                print(
                    f"[paddle error] dynamic backward {self.api_config.config}\n{err!s}",
                    flush=True,
                )
                write_to_log("paddle_error", self.api_config.config)
                return

            try:
                paddle.base.core.eager._for_test_check_cuda_error()
            except Exception as err:
                print(
                    f"[cuda error] dynamic backward {self.api_config.config}\n{err!s}",
                    flush=True,
                )
                write_to_log("cuda_error", self.api_config.config)
                raise

        try:
            build_strategy = paddle.static.BuildStrategy()
            build_strategy.build_cinn_pass = True

            @to_static(full_graph=True, build_strategy=build_strategy)
            def run_static(
                args,
                kwargs,
                static_grads_input_list=None,
            ):
                if self.test_amp:
                    with paddle.amp.auto_cast():
                        static_fwd_output = func(args, kwargs)
                else:
                    static_fwd_output = func(args, kwargs)

                if not need_check_grad:
                    return static_fwd_output, None

                # gen_paddle_output_and_output_grad can not be traced in static graph mode,
                # so we flatten the outputs here simply.
                static_outputs_list = []
                if paddle.is_tensor(static_fwd_output):
                    static_outputs_list.append(static_fwd_output)
                elif isinstance(static_fwd_output, (list, tuple)):
                    for out in static_fwd_output:
                        if paddle.is_tensor(out):
                            static_outputs_list.append(out)

                # get_paddle_input_list can not be traced in static graph mode as well,
                # and inputs_list can not be used here, so we copy it here.
                static_inputs_list = []
                for arg in args:
                    if paddle.is_tensor(arg):
                        static_inputs_list.append(arg)
                    elif isinstance(arg, (tuple, list)):
                        for item in arg:
                            if paddle.is_tensor(item):
                                static_inputs_list.append(item)
                for key in getattr(self, "paddle_merged_kwargs_config", []):
                    if key in kwargs:
                        value = kwargs[key]
                        if paddle.is_tensor(value):
                            static_inputs_list.append(value)
                        elif isinstance(value, (tuple, list)):
                            for item in value:
                                if paddle.is_tensor(item):
                                    static_inputs_list.append(item)
                else:  #  paddle_only
                    for key, value in kwargs.items():
                        if paddle.is_tensor(value):
                            static_inputs_list.append(value)
                        elif isinstance(value, (tuple, list)):
                            for item in value:
                                if paddle.is_tensor(item):
                                    static_inputs_list.append(item)

                # static_grads_input_list is as same as dynamic_grads_input_list, generated in graph mode but used in static graph mode.
                # Note that its shape and dtype may differ from static graph mode outputs.
                if not static_inputs_list or not static_outputs_list or not static_grads_input_list:
                    return static_fwd_output, None

                static_bwd_output = func_backward(
                    static_outputs_list, static_inputs_list, static_grads_input_list
                )
                return static_fwd_output, static_bwd_output

            if need_check_grad:
                static_fwd_output, static_bwd_output = run_static(
                    self.paddle_args, self.paddle_kwargs, dynamic_grads_input_list
                )
            else:
                static_fwd_output, static_bwd_output = run_static(
                    self.paddle_args, self.paddle_kwargs
                )
        except Exception as err:
            if str(err).startswith("Too large tensor to get cached numpy: "):
                print(
                    f"[numpy error] static backward {self.api_config.config}\n{err!s}",
                    flush=True,
                )
                write_to_log("numpy_error", self.api_config.config)
                return
            if self.should_ignore_paddle_error(str(err)):
                print(f"[Pass] {self.api_config.config}", flush=True)
                write_to_log("pass", self.api_config.config)
                return
            if any(cuda_err in str(err) for cuda_err in CUDA_ERROR):
                print(
                    f"[cuda error] static {self.api_config.config}\n{err!s}",
                )
                write_to_log("cuda_error", self.api_config.config)
                raise
            if any(cuda_err in str(err) for cuda_err in CUDA_OOM):
                print(
                    f"[oom] static {self.api_config.config}\n{err!s}",
                )
                write_to_log("oom", self.api_config.config)
                raise
            print(
                f"[paddle error] static {self.api_config.config}\n{err!s}",
                flush=True,
            )
            write_to_log("paddle_error", self.api_config.config)
            return

        try:
            paddle.base.core.eager._for_test_check_cuda_error()
        except Exception as err:
            print(
                f"[cuda error] static {self.api_config.config}\n{err!s}",
                flush=True,
            )
            write_to_log("cuda_error", self.api_config.config)
            raise

        if not self.compare(dynamic_fwd_output, static_fwd_output):
            return

        if need_check_grad:
            if not self.compare(dynamic_bwd_output, static_bwd_output, is_backward=True):  # type: ignore[reportGeneralTypeIssues]
                return

        print(f"[Pass] {(self.api_config.config,)}\n", flush=True)
        write_to_log("pass", self.api_config.config)

    def compare(self, dygraph_output, static_output, is_backward=False):
        backward_str = "backward " if is_backward else ""
        self.is_backward = is_backward
        if isinstance(dygraph_output, paddle.Tensor):
            if not isinstance(static_output, paddle.Tensor):
                print(
                    f"[match error] {backward_str}{self.api_config.config}\ntype not match,",
                    f"dygraph: {type(dygraph_output)}, static: {type(static_output)}\n",
                    flush=True,
                )
                write_to_log("match_error", self.api_config.config)
                return False
            try:
                self.paddle_assert_accuracy(dygraph_output, static_output)
            except Exception as err:
                print(
                    f"[accuracy error] {backward_str}{self.api_config.config}\n{err!s}",
                    flush=True,
                )
                write_to_log("accuracy_error", self.api_config.config)
                return False
        elif isinstance(dygraph_output, (list, tuple)):
            if not isinstance(static_output, (list, tuple)):
                print(
                    f"[match error] {backward_str}{self.api_config.config}\ntype not match,",
                    f"dygraph: {type(dygraph_output)}, static: {type(static_output)}\n",
                    flush=True,
                )
                write_to_log("match_error", self.api_config.config)
                return False
            dygraph_output = list(dygraph_output)
            static_output = list(static_output)
            if len(dygraph_output) != len(static_output):
                print(
                    f"[match error] {backward_str}{self.api_config.config}\nlength not match,",
                    f"dygraph: {len(dygraph_output)}, static: {len(static_output)}\n",
                    flush=True,
                )
                write_to_log("match_error", self.api_config.config)
                return False
            for i, (dygraph_item, static_item) in enumerate(
                zip(dygraph_output, static_output, strict=False)
            ):
                if dygraph_item is None and static_item is None:
                    continue
                if not isinstance(dygraph_item, paddle.Tensor) or not isinstance(
                    static_item, paddle.Tensor
                ):
                    print(
                        f"[match error] {backward_str}{self.api_config.config}\ntype not match at {i},",
                        f"dygraph: {type(dygraph_item)}, static: {type(static_item)}\n",
                        flush=True,
                    )
                    write_to_log("match_error", self.api_config.config)
                    return False
                try:
                    self.paddle_assert_accuracy(dygraph_item, static_item)
                except Exception as err:
                    print(
                        f"[accuracy error] {backward_str}{self.api_config.config}\n{err!s}",
                        flush=True,
                    )
                    write_to_log("accuracy_error", self.api_config.config)
                    return False
        elif dygraph_output is None and static_output is None:
            pass
        else:
            print(
                f"[match error] {backward_str}{self.api_config.config}\ntype not match,",
                f"dygraph: {type(dygraph_output)}, static: {type(static_output)}\n",
                flush=True,
            )
            write_to_log("match_error", self.api_config.config)
            return False
        return True
