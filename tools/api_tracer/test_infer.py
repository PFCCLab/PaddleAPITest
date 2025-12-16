from __future__ import annotations

import os

os.environ["HF_HOME"] = "tools/api_tracer/.huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import traceback
from pathlib import Path

import torch
import torchvision.transforms as T
from api_tracer import APITracer
from diffusers.pipelines.auto_pipeline import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

MODELS_DIR = Path("/root/paddlejob/workspace/env_run/models")
# MODELS_DIR = Path("/root/paddlejob/workspace/env_run/bos/huggingface")

TextGenerationMODELS = [
    # "Qwen/Qwen2-0.5B",
    # "Qwen/Qwen2-57B-A14B",
    # "Qwen/Qwen2.5-0.5B",
    # "Qwen/Qwen3-0.6B",
    # "Qwen/Qwen3-30B-A3B",
    # "meta-llama/Llama-2-7b-hf",
    # "meta-llama/Llama-3.1-8B"
    # "deepseek-ai/DeepSeek-V2-Lite",  # need transformers<4.49
    # "deepseek-ai/DeepSeek-V3",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    # "baidu/ERNIE-4.5-0.3B-PT",
    # "baidu/ERNIE-4.5-21B-A3B-PT",
    # "moonshotai/Kimi-K2-Instruct",
    # "zai-org/GLM-4.5",
    # "mistralai/Magistral-Small-2507",  # need install mistral-common
    # "MiniMaxAI/MiniMax-M1-40k",
    # "state-spaces/mamba2-2.7b",
    # "RWKV/RWKV7-Goose-World3-2.9B-HF",  #  maybe fail, change to fla-hub/rwkv7-2.9B-world, need transformers<4.50
    # "fla-hub/rwkv7-2.9B-world",
]

ImageTexttoTextModels = [
    # "Qwen/Qwen2.5-VL-3B-Instruct",
    # "deepseek-ai/deepseek-vl2-tiny",  # need to clone deepseek_vl2 project
    # "llava-hf/llava-1.5-7b-hf",
    # "meta-llama/Llama-4-Maverick-17B-128E",
    # "baidu/ERNIE-4.5-VL-28B-A3B-PT",  # need transformers<4.54
    # "zai-org/GLM-4.1V-9B-Thinking",
    # "ByteDance/Dolphin",
    # "Salesforce/blip2-opt-2.7b",  # need transformers<4.50
    # "OpenGVLab/InternVL3-1B",
    # "moonshotai/Kimi-VL-A3B-Instruct",  # need transformers<4.50
    # "XiaomiMiMo/MiMo-VL-7B-SFT",
    # "echo840/MonkeyOCR",  # need to clone MonkeyOCR project
]

VideoTexttoTextModels = [
    # "Kwai-Keye/Keye-VL-8B-Preview",
]

TexttoImageModels = [
    # "stabilityai/stable-diffusion-3-medium-diffusers",
    # "black-forest-labs/FLUX.1-dev",
    # "jieliu/SD3.5M-FlowGRPO-GenEval",
]

TexttoVideoModels = [
    # "Wan-AI/Wan2.1-T2V-14B-Diffusers",
]

Imageto3DModels = [
    # "tencent/HunyuanWorld-1",
]

AnytoAnyModels = [
    # "deepseek-ai/Janus-Pro-1B",
    # "ByteDance-Seed/BAGEL-7B-MoT",
]


def run_inference_test_tg(model_name: str):
    print(f"🚀 Running Text Generation Inference Test for: {model_name}")
    model_path = MODELS_DIR / model_name
    output_path = f"tools/api_tracer/trace_output_test_infer/{model_name}"
    tracer = APITracer("torch", output_path=output_path, levels=[0, 1], merge_output=True)
    tracer.start()

    try:
        if ("Qwen" in model_name or "baidu" in model_name) and "A" not in model_name:
            device_map = "cuda:0"
        else:
            device_map = "auto"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        print(f"Model Class: {model.__class__}")
        print(f"Tokenizer Class: {tokenizer.__class__}")

        with open(os.path.join(output_path, "model_info.txt"), "w") as f:
            f.write(f"Model: {model.__class__}\n")
            f.write(f"Tokenizer: {tokenizer.__class__}\n")

        prompt = "Hello! Can you tell me how to learn PyTorch?"
        if "RWKV" in model_name:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer([text], return_tensors="pt").to("cuda")
        else:
            inputs = tokenizer(
                prompt, return_tensors="pt", padding=True, return_attention_mask=True
            ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                num_return_sequences=1,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                attention_mask=inputs["attention_mask"],
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n--- Generated Response ---")
        print(response)
        print("--------------------------\n")
        print(f"✅ Test for {model_name} finished.")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ An error occurred during inference for {model_name}: {e}")
    finally:
        tracer.stop()


def run_inference_test_i2t(model_name: str):
    print(f"🚀 Running Image2Text Inference Test for: {model_name}")
    model_path = MODELS_DIR / model_name
    output_path = f"tools/api_tracer/trace_output_test_infer/{model_name}"
    os.makedirs(output_path, exist_ok=True)
    tracer = APITracer("torch", output_path=output_path, levels=[0, 1], merge_output=True)
    tracer.start()

    try:
        if "OpenGVLab" in model_name:
            model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        elif "baidu" in model_name or "moonshotai" in model_name or "deepseek" in model_name:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        elif "Salesforce" in model_name:
            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="cuda:0",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        print(f"Model Class: {model.__class__}")
        print(f"Processor Class: {processor.__class__}")

        with open(os.path.join(output_path, "model_info.txt"), "w") as f:
            f.write(f"Model: {model.__class__}\n")
            f.write(f"Processor: {processor.__class__}\n")

        question = "What is in this image?"
        image_path = "tools/api_tracer/sample_image.jpg"
        if "OpenGVLab" in model_name:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, use_fast=False
            )
            prompt = f"<image>\n{question}"
            image = Image.open(image_path).convert("RGB")
            transform = T.Compose(
                [
                    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
            pixel_values = transform(image).unsqueeze(0).to("cuda", torch.bfloat16)
        elif "baidu" in model_name:
            model.add_image_preprocess(processor)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_path},
                        },
                    ],
                }
            ]
            text = processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            image_inputs, video_inputs = processor.process_vision_info(conversation)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda", torch.bfloat16)
        elif "Salesforce" in model_name:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=[image], text=question, return_tensors="pt").to(
                "cuda", torch.bfloat16
            )
        else:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image"},
                    ],
                }
            ]
            if hasattr(processor, "chat_template") and processor.chat_template is not None:
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            else:
                prompt = f"<|im_start|>user\n<image>\n{question}<|im_end|>\n<|im_start|>assistant\n"
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=[image], text=prompt, return_tensors="pt").to(
                "cuda", torch.bfloat16
            )

        if "OpenGVLab" in model_name:
            generation_config = {"max_new_tokens": 1024, "do_sample": False}
            with torch.no_grad():
                outputs = model.chat(
                    tokenizer,
                    pixel_values,
                    prompt,
                    generation_config,
                )
        else:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                )

        if "OpenGVLab" in model_name:
            response = outputs
        else:
            response = processor.decode(outputs[0], skip_special_tokens=True)

        print("\n--- Generated Response ---")
        print(response)
        print("--------------------------\n")
        print(f"✅ Test for {model_name} finished.")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ An error occurred during inference for {model_name}: {e}")
    finally:
        tracer.stop()


def run_inference_test_v2t(model_name: str):
    print(f"🚀 Running Video2Text Inference Test for: {model_name}")
    model_path = MODELS_DIR / model_name
    output_path = f"tools/api_tracer/trace_output_test_infer/{model_name}"
    os.makedirs(output_path, exist_ok=True)
    tracer = APITracer("torch", output_path=output_path, levels=[0, 1], merge_output=True)
    tracer.start()

    video_path = "tools/api_tracer/sample_video.mp4"
    try:
        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to("cuda")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        print(f"Model Class: {model.__class__}")
        print(f"Processor Class: {processor.__class__}")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 1280 * 720,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": "Describe this video."},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        from keye_vl_utils import process_vision_info

        image_inputs, video_inputs, _video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=100,
            )

        response = processor.decode(outputs[0], skip_special_tokens=True)
        print("\n--- Generated Response ---")
        print(response)
        print("--------------------------\n")
        print(f"✅ Test for {model_name} finished.")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ An error occurred during inference for {model_name}: {e}")
    finally:
        tracer.stop()


def run_inference_test_t2i(model_name: str):
    print(f"🚀 Running Text2Image Inference Test for: {model_name}")
    model_path = MODELS_DIR / model_name
    output_path = f"tools/api_tracer/trace_output_test_infer/{model_name}"
    os.makedirs(output_path, exist_ok=True)
    tracer = APITracer("torch", output_path=output_path, levels=[0, 1], merge_output=True)
    tracer.start()

    try:
        lora_ckpt_path = model_name
        if "SD3.5M-FlowGRPO-GenEval" in model_name:
            model_name = "stabilityai/stable-diffusion-3.5-medium"

        load_kwargs = {
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
            "device_map": "balanced",
        }
        if "FLUX.1-dev" not in model_name:
            load_kwargs["variant"] = "fp16"
        pipe = AutoPipelineForText2Image.from_pretrained(model_path, **load_kwargs)

        if "SD3.5M-FlowGRPO-GenEval" in lora_ckpt_path:
            from peft import PeftModel

            pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_ckpt_path)
            pipe.transformer = pipe.transformer.merge_and_unload()

        print(f"Pipeline Class: {pipe.__class__}")

        prompt = "A majestic lion jumping from a big rock, high quality, cinematic"

        with torch.no_grad(), torch.inference_mode():
            pipe(prompt=prompt, num_inference_steps=25).images[0]

        print(f"✅ Test for {model_name} finished.")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ An error occurred during inference for {model_name}: {e}")
    finally:
        tracer.stop()


def run_inference_test_t2v(model_name: str):
    print(f"🚀 Running Text2Video Inference Test for: {model_name}")
    model_path = MODELS_DIR / model_name
    output_path = f"tools/api_tracer/trace_output_test_infer/{model_name}"
    os.makedirs(output_path, exist_ok=True)
    tracer = APITracer("torch", output_path=output_path, levels=[0, 1], merge_output=True)
    tracer.start()

    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="balanced"
        )

        print(f"Pipeline Class: {pipe.__class__}")

        prompt = "A panda eating bamboo on a rock."

        with torch.no_grad(), torch.inference_mode():
            pipe(prompt, num_inference_steps=25, num_frames=20).frames

        print(f"✅ Test for {model_name} finished.")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ An error occurred during inference for {model_name}: {e}")
    finally:
        tracer.stop()


def run_inference_test_i2d(model_name: str):
    print(f"🚀 Running Image23D Inference Test for: {model_name}")
    model_path = MODELS_DIR / model_name
    output_path = f"tools/api_tracer/trace_output_test_infer/{model_name}"
    os.makedirs(output_path, exist_ok=True)
    tracer = APITracer("torch", output_path=output_path, levels=[0, 1], merge_output=True)
    tracer.start()

    image_path = "tools/api_tracer/sample_image.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Image file not found at {image_path}.")
        return

    try:
        pipe = AutoPipelineForImage2Image.from_pretrained(
            model_path, torch_dtype=torch.float16, trust_remote_code=True
        ).to("cuda")

        print(f"Pipeline Class: {pipe.__class__}")

        input_image = Image.open(image_path).convert("RGB").resize((256, 256))

        with torch.no_grad():
            pipe(input_image, num_inference_steps=64, frame_size=256, output_type="pil").images

        print(f"✅ Test for {model_name} finished.")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ An error occurred during inference for {model_name}: {e}")
    finally:
        tracer.stop()


def run_inference_test_a2a(model_name: str):
    print(f"🚀 Running Any2Any Inference Test for: {model_name}")
    model_path = MODELS_DIR / model_name
    output_path = f"tools/api_tracer/trace_output_test_infer/{model_name}"
    os.makedirs(output_path, exist_ok=True)
    tracer = APITracer("torch", output_path=output_path, levels=[0, 1], merge_output=True)
    tracer.start()

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        print(f"Model Class: {model.__class__}")
        print(f"Processor Class: {processor.__class__}")

        with open(os.path.join(output_path, "model_info.txt"), "w") as f:
            f.write(f"Model: {model.__class__}\n")
            f.write(f"Processor: {processor.__class__}\n")

        prompt = "Describe the object in the image."
        image = Image.open("tools/api_tracer/sample_image.jpg")

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=text_prompt, images=image, return_tensors="pt").to(
            "cuda", dtype=torch.bfloat16
        )

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)

        response = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

        print("\n--- Generated Response ---")
        print(response)
        print("--------------------------\n")
        print(f"✅ Test for {model_name} finished.")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ An error occurred during inference for {model_name}: {e}")
    finally:
        tracer.stop()


def main():
    for model_name in TextGenerationMODELS:
        run_inference_test_tg(model_name)

    for model_name in ImageTexttoTextModels:
        run_inference_test_i2t(model_name)

    for model_name in VideoTexttoTextModels:
        run_inference_test_v2t(model_name)

    for model_name in TexttoImageModels:
        run_inference_test_t2i(model_name)

    for model_name in TexttoVideoModels:
        run_inference_test_t2v(model_name)

    for model_name in Imageto3DModels:
        run_inference_test_i2d(model_name)

    for model_name in AnytoAnyModels:
        run_inference_test_a2a(model_name)


# api_calls = []

# def trace_function(frame, event, arg):
#     if event == "call":
#         code = frame.f_code
#         if "torch" in code.co_filename:
#             func_name = code.co_name
#             args = frame.f_locals
#             api_calls.append({"function": func_name, "module": "torch", "args": args})
#     return trace_function

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#     with record_function("model_inference"):

# sys.setprofile(trace_function)

# <code>

# sys.setprofile(None)

# print(prof.key_averages().table(sort_by="cpu_time_total"))
# prof.export_chrome_trace("tools/api_tracer/trace_output/trace.json")

# with open("tools/api_tracer/trace_output/api_calls.json", "w", encoding="utf-8") as f:
#     json.dump(api_calls, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
