from safetensors.torch import load_file, save_file
import torch

def map_lora_to_diffusers(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        # 更宽松的映射规则，避免过度过滤
        if "lora_unet_img_in" in key:
            new_key = key.replace("lora_unet_img_in", "transformer.input_blocks.0")
        elif "lora_unet_txt_in" in key:
            new_key = key.replace("lora_unet_txt_in", "transformer.text_blocks.0")
        else:
            new_key = key  # 保留所有其他键
        new_state_dict[new_key] = value

    # 不再强制过滤，只打印键数量
    print("Converted key count:", len(new_state_dict))
    return new_state_dict

def load_and_convert_lora(lora_path, output_path="converted_lora.safetensors"):
    # 加载原始文件
    state_dict = load_file(lora_path, device="cpu")
    print("Original key count:", len(state_dict))
    print("Original keys:", list(state_dict.keys())[:10])  # 打印前10个键
    
    # 转换状态字典
    converted_state_dict = map_lora_to_diffusers(state_dict)
    
    # 保存为 safetensors 格式，优化元数据
    save_file(converted_state_dict, output_path, metadata=None)
    
    # 验证保存后的文件大小
    import os
    print(f"Saved file size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    return converted_state_dict

# 调用
lora_path = "xin-000001.safetensors"
converted_state_dict = load_and_convert_lora(lora_path)