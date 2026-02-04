import os
import shutil
from pathlib import Path

def copy_dataset(source_dir, target_dir, move_files=False):
    """
    将图片和对应的标签文件从源目录复制/移动到目标目录。
    
    :param source_dir: 源文件夹路径
    :param target_dir: 目标文件夹路径
    :param move_files: True为移动(剪切)，False为复制(默认)
    """
    
    # 1. 配置支持的文件扩展名
    # 图片扩展名 (大小写不敏感)
    valid_image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    # 对应的标签数据扩展名
    valid_data_exts = {'.txt', '.xml', '.json'} 

    # 转换路径对象
    src_path = Path(source_dir)
    dst_path = Path(target_dir)

    # 检查源目录是否存在
    if not src_path.exists():
        print(f"❌ 错误：源文件夹 '{source_dir}' 不存在！")
        return

    # 如果目标目录不存在，则创建
    if not dst_path.exists():
        dst_path.mkdir(parents=True, exist_ok=True)
        print(f"📂 已创建目标文件夹：{target_dir}")

    # 计数器
    count_pairs = 0
    count_missing_data = 0

    print(f"🚀 开始处理：从 [{source_dir}] 到 [{target_dir}] ...\n")

    # 2. 遍历源文件夹中的所有文件
    for file_path in src_path.iterdir():
        if file_path.is_file():
            # 获取文件名后缀（转小写）
            suffix = file_path.suffix.lower()

            # 如果是图片文件
            if suffix in valid_image_exts:
                image_stem = file_path.stem  # 获取文件名（不带后缀），例如 'data_123'
                
                # 寻找对应的标签文件
                found_data_file = None
                for data_ext in valid_data_exts:
                    # 尝试拼接可能的标签文件名，例如 'data_123.txt'
                    potential_data_path = src_path / (image_stem + data_ext)
                    
                    if potential_data_path.exists():
                        found_data_file = potential_data_path
                        break # 找到一个就停止，避免重复
                
                # 执行复制操作
                try:
                    # 定义操作函数 (复制 or 移动)
                    action_func = shutil.move if move_files else shutil.copy2
                    action_name = "移动" if move_files else "复制"

                    # 2. 处理标签文件 (如果存在)
                    if found_data_file:
                        action_func(str(file_path), str(dst_path / file_path.name))
                        action_func(str(found_data_file), str(dst_path / found_data_file.name))
                        print(f"✅ [{action_name}] {file_path.name} + {found_data_file.name}")
                        count_pairs += 1
                    else:
                        # 仅复制了图片，没找到标签
                        print(f"⚠️ [警告] 仅{action_name}图片（无标签）：{file_path.name}")
                        count_missing_data += 1
                        
                except Exception as e:
                    print(f"❌ 处理文件 {file_path.name} 时出错: {e}")

    # 3. 总结
    print("-" * 30)
    print(f"🏁 处理完成！")
    print(f"   成功处理成对数据：{count_pairs} 组")
    print(f"   缺失标签的图片数：{count_missing_data} 张")
    print(f"   文件保存在：{target_dir}")

if __name__ == '__main__':
    # ================= 配置区域 =================
    
    # 输入你的源文件夹路径 (可以使用绝对路径)
    source_folder = "/home/wangfeng/RM2026/amor_data/competation/5-24/3/images_0524_1608"
    
    # 输入你想保存到的文件夹路径
    target_folder = "/home/wangfeng/RM2026/amor_data/python_refactor/dataset/test_data"
    
    # 设置为 True 则是剪切（移动），False 是复制
    is_move = False 
    
    # ===========================================
    
    copy_dataset(source_folder, target_folder, is_move)