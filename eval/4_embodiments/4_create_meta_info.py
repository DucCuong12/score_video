import json
import os
import argparse
    
def create_meta_info_from_videos_and_prompt(video_folder, prompt_data):
    """
    Tạo meta info từ folder video + 1 prompt chung
    
    Args:
        video_folder: đường dẫn folder chứa video
        prompt_data: 1 object prompt (hoặc list với 1 phần tử)
    
    Returns:
        list của meta_info entries
    """
    # Nếu prompt_data là list, lấy phần tử đầu tiên
    if isinstance(prompt_data, list):
        prompt = prompt_data[0]
    else:
        prompt = prompt_data
    
    meta_infos = []
    video_files = sorted([f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))])
    
    for idx, video_file in enumerate(video_files):
        video_path = os.path.join(video_folder, video_file)
        
        meta_info = {
            "index": idx + 1,
            "video_name": os.path.splitext(video_file)[0],
            "prompt": prompt.get("prompt", ""),
            "robotic_manipulator": prompt.get("robotic manipulator", ""),
            "manipulated_object": prompt.get("manipulated object", ""),
            "filepath": os.path.abspath(video_path)
        }
        meta_infos.append(meta_info)
    
    return meta_infos

def main():
    parser = argparse.ArgumentParser(description="Create meta info JSON từ folder video + 1 prompt chung")
    parser.add_argument("-v", "--video_folder", required=True, help="Đường dẫn folder chứa video")
    parser.add_argument("-p", "--prompt_file", required=True, help="Đường dẫn file prompt JSON (chỉ cần 1 prompt)")
    parser.add_argument("-o", "--output_json", required=True, help="Đường dẫn lưu meta_info.json")
    
    args = parser.parse_args()

    video_folder = os.path.abspath(args.video_folder)

    # Đọc prompt file
    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        prompt_data = json.load(f)

    # Tạo meta info cho tất cả video
    meta_infos = create_meta_info_from_videos_and_prompt(video_folder, prompt_data)

    # Tạo thư mục output nếu cần
    output_dir = os.path.dirname(args.output_json)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Lưu meta info JSON
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(meta_infos, f, indent=2, ensure_ascii=False)

    print(f"✅ Tạo meta_info cho {len(meta_infos)} video, lưu vào: {args.output_json}")
    for meta in meta_infos:
        print(f"   - {meta['video_name']}: {meta['prompt']}")

if __name__ == "__main__":
    main()