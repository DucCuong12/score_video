import os
import cv2
import json
import base64
import argparse
import csv
from dotenv import load_dotenv
import numpy as np
import torch
from torchvision.io import write_video
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from openai import OpenAI, AzureOpenAI


def create_llm_client(model_name, api_key=None):
    if model_name.lower() == "gpt":
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not azure_api_key or not azure_endpoint:
            raise ValueError(
                "Missing Azure OpenAI env vars: AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT"
            )
        return AzureOpenAI(
            api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            api_version="2024-02-15-preview",
        ), "gpt-4.1"
    if model_name.lower() == "deepseek":
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not azure_api_key or not azure_endpoint:
            raise ValueError(
                "Missing Azure OpenAI env vars: AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT"
            )
        return AzureOpenAI(
            api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            api_version="2024-02-15-preview",
        ), "DeepSeek-V3.2"
    if model_name.lower() == "qwen":
        qwen_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        if not qwen_key:
            raise ValueError(
                "Missing Qwen API key. Set --api_key or DASHSCOPE_API_KEY/QWEN_API_KEY in env."
            )
        return OpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=qwen_key,
        ), "qwen3-vl-235b-a22b-instruct"

    raise ValueError("Unsupported --model, choose from: gpt, qwen")


class Video_preprocess():
    def __init__(self):
        pass
   
    def extract_frames(self, video_path, num_frames=16):
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= num_frames:
            frame_indices = np.arange(total_frames)
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        current_index = 0
        target_set = set(frame_indices)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if current_index in target_set:
                frames.append(frame)
                if len(frames) >= num_frames:
                    break
            current_index += 1

        cap.release()
        return frames

    def rgb_to_yuv(self, frame):
        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return yuv_frame

    def frames_to_video(self, frames, output_path, fps=8):
        yuv_frames = [self.rgb_to_yuv(frame) for frame in frames]
        video_tensor = torch.from_numpy(np.array(yuv_frames)).to(torch.uint8)
        write_video(output_path, video_tensor, fps, video_codec='h264', options={'crf': '18'})

    def convert_video(self, input_path, output_path, num_frames):
        frames = self.extract_frames(input_path,num_frames=num_frames)
        self.frames_to_video(frames, output_path)
    
    def merge_grid(self, image_list, rows=3, cols=2):
        """
        将 image_list 中的帧拼接成 rows × cols 的网格图
        假设 len(image_list) == rows * cols
        """
        assert len(image_list) == rows * cols, f"需要 {rows*cols} 张图片，但传入 {len(image_list)} 张"

        # 每行拼接
        row_images = []
        for r in range(rows):
            row = np.concatenate(image_list[r*cols:(r+1)*cols], axis=1)
            row_images.append(row)

        # 再把行拼接起来
        grid = np.concatenate(row_images, axis=0)
        return grid
    def read_video_path(self, video_path):
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
        if os.path.isdir(video_path):  # if video_path is a list of videos
            video = [
                v for v in os.listdir(video_path)
                if os.path.isfile(os.path.join(video_path, v))
                and os.path.splitext(v.lower())[1] in video_exts
            ]
        elif os.path.isfile(video_path):  # else if video_path is a single video
            video = [os.path.basename(video_path)]
            video_path = os.path.dirname(video_path)
        else:
            raise FileNotFoundError(f"Invalid --video_path: {video_path}")
        video.sort()
        if not video:
            raise ValueError(
                f"No supported video files found in: {video_path}. "
                "Expected extensions: .mp4, .avi, .mov, .mkv, .webm, .m4v"
            )
        return video, video_path
         

    def convert_video_to_grid(self, video_path,num_image=6):
        base_output_dir = video_path if os.path.isdir(video_path) else os.path.dirname(video_path)
        video, video_path = self.read_video_path(video_path)
        print(f"start converting video to image grid with {num_image} frames from path:", video_path)
    
        output_path = os.path.join(base_output_dir, f"image_grid_{num_image}frame")
        os.makedirs(output_path, exist_ok=True)
    
        for v in video:
            vid_id = v.split(".")[0]
            vid_path = os.path.join(video_path,v)
            frames = self.extract_frames(vid_path)
            if len(frames) == 0:
                print(f"[WARN] Skip '{v}' because no frames could be extracted.")
                continue
            frame_indices = np.linspace(0, len(frames) - 1, num_image, dtype=int) #take 6 from 16 evenly, 1st & last included
            grid = [frames[i] for i in frame_indices]
            grid_image = self.merge_grid(grid)
            grid_filename = os.path.join(output_path, f'{vid_id}.jpg')
            cv2.imwrite(grid_filename, grid_image)
        print("finish converting from path: ", video_path)
        print("image grid stored in: ", output_path)
        return output_path

def create_prompt(view: str, description: str, robotic_manipulator: str, manipulated_object: str) -> str:
    """Construct an English VQA evaluation prompt (including robotic manipulator and manipulated object), requiring strict JSON output."""
    return f"""
You are shown an image composed of a 3×2 grid of frames arranged in chronological order (read row by row).  
These frames are extracted from an AI-generated video recorded from the {view} perspective.  
Video content: {description}  
Robotic manipulator: {robotic_manipulator}  
Manipulated object: {manipulated_object}  

Your evaluation goal is: **robot manipulation performance**.

Hard identity rule (must enforce):
- The manipulator must visually match the declared robotic type: {robotic_manipulator}.
- If the manipulator appears human-like (human hand/skin/fingers/anatomy) or non-robotic while the prompt expects a robot,
  assign **action_execution = 1** and **manipulator_consistency = 1**.
- In this case, treat manipulation quality as failed even if task outcome looks completed.

Please evaluate the video from the following five aspects.  
Each aspect receives a score from **1 to 5**:
- If the aspect is judged as "No", assign **1 point**.  
- If "Yes", assign **2–5 points** depending on quality (5 = perfect).  
- If any aspect in Category A (Action and Task Performance) equals 1, the total score = 1.
- Otherwise, compute the mean of all five scores as the final score.
- BE STRICT WHEN SCORING — if any issue or imperfection is detected, assign 1 or 2 points decisively.

---

### Category A — Action and Task Performance
These aspects focus on how well the robot performs the manipulation behavior and achieves the intended goal.
1) **Action Effectiveness**  
   - Check whether the robot’s motion is physically reasonable (e.g., proper gripper closure, contact location, trajectory smoothness).  
    - If robot identity mismatch is detected, this aspect must be 1.
   - Reference scoring:  
     1 = Motion discontinuous, incomplete, or physically implausible.  
     2 = Basically correct and understandable motion.  
     3 = Generally reasonable with slight inaccuracy.  
     4 = Smooth motion and natural contact.  
     5 = Fully consistent with physical and logical principles.

2) **Task Completion**  
   - Check whether the robot successfully accomplishes the described task and whether key intermediate steps are reasonable.  
   - Reference scoring:  
     1 = Task failed or goal not achieved.  
     2 = Partially achieved with major deviation.  
     3 = Main steps completed.  
     4 = Nearly complete with minor error.  
     5 = Fully completed; process and outcome perfectly match the description.

---

### Category B — Visual and Physical Consistency
These aspects evaluate whether the visual and physical properties of the robot and objects remain stable and realistic throughout the video.
3) **Manipulated Object Consistency**  
   - manipulated object: {manipulated_object}
   - Check whether the manipulated object maintains a consistent shape, structure, and outline over time.
     - Color/material or lighting changes alone are acceptable if object identity, geometry, and topology remain stable.
   - Note: Evaluate this aspect by comparing all frames to the first frame.
   - Reference scoring:  
         1 = Severe geometry/topology inconsistency (e.g., clear shape collapse, fragmentation, identity switch).
         2 = Clear geometric deformation affecting structure continuity.
         3 = Geometry mostly stable; only moderate local deformation or temporal jitter.
         4 = Geometry and identity stable; only minor artifacts. Color/material shift alone should still be >=4 when structure is stable.
     5 = Completely stable and perfectly consistent appearance.

4) **Robotic Manipulator Consistency**  
   - Robotic Manipulator: {robotic_manipulator} 
    - Check whether the robotic entity, arm or gripper maintains stable geometry and articulation without disconnection or self-intersection.
    - Also verify identity fidelity to the declared robotic type; human-like anatomy is a severe mismatch and should be scored as 1.
   - Note: Evaluate this aspect by comparing all frames to the first frame.
   - Reference scoring:  
     1 = Changes in the form, structure, or appearance of the manipulator or its subcomponents; consistency not maintained.  
     2 = Moderate differences but still consistent.  
     3 = Minor deformation; overall consistency maintained.  
     4 = Very small jitter or local artifacts only.  
     5 = Completely stable and perfectly consistent appearance.

5) **Anomaly Check**  
   - Examine whether any of the following issues are avoided:  
        a) Floating or penetration (violation of physical grounding).  
        b) Objects or robot parts suddenly appear or disappear between frames.  
        c) Non-contact attachment or false grasp (object sticks to gripper without visible closure).  
    - Do NOT treat color/material change alone as an anomaly unless accompanied by geometric breakage or physical violations.
   - Reference scoring:  
     1 = Occurrence of any of the above anomalies.  
     2 = No obvious anomalies but with noticeable artifacts or noise.  
     3 = Slight noise affecting visual quality.  
     4 = Only tiny visual imperfections.  
     5 = No above anomalies.

---

**[Final Scoring Rule]**  
If any aspect in Category A (Action and Task Performance) equals 1, the total score = 1.
Otherwise, compute the mean of all five scores as the final score.

**[Output Format (Strict JSON, No Other Text)]**  
Each aspect and the total must include the following fields:  
- "reason": a brief justification for the score  
- "score": score

Example JSON structure:
{{
  "action_execution": {{"reason": "...", "score": 2}},
  "task_completion": {{"reason": "...", "score": 2}},
  "object_consistency": {{"reason": "...", "score": 3}},
  "manipulator_consistency": {{"reason": "...", "score": 3}},
  "anomaly_check": {{"reason": "...", "score": 2}},
  "total": {{"reason": "...", "score": 2.4}}
}}

"""

def extract_json(string):
    start = string.find('{')
    end = string.rfind('}') + 1
    json_part = string[start:end]
    return json.loads(json_part)

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def compute_final_score(resp: dict) -> float:

    if not isinstance(resp, dict):
        raise TypeError("response 必须是 dict 类型")

    score1 = float(resp["action_execution"]["score"])
    score2 = float(resp["task_completion"]["score"])

    # Follow prompt rule exactly:
    # If any Category A score is 1, final score = 1; otherwise mean of all five aspects.
    if score1 == 1 or score2 == 1:
        return 1.0

    score3 = float(resp["object_consistency"]["score"])
    score4 = float(resp["manipulator_consistency"]["score"])
    score5 = float(resp["anomaly_check"]["score"])

    final_score = (score1 + score2 + score3 + score4 + score5) / 5.0

    return final_score
   
def save_results_to_csv(results, output_csv):
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['name', 'score', 'prompt', 'details']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            resp = result.get('response', {})

            # 若 response 是字符串则尝试解析为 JSON
            if isinstance(resp, str):
                try:
                    resp = json.loads(resp)
                except Exception:
                    resp = {}
            try:
                score = compute_final_score(resp)
            except Exception:
                score = -1
                    
            writer.writerow({
                'name': result.get('name', ''),
                'score': score,
                'prompt': result.get('prompt', ''),
                'details': resp
            })


def load_prompts(prompt_file_path):
    """Load prompt JSON and normalize it to a list of dicts."""
    with open(prompt_file_path, 'r') as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        prompts = [raw]
    elif isinstance(raw, list):
        prompts = raw
    else:
        raise ValueError("Prompt file must be a JSON object or a JSON array of objects")

    if not prompts:
        raise ValueError("Prompt file is empty")
    return prompts


def resolve_prompt_info(prompts, grid_image_name):
    """Resolve prompt info for an image by index/name, with single-prompt fallback."""
    if len(prompts) == 1:
        return prompts[0]

    image_stem = os.path.splitext(grid_image_name)[0]

    # Backward-compatible behavior: filenames like 0001.jpg -> prompt index 0.
    if image_stem.isdigit():
        image_index = int(image_stem) - 1
        if 0 <= image_index < len(prompts):
            return prompts[image_index]

    # Name-based fallback if prompt objects include a "name" field.
    for p in prompts:
        if str(p.get("name", "")) == image_stem:
            return p

    raise IndexError(
        f"Cannot map image '{grid_image_name}' to prompt. "
        f"Provide one shared prompt, numeric-indexed prompts, or prompts with matching 'name'."
    )


def get_prompt_fields(prompt_info):
    """Extract required prompt fields with sensible defaults."""
    view = prompt_info.get("view", "unspecified")
    description = prompt_info.get("prompt", "")
    robotic_manipulator = prompt_info.get("robotic manipulator", "robotic manipulator")
    manipulated_object = prompt_info.get("manipulated object", "manipulated object")
    sample_name = prompt_info.get("name", "shared_prompt")
    return view, description, robotic_manipulator, manipulated_object, sample_name


def list_grid_images(image_grid_path):
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        [f for f in os.listdir(image_grid_path)
         if os.path.isfile(os.path.join(image_grid_path, f))
         and os.path.splitext(f.lower())[1] in image_exts]
    )

def process_single_image_gpt(args_tuple):
    grid_image_name, prompts, image_grid_path, api_key = args_tuple
    client, real_model_name = create_llm_client("gpt", api_key)
    try:
        prompt_info = resolve_prompt_info(prompts, grid_image_name)
        view, description, robotic_manipulator, manipulated_object, sample_name = get_prompt_fields(prompt_info)
        image_path = os.path.join(image_grid_path, grid_image_name)
        img_base64 = encode_image(image_path)

        Q = create_prompt(view, description, robotic_manipulator, manipulated_object)

        response = client.chat.completions.create(
            model=real_model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": Q},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img_base64}}
                ]
            }],
            max_completion_tokens=8000,
            seed=2026,
        ).choices[0].message.content.strip()

        try:
            cleaned_output = extract_json(response)
        except Exception:
            cleaned_output = {"score": -1, "reason": response}
            print("Wrong response format!")
            
        return {
            'name': os.path.splitext(grid_image_name)[0],
            'prompt': description,
            'response': cleaned_output
        }
    except Exception as e:
        return {'name': grid_image_name, 'prompt': 'N/A',
                'response': {'score': -1, 'reason': f'Error: {e}'}}

def run_gpt():
    prompts = load_prompts(args.read_prompt_file)

    image_grid_path = args.image_grid_path
    if image_grid_path is None or not os.path.exists(image_grid_path) or not os.listdir(image_grid_path):
        video_preprocess = Video_preprocess()
        image_grid_path = video_preprocess.convert_video_to_grid(args.video_path)
    grid_images = list_grid_images(image_grid_path)

    task_args = [(img, prompts, image_grid_path, args.api_key) for img in grid_images]
    with Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap(process_single_image_gpt, task_args), total=len(task_args)))

    os.makedirs(args.output_path, exist_ok=True)
    output_csv = os.path.join(args.output_path, 'results.csv')
    save_results_to_csv(results, output_csv)

def process_single_image_qwen(args_tuple):
    grid_image_name, prompts, image_grid_path, api_key = args_tuple
    client, real_model_name = create_llm_client("qwen", api_key)
    try:
        prompt_info = resolve_prompt_info(prompts, grid_image_name)
        view, description, robotic_manipulator, manipulated_object, sample_name = get_prompt_fields(prompt_info)
        image_path = os.path.join(image_grid_path, grid_image_name)
        img_base64 = encode_image(image_path)

        Q = create_prompt(view, description, robotic_manipulator, manipulated_object)

        response = client.chat.completions.create(
            model=real_model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": Q},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img_base64}}
                ]
            }],
            max_completion_tokens=800,
            seed=2026,
        ).choices[0].message.content.strip()

        try:
            cleaned_output = extract_json(response)
        except Exception:
            cleaned_output = {"score": -1, "reason": response}
            print("Wrong response format!")
            
        return {
            'name': os.path.splitext(grid_image_name)[0],
            'prompt': description,
            'response': cleaned_output
        }
    except Exception as e:
        return {'name': grid_image_name, 'prompt': 'N/A',
                'response': {'score': -1, 'reason': f'Error: {e}'}}

def run_qwen():
    prompts = load_prompts(args.read_prompt_file)

    image_grid_path = args.image_grid_path
    if image_grid_path is None or not os.path.exists(image_grid_path) or not os.listdir(image_grid_path):
        video_preprocess = Video_preprocess()
        image_grid_path = video_preprocess.convert_video_to_grid(args.video_path)
    grid_images = list_grid_images(image_grid_path)

    task_args = [(img, prompts, image_grid_path, args.api_key) for img in grid_images]
    with Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap(process_single_image_qwen, task_args), total=len(task_args)))

    os.makedirs(args.output_path, exist_ok=True)
    output_csv = os.path.join(args.output_path, 'results.csv')
    save_results_to_csv(results, output_csv)

# =================== 启动入口 ===================
if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, type=str)
    parser.add_argument("--image_grid_path", type=str)
    parser.add_argument("--read_prompt_file", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--num_workers", type=int, default=min(8, cpu_count()))
    parser.add_argument("--model", type=str, default="deepseek", choices=["deepseek", "gpt", "qwen"])

    args = parser.parse_args()

    if args.model == "deepseek":
        run_gpt()
    elif args.model == "gpt":
        run_gpt()
    else:
        run_qwen()
