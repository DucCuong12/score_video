import os
import cv2
import json
import base64
import argparse
import csv
from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from openai import OpenAI, AzureOpenAI


def create_llm_client(model_name, api_key=None):
    if model_name.lower() == "gpt":
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if azure_api_key and azure_endpoint:
            return AzureOpenAI(
                api_key=azure_api_key,
                azure_endpoint=azure_endpoint,
                api_version="2024-02-15-preview",
            ), "DeepSeek-V3.2"

        openai_key = api_key or os.getenv("OPENAI_API_KEY")
        if openai_key:
            return OpenAI(api_key=openai_key), "gpt-5-2025-08-07"

        raise ValueError(
            "Missing GPT credentials. Set AZURE_OPENAI_API_KEY/AZURE_OPENAI_ENDPOINT or OPENAI_API_KEY/--api_key."
        )

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

def create_prompt(view: str, description: str, manipulated_object: str, entity1: str, entity2: str) -> str:
    """
    Construct an English VQA evaluation prompt (multi-entity version).
    Support for robot-human or robot-object collaboration.
    Require strict JSON output.
    """

    entity2_l = str(entity2).lower()
    passive_keywords = [
        "box", "bin", "container", "basket", "tray", "table", "shelf", "rack", "target"
    ]
    is_passive_target = any(k in entity2_l for k in passive_keywords)

    if is_passive_target:
        category_a_note = (
            f"- IMPORTANT: {entity2} is treated as a passive target object. "
            "Do NOT require it to move or respond actively.\n"
        )
        action_effectiveness_rule = (
            f"- Check whether {entity1} performs a complete and physically plausible manipulation of {manipulated_object} "
            f"toward/into {entity2}. Focus on trajectory, grasp quality, and placement alignment. "
            f"Do not penalize simply because {entity2} remains static."
        )
        task_completion_rule = (
            f"- Check whether the task goal is achieved: {entity1} correctly manipulates {manipulated_object} "
            f"and places/transfers it relative to {entity2}. {entity2} may be passive."
        )
    else:
        category_a_note = ""
        action_effectiveness_rule = (
            f"- Check strictly whether every interaction action between {entity1} and {entity2} is successful and complete."
        )
        task_completion_rule = (
            f"- Check whether both {entity1} and {entity2} actively participate in the required sub-steps and perform the correct roles."
        )

    return f"""
You are shown an image composed of a 3×2 grid of frames arranged in chronological order (read row by row).  
These frames are extracted from an AI-generated video recorded from the {view} perspective.  

Video content: {description}  
Primary entity (Entity1): {entity1}  
Secondary entity (Entity2): {entity2}  
Manipulated object: {manipulated_object}  

Your evaluation goal is: **multi-entity manipulation and interaction performance**.

Hard identity rule (must enforce):
- If Entity1 is specified as a robot (e.g., robotic hand/arm/manipulator), its visual form must be robotic.
- If Entity1 appears human-like (human hand/skin/fingers/anatomy) or non-robotic while expected to be robot,
  assign **action_coordination = 1** and **entity_consistency = 1**.
- In this case, cooperative quality should be treated as failed even if final placement looks correct.

Please evaluate the video from the following five aspects.  
Each aspect receives a score from **1 to 5**:
- If the aspect is judged as "No", assign **1 point**.  
- If "Yes", assign **2–5 points** depending on quality (5 = perfect).  
- If any aspect in Category A (Task Performance and Interaction Action) equals 1, the total score = 1.
- Otherwise, compute the mean of all five scores as the final score.
- BE STRICT WHEN SCORING — if any issue or imperfection is detected, assign 1 or 2 points decisively.
{category_a_note}

---

### Category A — Task Performance and Interaction Action
These aspects focus on how well the **two entities cooperate and perform** the described behavior.

1) **Action Effectiveness**  
    {action_effectiveness_rule}  
    - If Entity1 robot identity mismatch is detected, this aspect must be 1.
   - Each required interaction must exhibit a complete and well-coordinated behavioral sequence consistent with the task type, demonstrating proper timing and causal response:
    - Contact interactions: approach → contact → release/transfer.
    - Non-contact interactions: initiation → alignment → sustained coordination.
    - Missing any phase, showing asynchronous responses, or violating physical logic means the interaction is not completed.
   - Reference scoring:  
     1 = Actions fail or interaction incomplete.  
     2 = Basically correct with small timing or contact issues.  
     3 = Actions generally correct and clear.  
     4 = Smooth and natural interactive motion.  
     5 = Fully successful and physically realistic cooperative motion.

2) **Task Completion**  
   - Check whether the task goal described in the prompt is fully achieved.
    {task_completion_rule}
   - BE STRICT WHEN JUDGING WHETHER THE GOAL IS COMPLETED
   - Reference scoring:  
     1 = Task failed or goal not achieved.  
     2 = Partially achieved with major deviation.  
     3 = Main steps completed.  
     4 = Nearly complete with minor error.  
     5 = Fully completed; process and outcome perfectly match the description.

---

### Category B — Visual and Physical Consistency

3) **Entity Consistency (Robot & Interaction Partner Stability)**  
   - Check whether the {entity1} and {entity2} maintains consistent geometry, articulation, and identity.
    - Also verify that {entity1} matches its declared identity (robot vs human-like appearance).
    - If {entity1} is expected to be robotic but appears human-like, this aspect should be 1.
   - Note: Evaluate this aspect by comparing all frames to the first frame.
   - Reference scoring:  
     1 = Clear deformation or identity loss or discontinuity.  
     2 = Moderate differences but still consistent.  
     3 = Minor deformation; overall consistency maintained.  
     4 = Very small jitter or local artifacts only.  
     5 = Completely stable and perfectly consistent appearance.

5) **Manipulated Object Consistency**  
   - manipulated object: {manipulated_object}
   - Check whether the manipulated object maintains a consistent shape, structure, and outline over time.
   - Note: Evaluate this aspect by comparing all frames to the first frame.
   - Reference scoring:  
     1 = Noticeable changes in appearance such as color, shape, or material; consistency not maintained.
     2 = Moderate differences but still consistent.  
     3 = Minor deformation; overall consistency maintained.  
     4 = Very small jitter or local artifacts only.  
     5 = Completely stable and perfectly consistent appearance.

6) **Physical Anomaly Check**  
   - Examine whether any of the following are avoided:  
        a) Interpenetration between {entity1}, {entity2}, or {manipulated_object}.  
        b) Floating — occurs when any object or robotic part unnaturally hovers without physical support or contact.
        c) Sudden object disappearance, duplication, or sticking artifacts.
        d) Non-contact attachment or false grasp (object sticks to gripper without visible closure).

    - Reference scoring:  
     1 = Occurrence of any of the above anomalies.  
     2 = No obvious anomalies but with noticeable artifacts or noise.  
     3 = Slight noise affecting visual quality.  
     4 = Only tiny visual imperfections.  
     5 = No above anomalies.

---

**[Final Scoring Rule]**  
If any aspect in Category A (Task Performance and Interaction Action) equals 1, the total score = 1.  
Otherwise, compute the mean of all five scores as the final score.

---

**[Output Format (Strict JSON, No Other Text)]**  
Each aspect and the total must include:  
- "reason": a concise explanation  
- "score": score

Example JSON structure:
{{
  "action_coordination": {{"reason": "...", "score": 3}},
  "task_completion": {{"reason": "...", "score": 3}},
  "entity_consistency": {{"reason": "...", "score": 3}},
  "object_consistency": {{"reason": "...", "score": 3}},
  "anomaly_check": {{"reason": "...", "score": 3}},
  "total": {{"reason": "...", "score": 3.0}}
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

    score1 = resp["action_coordination"]["score"]
    score2 = resp["task_completion"]["score"]
    total_score = resp["total"]["score"]

    a_mean = (score1 + score2) / 2

    final_score = max(min(a_mean, total_score), 1)

    return final_score

def save_results_to_csv(results, output_csv):
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['name', 'score', 'prompt', 'details']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            resp = result.get('response', {})

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
    if len(prompts) == 1:
        return prompts[0]

    image_stem = os.path.splitext(grid_image_name)[0]
    if image_stem.isdigit():
        image_index = int(image_stem) - 1
        if 0 <= image_index < len(prompts):
            return prompts[image_index]

    for p in prompts:
        if str(p.get("name", "")) == image_stem:
            return p

    raise IndexError(
        f"Cannot map image '{grid_image_name}' to prompt. "
        "Provide one shared prompt, numeric-indexed prompts, or prompts with matching 'name'."
    )


def get_prompt_fields(prompt_info):
    view = prompt_info.get("view", "unspecified")
    description = prompt_info.get("prompt", "")
    manipulated_object = prompt_info.get("manipulated object", "manipulated object")
    entity1 = prompt_info.get("entity1", prompt_info.get("robotic manipulator", "entity1"))
    entity2 = prompt_info.get("entity2", "entity2")
    return view, description, manipulated_object, entity1, entity2


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
        view, description, manipulated_object, entity1, entity2 = get_prompt_fields(prompt_info)
        image_path = os.path.join(image_grid_path, grid_image_name)
        img_base64 = encode_image(image_path)

        Q = create_prompt(view, description, manipulated_object, entity1, entity2)

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
        view, description, manipulated_object, entity1, entity2 = get_prompt_fields(prompt_info)
        image_path = os.path.join(image_grid_path, grid_image_name)
        img_base64 = encode_image(image_path)

        Q = create_prompt(view, description, manipulated_object, entity1, entity2)

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
    parser.add_argument("--model", type=str, default="gpt", choices=["gpt", "qwen"])
    
    args = parser.parse_args()

    if args.model == "gpt":
        run_gpt()
    else:
        run_qwen()
