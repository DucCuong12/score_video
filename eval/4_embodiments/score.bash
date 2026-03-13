# 1. Robot Subject Stability
python3 1_robot_subject_stability.py --video_path video_task1/ --read_prompt_file prompt_obj.json --output_path A --api_key AA --num_workers 1 --model gpt

# 2. Physical Plausibility (đã làm rồi)
python3 2_physical_plausibility.py --video_path video_task1/ --read_prompt_file prompt.json --output_path A --api_key AA --num_workers 1 --model gpt
# 3. Task Adherence Consistency
python3 3_task_adherence_consistency.py --video_path video_task1/ --read_prompt_file prompt.json --output_path A --api_key AA --num_workers 1 --model gpt

# 4. Create Meta Info (tạo file thông tin chứa tất cả điểm)
python3 4_create_meta_info.py --video_path video_task1/ 

# 5. Motion Amplitude
python3 5_motion_amplitude.py --video_path episode_000008.mp4 --output_path output5

# 6. Motion Smoothness
python3 6_motion_smoothness.py --video_path episode_000008.mp4 --output_path output6