import cv2
import os
from tqdm import tqdm
from typing import Tuple
from .file_utils import ensure_directory, get_video_info


def extract_frames_from_video(
    video_path: str, 
    output_dir: str, 
    frame_format: str = "jpg",
    quality: int = 95,
    start_frame: int = 0,
    end_frame: int = None
) -> Tuple[int, float]:
    """
    비디오 파일에서 프레임을 추출하여 이미지 파일로 저장합니다.
    
    Args:
        video_path (str): 입력 비디오 파일 경로
        output_dir (str): 프레임을 저장할 디렉토리
        frame_format (str): 저장할 이미지 형식 (jpg, png)
        quality (int): JPEG 품질 (1-100, jpg 형식일 때만 적용)
        start_frame (int): 추출 시작 프레임 번호
        end_frame (int): 추출 종료 프레임 번호 (None이면 끝까지)
        
    Returns:
        Tuple[int, float]: (추출된 프레임 수, 비디오 FPS)
    """
    # 출력 디렉토리 생성
    ensure_directory(output_dir)
    
    # 비디오 정보 가져오기
    width, height, total_frames, fps = get_video_info(video_path)
    
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
    
    # 프레임 범위 설정
    if end_frame is None:
        end_frame = total_frames
    
    # 추출할 프레임 수 계산
    frames_to_extract = end_frame - start_frame
    
    print(f"비디오 정보: {width}x{height}, {total_frames}프레임, {fps:.2f}fps")
    print(f"추출 범위: {start_frame} ~ {end_frame} ({frames_to_extract}프레임)")
    
    # 시작 프레임으로 이동
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 프레임 추출
    extracted_count = 0
    current_frame = start_frame
    
    with tqdm(total=frames_to_extract, desc="프레임 추출 중") as pbar:
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 프레임 파일명 생성 (6자리 숫자로 패딩)
            frame_filename = f"frame_{current_frame:06d}.{frame_format}"
            frame_path = os.path.join(output_dir, frame_filename)
            
            # 이미지 저장
            if frame_format.lower() == "jpg":
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                cv2.imwrite(frame_path, frame, encode_params)
            else:
                cv2.imwrite(frame_path, frame)
            
            extracted_count += 1
            current_frame += 1
            pbar.update(1)
    
    cap.release()
    
    print(f"프레임 추출 완료: {extracted_count}개 프레임을 {output_dir}에 저장했습니다.")
    
    return extracted_count, fps


def extract_frames_with_ffmpeg(
    video_path: str, 
    output_dir: str, 
    frame_format: str = "jpg",
    quality: int = 95,
    start_frame: int = 0,
    end_frame: int = None
) -> Tuple[int, float]:
    """
    ffmpeg를 사용하여 비디오에서 프레임을 추출합니다. (더 빠른 방법)
    
    Args:
        video_path (str): 입력 비디오 파일 경로
        output_dir (str): 프레임을 저장할 디렉토리
        frame_format (str): 저장할 이미지 형식 (jpg, png)
        quality (int): JPEG 품질 (1-100, jpg 형식일 때만 적용)
        start_frame (int): 추출 시작 프레임 번호
        end_frame (int): 추출 종료 프레임 번호 (None이면 끝까지)
        
    Returns:
        Tuple[int, float]: (추출된 프레임 수, 비디오 FPS)
    """
    import subprocess
    
    # 출력 디렉토리 생성
    ensure_directory(output_dir)
    
    # 비디오 정보 가져오기
    width, height, total_frames, fps = get_video_info(video_path)
    
    # 프레임 범위 설정
    if end_frame is None:
        end_frame = total_frames
    
    frames_to_extract = end_frame - start_frame
    
    print(f"ffmpeg로 프레임 추출 중: {frames_to_extract}개 프레임")
    
    # ffmpeg 명령어 구성
    if frame_format.lower() == "jpg":
        output_pattern = os.path.join(output_dir, "frame_%06d.jpg")
        quality_param = f"-q:v {max(1, min(31, 31 - int(quality * 0.31)))}"  # ffmpeg quality scale
    else:
        output_pattern = os.path.join(output_dir, "frame_%06d.png")
        quality_param = ""
    
    # 시작 시간 계산 (초 단위)
    start_time = start_frame / fps if fps > 0 else 0
    
    # ffmpeg 명령어 실행
    cmd = [
        "ffmpeg", "-y",  # 덮어쓰기 허용
        "-ss", str(start_time),  # 시작 시간
        "-i", video_path,  # 입력 파일
        "-vf", f"select=between(n\\,{start_frame}\\,{end_frame-1})",  # 프레임 범위 선택
        "-vsync", "0",  # 동기화 비활성화
        "-frame_pts", "1",  # 프레임 타임스탬프 포함
    ]
    
    if quality_param:
        cmd.extend(quality_param.split())
    
    cmd.extend([
        "-f", "image2",  # 이미지 시퀀스 출력
        output_pattern
    ])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("ffmpeg 프레임 추출 완료")
        
        # 실제 추출된 프레임 수 확인
        import glob
        extracted_files = glob.glob(os.path.join(output_dir, f"*.{frame_format}"))
        extracted_count = len(extracted_files)
        
        return extracted_count, fps
        
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg 실행 오류: {e}")
        print(f"stderr: {e.stderr}")
        raise 