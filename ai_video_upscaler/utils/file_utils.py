import os
import glob
from pathlib import Path
from typing import List, Tuple


def ensure_directory(directory_path: str) -> None:
    """
    디렉토리가 존재하지 않으면 생성합니다.
    
    Args:
        directory_path (str): 생성할 디렉토리 경로
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def get_video_info(video_path: str) -> Tuple[int, int, int, float]:
    """
    비디오 파일의 기본 정보를 추출합니다.
    
    Args:
        video_path (str): 비디오 파일 경로
        
    Returns:
        Tuple[int, int, int, float]: (width, height, total_frames, fps)
    """
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    cap.release()
    
    return width, height, total_frames, fps


def get_frame_files(frames_dir: str, extension: str = "jpg") -> List[str]:
    """
    프레임 디렉토리에서 모든 프레임 파일을 정렬된 순서로 가져옵니다.
    
    Args:
        frames_dir (str): 프레임 파일들이 저장된 디렉토리
        extension (str): 파일 확장자 (기본값: "jpg")
        
    Returns:
        List[str]: 정렬된 프레임 파일 경로 리스트
    """
    pattern = os.path.join(frames_dir, f"*.{extension}")
    frame_files = glob.glob(pattern)
    frame_files.sort()  # 파일명 순서대로 정렬
    return frame_files


def clean_directory(directory_path: str) -> None:
    """
    디렉토리 내의 모든 파일을 삭제합니다.
    
    Args:
        directory_path (str): 정리할 디렉토리 경로
    """
    if os.path.exists(directory_path):
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)


def format_time(seconds: float) -> str:
    """
    초 단위 시간을 HH:MM:SS 형식으로 변환합니다.
    
    Args:
        seconds (float): 초 단위 시간
        
    Returns:
        str: HH:MM:SS 형식의 시간 문자열
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" 