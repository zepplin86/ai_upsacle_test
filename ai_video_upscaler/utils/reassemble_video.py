import cv2
import os
import subprocess
from typing import List
from tqdm import tqdm
from .file_utils import ensure_directory, get_frame_files


def reassemble_video_from_frames(
    frames_dir: str,
    output_video_path: str,
    fps: float = 30.0,
    frame_format: str = "jpg",
    codec: str = "mp4v",
    quality: int = 95
) -> None:
    """
    프레임 이미지들을 비디오로 재합성합니다.
    
    Args:
        frames_dir (str): 프레임 이미지들이 저장된 디렉토리
        output_video_path (str): 출력 비디오 파일 경로
        fps (float): 출력 비디오의 프레임레이트
        frame_format (str): 프레임 이미지 형식 (jpg, png)
        codec (str): 비디오 코덱 (mp4v, avc1, h264)
        quality (int): 비디오 품질 (1-100)
    """
    # 출력 디렉토리 생성
    output_dir = os.path.dirname(output_video_path)
    ensure_directory(output_dir)
    
    # 프레임 파일 목록 가져오기
    frame_files = get_frame_files(frames_dir, frame_format)
    
    if not frame_files:
        raise ValueError(f"프레임 파일을 찾을 수 없습니다: {frames_dir}")
    
    print(f"총 {len(frame_files)}개 프레임을 비디오로 합성합니다.")
    
    # 첫 번째 프레임으로부터 이미지 크기 확인
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        raise ValueError(f"첫 번째 프레임을 읽을 수 없습니다: {frame_files[0]}")
    
    height, width = first_frame.shape[:2]
    print(f"프레임 크기: {width}x{height}")
    
    # 비디오 작성자 설정
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(
        output_video_path, 
        fourcc, 
        fps, 
        (width, height),
        isColor=True
    )
    
    if not out.isOpened():
        raise ValueError(f"비디오 작성자를 초기화할 수 없습니다: {output_video_path}")
    
    # 프레임들을 비디오에 추가
    with tqdm(total=len(frame_files), desc="비디오 합성 중") as pbar:
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            if frame is not None:
                out.write(frame)
            pbar.update(1)
    
    out.release()
    print(f"비디오 합성 완료: {output_video_path}")


def reassemble_video_with_ffmpeg(
    frames_dir: str,
    output_video_path: str,
    fps: float = 30.0,
    frame_format: str = "jpg",
    codec: str = "libx264",
    quality: int = 23,
    preset: str = "medium"
) -> None:
    """
    ffmpeg를 사용하여 프레임들을 비디오로 재합성합니다. (더 빠르고 효율적)
    
    Args:
        frames_dir (str): 프레임 이미지들이 저장된 디렉토리
        output_video_path (str): 출력 비디오 파일 경로
        fps (float): 출력 비디오의 프레임레이트
        frame_format (str): 프레임 이미지 형식 (jpg, png)
        codec (str): 비디오 코덱 (libx264, libx265, mpeg4)
        quality (int): 비디오 품질 (CRF 값, 낮을수록 고품질)
        preset (str): 인코딩 프리셋 (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
    """
    # 출력 디렉토리 생성
    output_dir = os.path.dirname(output_video_path)
    ensure_directory(output_dir)
    
    # 프레임 파일 패턴 수정
    frame_pattern = os.path.join(frames_dir, f"frame_%06d.{frame_format}")
    
    print(f"ffmpeg로 비디오 합성 중...")
    print(f"프레임 패턴: {frame_pattern}")
    print(f"출력: {output_video_path}")
    print(f"FPS: {fps}, 코덱: {codec}, 품질: {quality}")
    
    # ffmpeg 명령어 구성
    cmd = [
        "ffmpeg", "-y",  # 덮어쓰기 허용
        "-framerate", str(fps),  # 입력 프레임레이트
        "-i", frame_pattern,  # 입력 프레임 패턴
        "-c:v", codec,  # 비디오 코덱
        "-preset", preset,  # 인코딩 프리셋
    ]
    
    # 코덱별 품질 설정
    if codec in ["libx264", "libx265"]:
        cmd.extend(["-crf", str(quality)])  # CRF 품질 설정
    elif codec == "mpeg4":
        cmd.extend(["-q:v", str(quality)])  # MPEG4 품질 설정
    
    cmd.extend([
        "-pix_fmt", "yuv420p",  # 픽셀 포맷 (호환성을 위해)
        "-movflags", "+faststart",  # 웹 스트리밍 최적화
        output_video_path
    ])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("ffmpeg 비디오 합성 완료")
        
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg 실행 오류: {e}")
        print(f"stderr: {e.stderr}")
        raise


def create_video_with_audio(
    video_path: str,
    audio_path: str,
    output_path: str,
    audio_offset: float = 0.0
) -> None:
    """
    비디오에 오디오를 추가합니다.
    
    Args:
        video_path (str): 비디오 파일 경로
        audio_path (str): 오디오 파일 경로
        output_path (str): 출력 파일 경로
        audio_offset (float): 오디오 오프셋 (초)
    """
    print(f"비디오에 오디오 추가 중...")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
    ]
    
    if audio_offset != 0.0:
        cmd.extend(["-itsoffset", str(audio_offset)])
    
    cmd.extend([
        "-c:v", "copy",  # 비디오 스트림 복사
        "-c:a", "aac",   # 오디오 코덱
        "-shortest",     # 가장 짧은 스트림 길이에 맞춤
        output_path
    ])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("오디오 추가 완료")
        
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg 실행 오류: {e}")
        print(f"stderr: {e.stderr}")
        raise 