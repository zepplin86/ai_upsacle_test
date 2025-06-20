#!/usr/bin/env python3
"""
AI 비디오 업스케일링 파이프라인 테스트 스크립트
"""

import os
import sys
import subprocess
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_dependencies():
    """필요한 패키지들이 설치되어 있는지 확인합니다."""
    print("🔍 의존성 패키지 확인 중...")
    
    required_packages = [
        "torch",
        "torchvision", 
        "opencv-python",
        "numpy",
        "Pillow",
        "tqdm"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "opencv-python":
                import cv2
            elif package == "Pillow":
                import PIL
            else:
                __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (설치 필요)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 다음 패키지들을 설치하세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def test_ffmpeg():
    """ffmpeg가 설치되어 있는지 확인합니다."""
    print("\n🎬 ffmpeg 확인 중...")
    
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        print("✅ ffmpeg 설치됨")
        print(f"   버전: {result.stdout.split('ffmpeg version')[1].split()[0]}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ffmpeg가 설치되지 않았습니다.")
        print("   macOS: brew install ffmpeg")
        print("   Ubuntu: sudo apt install ffmpeg")
        print("   Windows: https://ffmpeg.org/download.html")
        return False


def test_cuda():
    """CUDA 사용 가능 여부를 확인합니다."""
    print("\n🚀 CUDA 확인 중...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ CUDA 사용 가능")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA 버전: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA 사용 불가 (CPU 모드로 실행)")
            return False
    except ImportError:
        print("❌ PyTorch가 설치되지 않았습니다.")
        return False


def test_sample_video():
    """샘플 비디오 파일이 있는지 확인합니다."""
    print("\n📹 샘플 비디오 확인 중...")
    
    input_dir = Path("input")
    if not input_dir.exists():
        input_dir.mkdir()
        print("📁 input 디렉토리 생성됨")
    
    video_files = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.mov"))
    
    if video_files:
        print(f"✅ {len(video_files)}개의 비디오 파일 발견:")
        for video in video_files:
            print(f"   - {video.name}")
        return True
    else:
        print("❌ input 디렉토리에 비디오 파일이 없습니다.")
        print("   테스트를 위해 MP4 또는 MOV 파일을 input/ 디렉토리에 넣어주세요.")
        return False


def run_quick_test():
    """빠른 테스트를 실행합니다."""
    print("\n🧪 빠른 테스트 실행 중...")
    
    # 샘플 비디오 파일 찾기
    input_dir = Path("input")
    video_files = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.mov"))
    
    if not video_files:
        print("❌ 테스트할 비디오 파일이 없습니다.")
        return False
    
    test_video = video_files[0]
    print(f"테스트 비디오: {test_video.name}")
    
    # 10프레임만 테스트
    cmd = [
        sys.executable, "main.py",
        "--input", str(test_video),
        "--output", "output/test_output.mp4",
        "--start-frame", "0",
        "--end-frame", "10",
        "--scale", "2",
        "--device", "cpu",
        "--keep-frames"
    ]
    
    print(f"실행 명령: {' '.join(cmd)}")
    print("\n⚠️  이 테스트는 시간이 오래 걸릴 수 있습니다.")
    print("   중단하려면 Ctrl+C를 누르세요.")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ 테스트 완료!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 테스트 실패: {e}")
        return False
    except KeyboardInterrupt:
        print("\n❌ 테스트가 중단되었습니다.")
        return False


def main():
    """메인 테스트 함수"""
    print("=" * 60)
    print("🧪 AI 비디오 업스케일링 파이프라인 테스트")
    print("=" * 60)
    
    # 1. 의존성 확인
    deps_ok = test_dependencies()
    
    # 2. ffmpeg 확인
    ffmpeg_ok = test_ffmpeg()
    
    # 3. CUDA 확인
    cuda_ok = test_cuda()
    
    # 4. 샘플 비디오 확인
    video_ok = test_sample_video()
    
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    print(f"의존성 패키지: {'✅' if deps_ok else '❌'}")
    print(f"ffmpeg: {'✅' if ffmpeg_ok else '❌'}")
    print(f"CUDA: {'✅' if cuda_ok else '⚠️'}")
    print(f"샘플 비디오: {'✅' if video_ok else '❌'}")
    
    if not deps_ok:
        print("\n❌ 의존성 패키지가 설치되지 않았습니다.")
        print("   다음 명령을 실행하세요:")
        print("   pip install -r requirements.txt")
        return
    
    if not video_ok:
        print("\n❌ 테스트할 비디오 파일이 없습니다.")
        print("   input/ 디렉토리에 MP4 또는 MOV 파일을 넣어주세요.")
        return
    
    # 5. 빠른 테스트 실행 여부 확인
    print("\n" + "=" * 60)
    response = input("빠른 테스트를 실행하시겠습니까? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        run_quick_test()
    else:
        print("테스트를 건너뜁니다.")
    
    print("\n" + "=" * 60)
    print("🎉 테스트 완료!")
    print("=" * 60)
    print("사용법:")
    print("  python main.py --input input/your_video.mp4")
    print("  python main.py --input input/your_video.mp4 --help")


if __name__ == "__main__":
    main() 