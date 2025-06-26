#!/usr/bin/env python3
"""
AI 비디오 업스케일링 파이프라인
저화질 영상을 프레임 단위로 분리하고 AI 모델로 복원한 후 다시 영상으로 합성합니다.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.file_utils import ensure_directory, get_video_info, format_time
from utils.extract_frames import extract_frames_with_ffmpeg, extract_frames_from_video
from utils.reassemble_video import reassemble_video_with_ffmpeg, reassemble_video_from_frames
from models.realesrgan_runner import RealESRGANRunner, SwinIRRunner


def parse_arguments():
    """명령행 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="AI 비디오 업스케일링 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py --input input/sample.mp4
  python main.py --input input/sample.mp4 --model realesrgan --scale 2
  python main.py --input input/sample.mp4 --model swinir --device cuda --fps 30
  
고품질 옵션:
  python main.py --input input/sample.mp4 --high-quality
  python main.py --input input/sample.mp4 --tile-size 200 --tile-pad 20
  python main.py --input input/sample.mp4 --model realesrgan --scale 2 --high-quality --use-ffmpeg

성능 최적화:
  python main.py --input input/sample.mp4 --max-speed
  python main.py --input input/sample.mp4 --memory-efficient --use-amp --num-workers 4
  python main.py --input input/sample.mp4 --device cuda --batch-size 2 --half-precision
        """
    )
    
    # 필수 인수
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="입력 비디오 파일 경로"
    )
    
    # 선택적 인수
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output/final_upscaled_video.mp4",
        help="출력 비디오 파일 경로 (기본값: output/final_upscaled_video.mp4)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["realesrgan", "swinir"],
        default="realesrgan",
        help="사용할 AI 모델 (기본값: realesrgan)"
    )
    
    parser.add_argument(
        "--scale", "-s",
        type=int,
        choices=[2, 3, 4],
        default=4,
        help="업스케일 배율 (기본값: 4)"
    )
    
    parser.add_argument(
        "--device", "-d",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="사용할 디바이스 (기본값: auto)"
    )
    
    parser.add_argument(
        "--fps", "-f",
        type=float,
        default=None,
        help="출력 비디오 FPS (기본값: 원본과 동일)"
    )
    
    parser.add_argument(
        "--frame-format",
        type=str,
        choices=["jpg", "png"],
        default="jpg",
        help="프레임 이미지 형식 (기본값: jpg)"
    )
    
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="이미지 품질 (1-100, 기본값: 95)"
    )
    
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="처리 시작 프레임 번호 (기본값: 0)"
    )
    
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="처리 종료 프레임 번호 (기본값: 끝까지)"
    )
    
    parser.add_argument(
        "--use-ffmpeg",
        action="store_true",
        help="ffmpeg를 사용하여 프레임 추출 및 비디오 합성 (더 빠름)"
    )
    
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="중간 프레임 파일들을 유지 (기본값: 삭제)"
    )
    
    parser.add_argument(
        "--codec",
        type=str,
        default="libx264",
        help="비디오 코덱 (기본값: libx264)"
    )
    
    parser.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
        help="인코딩 프리셋 (기본값: medium)"
    )
    
    # 고품질 옵션들
    parser.add_argument(
        "--tile-size",
        type=int,
        default=400,
        help="타일 크기 (작을수록 더 선명하지만 느림, 기본값: 400)"
    )
    
    parser.add_argument(
        "--tile-pad",
        type=int,
        default=10,
        help="타일 패딩 크기 (클수록 더 선명하지만 느림, 기본값: 10)"
    )
    
    parser.add_argument(
        "--half-precision",
        action="store_true",
        help="반정밀도 사용 (GPU에서만, 더 빠르지만 품질 저하 가능)"
    )
    
    parser.add_argument(
        "--pre-pad",
        type=int,
        default=0,
        help="사전 패딩 크기 (기본값: 0)"
    )
    
    parser.add_argument(
        "--high-quality",
        action="store_true",
        help="고품질 모드 (tile-size=200, tile-pad=20, pre-pad=10)"
    )
    
    parser.add_argument(
        "--text-sharpen",
        type=str,
        choices=["none", "opencv", "pillow"],
        default="none",
        help="텍스트 샤프닝 후처리 방식 (none, opencv, pillow)"
    )
    
    parser.add_argument(
        "--sharpen-strength",
        type=float,
        default=0.5,
        help="선명화 강도 (0-1, 기본값: 0.5)"
    )
    
    # 성능 최적화 옵션들
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="배치 크기 (GPU 메모리에 따라 조정, 기본값: 1)"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="병렬 처리 워커 수 (0=비활성화, 기본값: 0)"
    )
    
    parser.add_argument(
        "--memory-efficient",
        action="store_true",
        help="메모리 효율 모드 (타일 크기 자동 조정)"
    )
    
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Automatic Mixed Precision 사용 (GPU에서만, 더 빠름)"
    )
    
    parser.add_argument(
        "--max-speed",
        action="store_true",
        help="최대 속도 모드 (모든 최적화 옵션 활성화)"
    )
    
    return parser.parse_args()


def main():
    """메인 실행 함수"""
    args = parse_arguments()
    
    # 시작 시간 기록
    start_time = time.time()
    
    print("=" * 60)
    print("🎬 AI 비디오 업스케일링 파이프라인")
    print("=" * 60)
    
    # 입력 파일 확인
    if not os.path.exists(args.input):
        print(f"❌ 입력 파일을 찾을 수 없습니다: {args.input}")
        sys.exit(1)
    
    # 비디오 정보 가져오기
    try:
        width, height, total_frames, original_fps = get_video_info(args.input)
        print(f"📹 비디오 정보:")
        print(f"   해상도: {width}x{height}")
        print(f"   총 프레임: {total_frames}")
        print(f"   FPS: {original_fps:.2f}")
        
        # FPS 설정
        output_fps = args.fps if args.fps is not None else original_fps
        
    except Exception as e:
        print(f"❌ 비디오 정보를 가져올 수 없습니다: {e}")
        sys.exit(1)
    
    # 출력 디렉토리 생성
    output_dir = os.path.dirname(args.output)
    ensure_directory(output_dir)
    
    # 임시 디렉토리 설정
    temp_dir = os.path.join(output_dir, "temp")
    ensure_directory(temp_dir)
    
    extracted_frames_dir = os.path.join(temp_dir, "extracted_frames")
    upscaled_frames_dir = os.path.join(temp_dir, "upscaled_frames")
    
    try:
        # 1단계: 프레임 추출
        print("\n🔍 1단계: 비디오에서 프레임 추출")
        print("-" * 40)
        
        if args.use_ffmpeg:
            extracted_count, fps = extract_frames_with_ffmpeg(
                video_path=args.input,
                output_dir=extracted_frames_dir,
                frame_format=args.frame_format,
                quality=args.quality,
                start_frame=args.start_frame,
                end_frame=args.end_frame
            )
        else:
            extracted_count, fps = extract_frames_from_video(
                video_path=args.input,
                output_dir=extracted_frames_dir,
                frame_format=args.frame_format,
                quality=args.quality,
                start_frame=args.start_frame,
                end_frame=args.end_frame
            )
        
        # 2단계: AI 모델 초기화
        print("\n🤖 2단계: AI 모델 초기화")
        print("-" * 40)
        
        # 고품질 모드 설정
        if args.high_quality:
            tile_size = 200
            tile_pad = 20
            pre_pad = 10
            print("🎯 고품질 모드 활성화")
        else:
            tile_size = args.tile_size
            tile_pad = args.tile_pad
            pre_pad = args.pre_pad
        
        # 최대 속도 모드 설정
        if args.max_speed:
            print("⚡ 최대 속도 모드 활성화")
            # 모든 최적화 옵션 활성화
            args.memory_efficient = True
            args.use_amp = True
            args.half_precision = True
            args.num_workers = min(4, os.cpu_count() or 1)  # CPU 코어 수에 따라 조정
            args.batch_size = 2  # GPU 메모리 허용 시 배치 크기 증가
            tile_size = 600  # 더 큰 타일로 처리 속도 향상
            tile_pad = 5  # 패딩 최소화
            pre_pad = 0  # 사전 패딩 비활성화
            print(f"  - 메모리 효율 모드: 활성화")
            print(f"  - AMP: 활성화")
            print(f"  - 반정밀도: 활성화")
            print(f"  - 병렬 워커: {args.num_workers}")
            print(f"  - 배치 크기: {args.batch_size}")
            print(f"  - 타일 크기: {tile_size}")
        
        if args.model == "realesrgan":
            upscaler = RealESRGANRunner(
                model_name="RealESRGAN_x4plus",
                device=args.device,
                scale=args.scale,
                tile_size=tile_size,
                tile_pad=tile_pad,
                half_precision=args.half_precision,
                pre_pad=pre_pad,
                text_sharpen=args.text_sharpen,
                sharpen_strength=args.sharpen_strength,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                memory_efficient=args.memory_efficient,
                use_amp=args.use_amp
            )
        elif args.model == "swinir":
            upscaler = SwinIRRunner(
                scale=args.scale,
                device=args.device
            )
        
        model_info = upscaler.get_model_info()
        print(f"모델: {args.model.upper()}")
        print(f"배율: {args.scale}x")
        print(f"디바이스: {model_info['device']}")
        if args.model == "realesrgan":
            print(f"타일 크기: {tile_size}")
            print(f"타일 패딩: {tile_pad}")
            print(f"사전 패딩: {pre_pad}")
            if args.half_precision:
                print("반정밀도: 활성화")
            if args.batch_size > 1:
                print(f"배치 크기: {args.batch_size}")
            if args.num_workers > 0:
                print(f"병렬 워커: {args.num_workers}")
            if args.memory_efficient:
                print("메모리 효율 모드: 활성화")
            if args.use_amp:
                print("AMP: 활성화")
        
        # 3단계: 프레임 업스케일링
        print("\n✨ 3단계: 프레임 업스케일링")
        print("-" * 40)
        
        processed_count = upscaler.upscale_batch(
            input_dir=extracted_frames_dir,
            output_dir=upscaled_frames_dir,
            frame_format=args.frame_format
        )
        
        # 4단계: 비디오 재합성
        print("\n🎬 4단계: 비디오 재합성")
        print("-" * 40)
        
        if args.use_ffmpeg:
            reassemble_video_with_ffmpeg(
                frames_dir=upscaled_frames_dir,
                output_video_path=args.output,
                fps=output_fps,
                frame_format=args.frame_format,
                codec=args.codec,
                quality=23,  # CRF 값
                preset=args.preset
            )
        else:
            reassemble_video_from_frames(
                frames_dir=upscaled_frames_dir,
                output_video_path=args.output,
                fps=output_fps,
                frame_format=args.frame_format,
                codec="mp4v",
                quality=args.quality
            )
        
        # 5단계: 정리
        print("\n🧹 5단계: 정리")
        print("-" * 40)
        
        if not args.keep_frames:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"임시 파일 삭제 완료: {temp_dir}")
        else:
            print(f"임시 파일 유지: {temp_dir}")
        
        # 완료 시간 계산
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("✅ 업스케일링 완료!")
        print("=" * 60)
        print(f"📁 출력 파일: {args.output}")
        print(f"⏱️  총 소요 시간: {format_time(total_time)}")
        print(f"📊 처리된 프레임: {processed_count}개")
        print(f"🎯 업스케일 배율: {args.scale}x")
        print(f"🎬 출력 FPS: {output_fps:.2f}")
        
        # 파일 크기 정보
        if os.path.exists(args.output):
            file_size = os.path.getsize(args.output) / (1024 * 1024)  # MB
            print(f"💾 파일 크기: {file_size:.2f} MB")
        
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n❌ 사용자에 의해 중단되었습니다.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 