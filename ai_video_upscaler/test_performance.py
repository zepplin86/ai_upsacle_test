#!/usr/bin/env python3
"""
GPU 성능 최적화 테스트 스크립트
다양한 최적화 옵션으로 처리 속도를 비교합니다.
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_test(input_video: str, test_name: str, command_args: list) -> dict:
    """특정 설정으로 성능 테스트를 실행합니다."""
    print(f"\n{'='*60}")
    print(f"🧪 테스트: {test_name}")
    print(f"{'='*60}")
    
    # 명령어 구성
    cmd = ["python", "main.py", "--input", input_video] + command_args
    
    print(f"실행 명령어: {' '.join(cmd)}")
    
    # 시작 시간 기록
    start_time = time.time()
    
    try:
        # 명령어 실행
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1시간 타임아웃
        )
        
        # 종료 시간 기록
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ 성공! 소요 시간: {duration:.2f}초")
            return {
                "test_name": test_name,
                "duration": duration,
                "success": True,
                "output": result.stdout
            }
        else:
            print(f"❌ 실패! 오류: {result.stderr}")
            return {
                "test_name": test_name,
                "duration": duration,
                "success": False,
                "error": result.stderr
            }
            
    except subprocess.TimeoutExpired:
        print("❌ 타임아웃 (1시간 초과)")
        return {
            "test_name": test_name,
            "duration": 3600,
            "success": False,
            "error": "Timeout"
        }
    except Exception as e:
        print(f"❌ 예외 발생: {e}")
        return {
            "test_name": test_name,
            "duration": 0,
            "success": False,
            "error": str(e)
        }

def print_results(results: list):
    """테스트 결과를 출력합니다."""
    print(f"\n{'='*80}")
    print("📊 성능 테스트 결과")
    print(f"{'='*80}")
    
    # 성공한 테스트만 필터링
    successful_results = [r for r in results if r["success"]]
    
    if not successful_results:
        print("❌ 성공한 테스트가 없습니다.")
        return
    
    # 시간순으로 정렬
    successful_results.sort(key=lambda x: x["duration"])
    
    print(f"{'테스트명':<25} {'소요시간':<15} {'상대속도':<15}")
    print("-" * 55)
    
    fastest_time = successful_results[0]["duration"]
    
    for result in successful_results:
        relative_speed = fastest_time / result["duration"] * 100
        print(f"{result['test_name']:<25} {result['duration']:<15.2f}초 {relative_speed:<15.1f}%")
    
    print(f"\n🏆 최고 성능: {successful_results[0]['test_name']} ({successful_results[0]['duration']:.2f}초)")
    print(f"🐌 최저 성능: {successful_results[-1]['test_name']} ({successful_results[-1]['duration']:.2f}초)")
    
    # 성능 향상률 계산
    if len(successful_results) > 1:
        improvement = (successful_results[-1]['duration'] - fastest_time) / fastest_time * 100
        print(f"📈 최대 성능 향상: {improvement:.1f}%")

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="GPU 성능 최적화 테스트")
    parser.add_argument("--input", "-i", required=True, help="테스트할 비디오 파일")
    parser.add_argument("--scale", "-s", type=int, default=2, choices=[2, 3, 4], help="업스케일 배율")
    parser.add_argument("--frames", "-f", type=int, default=50, help="테스트할 프레임 수")
    parser.add_argument("--output-dir", "-o", default="performance_test_results", help="결과 저장 디렉토리")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ 입력 파일을 찾을 수 없습니다: {args.input}")
        sys.exit(1)
    
    # 결과 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 GPU 성능 최적화 테스트 시작")
    print(f"입력 파일: {args.input}")
    print(f"업스케일 배율: {args.scale}x")
    print(f"테스트 프레임 수: {args.frames}")
    
    # 테스트 설정들
    test_configs = [
        {
            "name": "기본 설정",
            "args": [
                "--scale", str(args.scale),
                "--end-frame", str(args.frames),
                "--output", f"{args.output_dir}/basic_output.mp4"
            ]
        },
        {
            "name": "GPU 기본",
            "args": [
                "--scale", str(args.scale),
                "--device", "cuda",
                "--end-frame", str(args.frames),
                "--output", f"{args.output_dir}/gpu_basic_output.mp4"
            ]
        },
        {
            "name": "메모리 효율",
            "args": [
                "--scale", str(args.scale),
                "--device", "cuda",
                "--memory-efficient",
                "--end-frame", str(args.frames),
                "--output", f"{args.output_dir}/memory_efficient_output.mp4"
            ]
        },
        {
            "name": "AMP 활성화",
            "args": [
                "--scale", str(args.scale),
                "--device", "cuda",
                "--use-amp",
                "--end-frame", str(args.frames),
                "--output", f"{args.output_dir}/amp_output.mp4"
            ]
        },
        {
            "name": "병렬 처리",
            "args": [
                "--scale", str(args.scale),
                "--device", "cuda",
                "--num-workers", "4",
                "--end-frame", str(args.frames),
                "--output", f"{args.output_dir}/parallel_output.mp4"
            ]
        },
        {
            "name": "반정밀도",
            "args": [
                "--scale", str(args.scale),
                "--device", "cuda",
                "--half-precision",
                "--end-frame", str(args.frames),
                "--output", f"{args.output_dir}/half_precision_output.mp4"
            ]
        },
        {
            "name": "최대 속도",
            "args": [
                "--scale", str(args.scale),
                "--device", "cuda",
                "--max-speed",
                "--end-frame", str(args.frames),
                "--output", f"{args.output_dir}/max_speed_output.mp4"
            ]
        },
        {
            "name": "ffmpeg + 최대 속도",
            "args": [
                "--scale", str(args.scale),
                "--device", "cuda",
                "--max-speed",
                "--use-ffmpeg",
                "--end-frame", str(args.frames),
                "--output", f"{args.output_dir}/ffmpeg_max_speed_output.mp4"
            ]
        }
    ]
    
    results = []
    
    # 각 테스트 실행
    for config in test_configs:
        result = run_test(args.input, config["name"], config["args"])
        results.append(result)
        
        # 중간 결과 출력
        if result["success"]:
            print(f"✅ {config['name']}: {result['duration']:.2f}초")
        else:
            print(f"❌ {config['name']}: 실패")
    
    # 최종 결과 출력
    print_results(results)
    
    # 결과를 파일로 저장
    results_file = os.path.join(args.output_dir, "performance_results.txt")
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("GPU 성능 최적화 테스트 결과\n")
        f.write("=" * 50 + "\n\n")
        
        for result in results:
            f.write(f"테스트: {result['test_name']}\n")
            f.write(f"성공: {result['success']}\n")
            f.write(f"소요시간: {result['duration']:.2f}초\n")
            if not result['success']:
                f.write(f"오류: {result.get('error', 'Unknown')}\n")
            f.write("-" * 30 + "\n")
    
    print(f"\n📁 결과가 {results_file}에 저장되었습니다.")

if __name__ == "__main__":
    main() 