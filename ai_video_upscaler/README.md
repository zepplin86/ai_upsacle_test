# AI 비디오 업스케일링 파이프라인

PyTorch와 ffmpeg를 사용하여 저화질 영상을 프레임 단위로 복원하고 다시 영상으로 합성하는 Python 파이프라인입니다.

## 🎯 주요 기능

- **프레임 추출**: OpenCV 또는 ffmpeg를 사용한 고속 프레임 추출
- **AI 업스케일링**: Real-ESRGAN 또는 SwinIR 모델을 사용한 고해상도 복원
- **영상 재합성**: 복원된 프레임을 원본 FPS로 영상 재합성
- **배치 처리**: 대용량 영상의 효율적인 처리
- **다양한 옵션**: 해상도, FPS, 코덱 등 다양한 파라미터 조정 가능

## 📁 프로젝트 구조

```
ai_video_upscaler/
├── main.py                    # 메인 실행 파일
├── models/
│   ├── __init__.py
│   └── realesrgan_runner.py   # AI 모델 러너 (Real-ESRGAN, SwinIR)
├── utils/
│   ├── __init__.py
│   ├── extract_frames.py      # 영상 → 프레임 추출
│   ├── reassemble_video.py    # 프레임 → 영상 재합성
│   └── file_utils.py          # 파일 유틸리티
├── input/                     # 입력 영상 파일 (gitignore됨)
├── output/                    # 결과 영상 및 이미지 저장 (gitignore됨)
├── requirements.txt           # Python 패키지 의존성
├── test_pipeline.py           # 테스트 스크립트
└── README.md                  # 프로젝트 문서
```

## 🚀 설치 및 설정

### 1. 시스템 요구사항

- Python 3.8 이상 (권장: Python 3.10)
- CUDA 지원 GPU (선택사항, CPU도 사용 가능)
- ffmpeg (권장, 더 빠른 처리)

### 2. 저장소 클론

```bash
git clone <repository-url>
cd ai_video_upscaler
```

### 3. 가상환경 설정 (권장)

```bash
# Python 3.10 가상환경 생성 (PyTorch 호환성을 위해)
python3.10 -m venv venv

# 가상환경 활성화
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate     # Windows
```

### 4. 의존성 설치

```bash
# 기본 패키지 설치
pip install -r requirements.txt

# numpy 버전 다운그레이드 (PyTorch 호환성)
pip install numpy==1.24.3

# ffmpeg 설치 (macOS)
brew install ffmpeg

# ffmpeg 설치 (Ubuntu/Debian)
sudo apt update
sudo apt install ffmpeg

# ffmpeg 설치 (Windows)
# https://ffmpeg.org/download.html 에서 다운로드
```

### 5. GPU 사용 (선택사항)

CUDA를 사용하려면 PyTorch를 GPU 버전으로 재설치하세요:

```bash
# CUDA 11.8 버전 예시
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 💻 사용법

### 기본 사용법

```bash
# 가상환경 활성화
source venv/bin/activate

# 기본 설정으로 업스케일링
python main.py --input input/sample.mp4

# 2배 업스케일링
python main.py --input input/sample.mp4 --scale 2

# SwinIR 모델 사용
python main.py --input input/sample.mp4 --model swinir

# GPU 사용
python main.py --input input/sample.mp4 --device cuda
```

### 고급 옵션

```bash
# ffmpeg 사용 (더 빠름)
python main.py --input input/sample.mp4 --use-ffmpeg

# 특정 프레임 범위만 처리
python main.py --input input/sample.mp4 --start-frame 100 --end-frame 200

# 중간 프레임 파일 유지
python main.py --input input/sample.mp4 --keep-frames

# 사용자 정의 출력 경로
python main.py --input input/sample.mp4 --output output/my_video_upscaled.mp4

# 고품질 설정
python main.py --input input/sample.mp4 --quality 100 --codec libx265 --preset slow

# 최대 속도 모드 (모든 최적화 활성화)
python main.py --input input/sample.mp4 --max-speed

# 개별 최적화 옵션
python main.py --input input/sample.mp4 --memory-efficient --use-amp --num-workers 4
```

### 성능 최적화 옵션

```bash
# 배치 크기 조정 (GPU 메모리에 따라)
python main.py --input input/sample.mp4 --batch-size 2

# 병렬 처리 워커 수 설정
python main.py --input input/sample.mp4 --num-workers 4

# 메모리 효율 모드 (타일 크기 자동 조정)
python main.py --input input/sample.mp4 --memory-efficient

# Automatic Mixed Precision 사용 (GPU에서만)
python main.py --input input/sample.mp4 --use-amp

# 반정밀도 사용 (GPU에서만)
python main.py --input input/sample.mp4 --half-precision
```

### 명령행 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--input, -i` | 입력 비디오 파일 경로 | 필수 |
| `--output, -o` | 출력 비디오 파일 경로 | `output/final_upscaled_video.mp4` |
| `--model, -m` | AI 모델 선택 (`realesrgan`, `swinir`) | `realesrgan` |
| `--scale, -s` | 업스케일 배율 (2, 3, 4) | `4` |
| `--device, -d` | 사용 디바이스 (`auto`, `cpu`, `cuda`) | `auto` |
| `--fps, -f` | 출력 비디오 FPS | 원본과 동일 |
| `--frame-format` | 프레임 이미지 형식 (`jpg`, `png`) | `jpg` |
| `--quality` | 이미지 품질 (1-100) | `95` |
| `--start-frame` | 처리 시작 프레임 번호 | `0` |
| `--end-frame` | 처리 종료 프레임 번호 | 끝까지 |
| `--use-ffmpeg` | ffmpeg 사용 (더 빠름) | False |
| `--keep-frames` | 중간 프레임 파일 유지 | False |
| `--codec` | 비디오 코덱 | `libx264` |
| `--preset` | 인코딩 프리셋 | `medium` |
| `--tile-size` | 타일 크기 (메모리 절약용) | `400` |
| `--tile-pad` | 타일 패딩 크기 | `10` |
| `--half-precision` | 반정밀도 사용 (GPU에서만) | False |
| `--pre-pad` | 사전 패딩 크기 | `0` |
| `--high-quality` | 고품질 모드 | False |
| `--text-sharpen` | 텍스트 샤프닝 후처리 | `none` |
| `--sharpen-strength` | 선명화 강도 (0-1) | `0.5` |
| `--batch-size` | 배치 크기 | `1` |
| `--num-workers` | 병렬 처리 워커 수 | `0` |
| `--memory-efficient` | 메모리 효율 모드 | False |
| `--use-amp` | Automatic Mixed Precision | False |
| `--max-speed` | 최대 속도 모드 | False |

## 🤖 지원하는 AI 모델

### Real-ESRGAN
- **장점**: 일반적인 이미지에 대한 우수한 성능
- **특징**: GAN 기반, 다양한 노이즈 제거
- **권장 용도**: 일반적인 영상, 노이즈가 있는 영상

### SwinIR
- **장점**: Transformer 기반, 높은 품질
- **특징**: 더 정확한 디테일 복원
- **권장 용도**: 고품질이 필요한 영상, 텍스트가 많은 영상

## ⚡ 성능 최적화 팁

### 1. 최대 속도 모드 (권장)
```bash
python main.py --input video.mp4 --max-speed
```
- 모든 최적화 옵션을 자동으로 활성화
- GPU 메모리에 따라 자동 조정
- 가장 빠른 처리 속도

### 2. 개별 최적화 옵션
```bash
# 메모리 효율 모드 (GPU 메모리 자동 조정)
python main.py --input video.mp4 --memory-efficient

# Automatic Mixed Precision (GPU에서만)
python main.py --input video.mp4 --use-amp

# 병렬 처리 (CPU 코어 활용)
python main.py --input video.mp4 --num-workers 4

# 배치 처리 (GPU 메모리 허용 시)
python main.py --input video.mp4 --batch-size 2

# 반정밀도 (GPU에서만, 품질 약간 저하)
python main.py --input video.mp4 --half-precision
```

### 3. ffmpeg 사용
```bash
python main.py --input video.mp4 --use-ffmpeg
```
- 프레임 추출과 비디오 합성이 더 빠름
- 메모리 사용량 감소

### 4. GPU 사용
```bash
python main.py --input video.mp4 --device cuda
```
- AI 모델 추론 속도 대폭 향상
- CUDA 지원 GPU 필요

### 5. 프레임 범위 제한
```bash
python main.py --input video.mp4 --start-frame 100 --end-frame 200
```
- 긴 영상의 일부분만 테스트할 때 유용

### 6. 품질 vs 속도 트레이드오프
```bash
# 빠른 처리
python main.py --input video.mp4 --preset ultrafast --quality 80

# 고품질 처리
python main.py --input video.mp4 --preset slow --quality 100
```

### 7. GPU 메모리별 최적 설정

#### 4GB 이하 GPU
```bash
python main.py --input video.mp4 --memory-efficient --tile-size 200
```

#### 8GB GPU
```bash
python main.py --input video.mp4 --batch-size 2 --num-workers 2
```

#### 12GB 이상 GPU
```bash
python main.py --input video.mp4 --max-speed
```

## 🧪 성능 테스트

다양한 최적화 옵션의 성능을 비교하려면 성능 테스트 스크립트를 사용하세요:

```bash
# 기본 성능 테스트 (50프레임, 2배 업스케일)
python test_performance.py --input input/sample.mp4

# 커스텀 설정으로 테스트
python test_performance.py --input input/sample.mp4 --scale 4 --frames 100

# 결과를 특정 디렉토리에 저장
python test_performance.py --input input/sample.mp4 --output-dir my_test_results
```

### 테스트 항목
- 기본 설정 (CPU)
- GPU 기본
- 메모리 효율 모드
- AMP (Automatic Mixed Precision)
- 병렬 처리
- 반정밀도
- 최대 속도 모드
- ffmpeg + 최대 속도

### 결과 해석
- 각 설정별 처리 시간 비교
- 상대적 성능 비율 표시
- 최고/최저 성능 설정 식별
- 성능 향상률 계산

## 📊 처리 시간 예상

| 영상 길이 | 해상도 | 배율 | GPU | 예상 시간 |
|-----------|--------|------|-----|-----------|
| 1분 | 720p | 4x | RTX 3080 | 5-10분 |
| 1분 | 720p | 4x | CPU | 30-60분 |
| 5분 | 1080p | 4x | RTX 3080 | 25-50분 |
| 5분 | 1080p | 4x | CPU | 2-4시간 |
| 6분 33초 | 720p | 2x | CPU | 1-2시간 |

## 🧪 테스트

프로젝트에 포함된 테스트 스크립트를 사용하여 파이프라인을 테스트할 수 있습니다:

```bash
# 테스트 실행
python test_pipeline.py

# 특정 모델로 테스트
python test_pipeline.py --model realesrgan
```

## 🔧 문제 해결

### 일반적인 오류

1. **모델 로딩 실패**
   ```bash
   # Real-ESRGAN 패키지 설치
   pip install realesrgan
   ```

2. **CUDA 메모리 부족**
   ```bash
   # CPU 사용
   python main.py --input video.mp4 --device cpu
   ```

3. **ffmpeg 오류**
   ```bash
   # OpenCV 사용
   python main.py --input video.mp4
   ```

4. **Python 버전 호환성 문제**
   ```bash
   # Python 3.10 사용 권장
   python3.10 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### 디버깅

```bash
# 중간 파일 유지하여 디버깅
python main.py --input video.mp4 --keep-frames

# 특정 프레임만 테스트
python main.py --input video.mp4 --start-frame 0 --end-frame 10
```

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 지원

문제가 발생하거나 질문이 있으시면 이슈를 생성해 주세요.

---

**참고**: 이 프로젝트는 교육 및 연구 목적으로 제작되었습니다. 상업적 사용 시 관련 라이선스를 확인하세요. 