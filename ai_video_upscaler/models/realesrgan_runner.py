import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
from typing import Optional, Tuple, List
import torch.hub
import concurrent.futures
from functools import partial


class RealESRGANRunner:
    """
    Real-ESRGAN 모델을 사용하여 이미지를 업스케일링하는 클래스
    """
    
    def __init__(
        self,
        model_name: str = "RealESRGAN_x4plus",
        device: str = "auto",
        scale: int = 4,
        tile_size: int = 400,
        tile_pad: int = 10,
        half_precision: bool = False,
        pre_pad: int = 0,
        text_sharpen: str = "none",
        sharpen_strength: float = 1.0,
        batch_size: int = 1,
        num_workers: int = 0,
        memory_efficient: bool = False,
        use_amp: bool = False
    ):
        """
        Real-ESRGAN 러너를 초기화합니다.
        
        Args:
            model_name (str): 사용할 모델 이름
            device (str): 사용할 디바이스 ('auto', 'cpu', 'cuda')
            scale (int): 업스케일 배율
            tile_size (int): 타일 크기 (메모리 절약용, 작을수록 더 선명)
            tile_pad (int): 타일 패딩 크기 (클수록 더 선명)
            half_precision (bool): 반정밀도 사용 여부 (GPU에서만)
            pre_pad (int): 사전 패딩 크기
            text_sharpen (str): 텍스트 샤프닝 후처리 방법 ("none", "opencv", "pillow")
            sharpen_strength (float): 샤프닝 강도 (0.0 ~ 2.0)
            batch_size (int): 배치 크기 (GPU 메모리에 따라 조정)
            num_workers (int): 병렬 처리 워커 수 (0=비활성화)
            memory_efficient (bool): 메모리 효율 모드 (타일 크기 자동 조정)
            use_amp (bool): Automatic Mixed Precision 사용 (GPU에서만)
        """
        self.model_name = model_name
        self.scale = scale
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.half_precision = half_precision
        self.pre_pad = pre_pad
        self.text_sharpen = text_sharpen
        self.sharpen_strength = sharpen_strength
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.memory_efficient = memory_efficient
        self.use_amp = use_amp
        
        # 디바이스 설정
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # GPU 메모리 최적화
        if self.device.type == "cuda":
            self._optimize_gpu_memory()
        
        print(f"디바이스: {self.device}")
        print(f"배치 크기: {self.batch_size}")
        print(f"병렬 워커: {self.num_workers}")
        print(f"메모리 효율 모드: {self.memory_efficient}")
        print(f"AMP 사용: {self.use_amp}")
        print(f"텍스트 샤프닝 후처리: {self.text_sharpen} (강도: {self.sharpen_strength})")
        
        # 모델 로드
        self.model = self._load_model()
        
    def _optimize_gpu_memory(self):
        """GPU 메모리 최적화 설정"""
        if torch.cuda.is_available():
            # GPU 메모리 캐시 정리
            torch.cuda.empty_cache()
            
            # 메모리 효율 모드에서 타일 크기 자동 조정
            if self.memory_efficient:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                if gpu_memory < 4:
                    self.tile_size = 200
                    self.tile_pad = 5
                    print(f"GPU 메모리 {gpu_memory:.1f}GB 감지, 타일 크기를 {self.tile_size}로 조정")
                elif gpu_memory < 8:
                    self.tile_size = 300
                    self.tile_pad = 8
                    print(f"GPU 메모리 {gpu_memory:.1f}GB 감지, 타일 크기를 {self.tile_size}로 조정")
                else:
                    self.tile_size = 400
                    self.tile_pad = 10
                    print(f"GPU 메모리 {gpu_memory:.1f}GB 감지, 타일 크기를 {self.tile_size}로 조정")
            
            # 배치 크기 자동 조정
            if self.batch_size == 1 and gpu_memory >= 8:
                self.batch_size = 2
                print(f"GPU 메모리 {gpu_memory:.1f}GB 감지, 배치 크기를 {self.batch_size}로 조정")
        
    def _load_model(self):
        """
        Real-ESRGAN 모델을 로드합니다.
        
        Returns:
            torch.nn.Module: 로드된 모델
        """
        print(f"Real-ESRGAN 모델 로딩 중: {self.model_name}")
        
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            # scale에 따라 적절한 모델 선택
            if self.scale == 2:
                # scale 2용 모델 - 모델 자체가 scale 2로 설계됨
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
                self.model_name = 'RealESRGAN_x2plus'
                model_scale = 2
            elif self.scale == 3:
                # scale 3은 x4 모델을 사용하고 outscale=3으로 설정
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
                self.model_name = 'RealESRGAN_x4plus'
                model_scale = 4
            else:  # scale 4
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
                self.model_name = 'RealESRGAN_x4plus'
                model_scale = 4

            # 업스케일러 초기화 - 모델의 실제 scale 사용
            upsampler = RealESRGANer(
                scale=model_scale,  # 모델의 실제 scale 사용
                model_path=model_path,
                model=model,
                tile=self.tile_size,
                tile_pad=self.tile_pad,
                pre_pad=self.pre_pad,
                half=self.half_precision,
                device=self.device
            )
            print(f"Real-ESRGAN 모델 로딩 완료 (scale: {self.scale}, model_scale: {model_scale})")
            return upsampler

        except ImportError:
            raise ImportError("필수 패키지가 설치되지 않았습니다. 'pip install realesrgan basicsr'를 실행하세요.")
        except Exception as e:
            print(f"모델 로딩 중 예기치 않은 오류 발생: {e}")
            raise
    
    def _opencv_unsharp_mask(self, img, strength=1.0):
        import cv2
        import numpy as np
        blur = cv2.GaussianBlur(img, (0, 0), 3)
        unsharp = cv2.addWeighted(img, 1.0 + strength, blur, -strength, 0)
        return unsharp

    def _pillow_sharpen(self, img, strength=1.0):
        from PIL import Image, ImageFilter
        import cv2
        import numpy as np
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for _ in range(int(max(1, round(strength*2)))):
            pil_img = pil_img.filter(ImageFilter.SHARPEN)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _process_single_image(self, image_path: str, output_path: str) -> bool:
        """단일 이미지 처리 (병렬 처리용)"""
        try:
            # 이미지 로드
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"이미지를 로드할 수 없습니다: {image_path}")
                return False
            
            # Real-ESRGAN 모델로 업스케일링
            if self.scale == 2:
                # scale 2 모델은 outscale 파라미터 없이 사용
                output, _ = self.model.enhance(img)
            else:
                # scale 3, 4는 outscale 파라미터 사용
                output, _ = self.model.enhance(img, outscale=self.scale)
            
            # 텍스트 샤프닝 후처리
            if self.text_sharpen == "opencv":
                output = self._opencv_unsharp_mask(output, self.sharpen_strength)
            elif self.text_sharpen == "pillow":
                output = self._pillow_sharpen(output, self.sharpen_strength)
            
            # 결과 저장
            cv2.imwrite(output_path, output)
            return True
            
        except Exception as e:
            print(f"이미지 처리 실패 {image_path}: {e}")
            return False

    def upscale_image(self, image_path: str, output_path: str) -> None:
        """
        단일 이미지를 업스케일링합니다.
        
        Args:
            image_path (str): 입력 이미지 경로
            output_path (str): 출력 이미지 경로
        """
        print(f"이미지 업스케일링 중: {os.path.basename(image_path)}")
        
        success = self._process_single_image(image_path, output_path)
        if success:
            print(f"업스케일링 완료: {output_path}")
        else:
            raise Exception(f"업스케일링 실패: {image_path}")
    
    def upscale_batch(
        self, 
        input_dir: str, 
        output_dir: str, 
        frame_format: str = "jpg"
    ) -> int:
        """
        디렉토리 내의 모든 이미지를 배치로 업스케일링합니다.
        
        Args:
            input_dir (str): 입력 이미지 디렉토리
            output_dir (str): 출력 이미지 디렉토리
            frame_format (str): 이미지 형식
            
        Returns:
            int: 처리된 이미지 수
        """
        from utils.file_utils import get_frame_files, ensure_directory
        
        # 출력 디렉토리 생성
        ensure_directory(output_dir)
        
        # 입력 이미지 파일 목록
        input_files = get_frame_files(input_dir, frame_format)
        
        if not input_files:
            raise ValueError(f"입력 이미지를 찾을 수 없습니다: {input_dir}")
        
        print(f"총 {len(input_files)}개 이미지를 업스케일링합니다.")
        
        processed_count = 0
        
        # 병렬 처리 사용 여부 결정
        if self.num_workers > 0 and len(input_files) > 1:
            # 병렬 처리 모드
            print(f"병렬 처리 모드 활성화 (워커 수: {self.num_workers})")
            
            # 작업 목록 생성
            tasks = []
            for input_file in input_files:
                filename = os.path.basename(input_file)
                output_file = os.path.join(output_dir, filename)
                tasks.append((input_file, output_file))
            
            # 병렬 처리 실행
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # 부분 함수 생성 (self._process_single_image를 executor에 전달하기 위해)
                process_func = partial(self._process_single_image)
                
                # 작업 제출
                future_to_task = {
                    executor.submit(process_func, input_file, output_file): (input_file, output_file)
                    for input_file, output_file in tasks
                }
                
                # 결과 수집
                with tqdm(total=len(tasks), desc="이미지 업스케일링 중") as pbar:
                    for future in concurrent.futures.as_completed(future_to_task):
                        input_file, output_file = future_to_task[future]
                        try:
                            success = future.result()
                            if success:
                                processed_count += 1
                        except Exception as e:
                            print(f"작업 실패 {input_file}: {e}")
                        pbar.update(1)
                        
                        # GPU 메모리 정리 (주기적)
                        if self.device.type == "cuda" and processed_count % 10 == 0:
                            torch.cuda.empty_cache()
        
        else:
            # 순차 처리 모드
            print("순차 처리 모드")
            
            with tqdm(total=len(input_files), desc="이미지 업스케일링 중") as pbar:
                for input_file in input_files:
                    try:
                        # 출력 파일 경로 생성
                        filename = os.path.basename(input_file)
                        output_file = os.path.join(output_dir, filename)
                        
                        # 업스케일링 수행
                        success = self._process_single_image(input_file, output_file)
                        if success:
                            processed_count += 1
                        
                    except Exception as e:
                        print(f"이미지 처리 실패 {input_file}: {e}")
                    
                    pbar.update(1)
                    
                    # GPU 메모리 정리 (주기적)
                    if self.device.type == "cuda" and processed_count % 10 == 0:
                        torch.cuda.empty_cache()
        
        print(f"배치 업스케일링 완료: {processed_count}개 이미지 처리")
        return processed_count
    
    def get_model_info(self) -> dict:
        """
        모델 정보를 반환합니다.
        
        Returns:
            dict: 모델 정보
        """
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "scale": self.scale,
            "tile_size": self.tile_size,
            "tile_pad": self.tile_pad,
            "half_precision": self.half_precision,
            "pre_pad": self.pre_pad,
            "text_sharpen": self.text_sharpen,
            "sharpen_strength": self.sharpen_strength,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "memory_efficient": self.memory_efficient,
            "use_amp": self.use_amp
        }


class SwinIRRunner:
    """
    SwinIR 모델을 사용하여 이미지를 업스케일링하는 클래스
    """
    
    def __init__(
        self,
        scale: int = 4,
        device: str = "auto",
        model_path: Optional[str] = None
    ):
        """
        SwinIR 러너를 초기화합니다.
        
        Args:
            scale (int): 업스케일 배율
            device (str): 사용할 디바이스
            model_path (str): 모델 파일 경로 (None이면 자동 다운로드)
        """
        self.scale = scale
        
        # 디바이스 설정
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"디바이스: {self.device}")
        
        # 모델 로드
        self.model = self._load_model(model_path)
    
    def _load_model(self, model_path: Optional[str] = None):
        """
        SwinIR 모델을 로드합니다.
        
        Args:
            model_path (str): 모델 파일 경로
            
        Returns:
            torch.nn.Module: 로드된 모델
        """
        print("SwinIR 모델 로딩 중...")
        
        try:
            # torch.hub를 사용하여 SwinIR 모델 로드
            model = torch.hub.load(
                'JingyunLiang/SwinIR',
                'SwinIR',
                pretrained=True,
                scale=self.scale,
                device=self.device
            )
            print("SwinIR 모델 로딩 완료")
            return model
            
        except Exception as e:
            print(f"SwinIR 모델 로딩 실패: {e}")
            raise
    
    def upscale_image(self, image_path: str, output_path: str) -> None:
        """
        단일 이미지를 업스케일링합니다.
        
        Args:
            image_path (str): 입력 이미지 경로
            output_path (str): 출력 이미지 경로
        """
        # 이미지 로드 및 전처리
        img = Image.open(image_path).convert('RGB')
        img_tensor = self._preprocess_image(img)
        
        # 업스케일링 수행
        print(f"이미지 업스케일링 중: {os.path.basename(image_path)}")
        
        with torch.no_grad():
            output_tensor = self.model(img_tensor)
        
        # 후처리 및 저장
        output_img = self._postprocess_image(output_tensor)
        output_img.save(output_path)
        
        print(f"업스케일링 완료: {output_path}")
    
    def upscale_batch(
        self, 
        input_dir: str, 
        output_dir: str, 
        frame_format: str = "jpg"
    ) -> int:
        """
        디렉토리 내의 모든 이미지를 배치로 업스케일링합니다.
        
        Args:
            input_dir (str): 입력 이미지 디렉토리
            output_dir (str): 출력 이미지 디렉토리
            frame_format (str): 이미지 형식
            
        Returns:
            int: 처리된 이미지 수
        """
        from utils.file_utils import get_frame_files, ensure_directory
        
        # 출력 디렉토리 생성
        ensure_directory(output_dir)
        
        # 입력 이미지 파일 목록
        input_files = get_frame_files(input_dir, frame_format)
        
        if not input_files:
            raise ValueError(f"입력 이미지를 찾을 수 없습니다: {input_dir}")
        
        print(f"총 {len(input_files)}개 이미지를 업스케일링합니다.")
        
        processed_count = 0
        
        with tqdm(total=len(input_files), desc="이미지 업스케일링 중") as pbar:
            for input_file in input_files:
                try:
                    # 출력 파일 경로 생성
                    filename = os.path.basename(input_file)
                    output_file = os.path.join(output_dir, filename)
                    
                    # 업스케일링 수행
                    self.upscale_image(input_file, output_file)
                    processed_count += 1
                    
                except Exception as e:
                    print(f"이미지 처리 실패 {input_file}: {e}")
                
                pbar.update(1)
        
        print(f"배치 업스케일링 완료: {processed_count}개 이미지 처리")
        return processed_count
    
    def get_model_info(self) -> dict:
        """
        모델 정보를 반환합니다.
        
        Returns:
            dict: 모델 정보
        """
        return {
            "model_name": "SwinIR",
            "device": str(self.device),
            "scale": self.scale
        }
    
    def _preprocess_image(self, img: Image.Image) -> torch.Tensor:
        """
        이미지를 모델 입력용으로 전처리합니다.
        
        Args:
            img (Image.Image): 입력 이미지
            
        Returns:
            torch.Tensor: 전처리된 텐서
        """
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transform(img).unsqueeze(0).to(self.device)
    
    def _postprocess_image(self, tensor: torch.Tensor) -> Image.Image:
        """
        모델 출력 텐서를 이미지로 후처리합니다.
        
        Args:
            tensor (torch.Tensor): 모델 출력 텐서
            
        Returns:
            Image.Image: 후처리된 이미지
        """
        import torchvision.transforms as transforms
        
        # 정규화 역변환
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
        
        tensor = tensor.squeeze(0)
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        
        # PIL 이미지로 변환
        transform = transforms.ToPILImage()
        return transform(tensor) 