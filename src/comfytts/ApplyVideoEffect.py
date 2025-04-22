import numpy as np
import cv2
import random
from PIL import Image
import torch
import torchvision.transforms as T


class ApplyVideoEffect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "effect_frames": ("IMAGE",),
                "num_effects": ("INT", {"default": 20, "min": 1, "max": 1000}),
                "start_from_sec": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10000.0}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0}),
                "effect_start_frame": ("INT", {"default": 5, "min": 0, "max": 1000}),
                "effect_end_frame": ("INT", {"default": 35, "min": 0, "max": 1000}),
                "effect_speed": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10.0}),
                "BLEND_MODE": (["blend", "replace"],),
                "GRAY_THRESHOLD": ("INT", {"default": 80, "min": 0, "max": 255}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_effect"
    CATEGORY = "custom"

    def apply_effect(self, video_frames, effect_frames, num_effects, start_from_sec, fps,
                     effect_start_frame, effect_end_frame, effect_speed, BLEND_MODE, GRAY_THRESHOLD):

        # Преобразуем батчи в списки numpy массивов
        video_list = [self.tensor_to_numpy(img) for img in video_frames]
        effect_list = [self.tensor_to_numpy(img) for img in effect_frames]

        # Определяем параметры видео
        height, width = video_list[0].shape[:2]
        total_frames = len(video_list)
        start_frame = int(start_from_sec * fps)

        # Генерируем моменты для вставки эффектов
        frames_with_effect = sorted(random.choices(
            range(start_frame, total_frames),
            k=num_effects
        ))

        # Обрабатываем кадры эффекта
        scaled_effect = self.load_effect_frames(
            effect_list,
            width,
            height,
            effect_start_frame,
            effect_end_frame,
            effect_speed,
            GRAY_THRESHOLD
        )

        # Создаем расписание эффектов
        effect_schedule = [(s, s + len(scaled_effect)) for s in frames_with_effect]

        # Применяем эффекты к видео
        output_frames = self.apply_effects(
            video_list,
            scaled_effect,
            effect_schedule,
            BLEND_MODE
        )

        # Конвертируем результат обратно в тензор
        return (torch.stack([self.numpy_to_tensor(f) for f in output_frames]),)

    def load_effect_frames(self, effect_list, t_width, t_height, start, end, speed, thresh):
        # Выбираем нужные кадры эффекта
        raw = effect_list[start:end]
        processed = []

        for frame in raw:
            # Конвертируем в BGR и альфа-канал (остается без изменений)
            if frame.shape[2] == 4:
                bgr = frame[..., :3]
                alpha = frame[..., 3] / 255.0
            else:
                bgr = frame
                gray = cv2.cvtColor(bgr, cv2.COLOR_RGB2GRAY)
                alpha = (gray > thresh).astype(np.float32)

            # Центральный кроп и ресайз (остается без изменений)
            h, w = bgr.shape[:2]
            crop_h = min(h, t_height)
            crop_w = min(w, t_width)

            y = (h - crop_h) // 2
            x = (w - crop_w) // 2

            cropped = bgr[y:y + crop_h, x:x + crop_w]
            a_crop = alpha[y:y + crop_h, x:x + crop_w]

            if cropped.shape[:2] != (t_height, t_width):
                cropped = cv2.resize(cropped, (t_width, t_height))
                a_crop = cv2.resize(a_crop, (t_width, t_height))

            processed.append((cropped, a_crop))

        # Исправленное применение скорости эффекта
        scaled = []
        max_index = len(processed) - 1  # Максимальный допустимый индекс
        i = 0.0

        while True:
            idx = int(i)
            if idx > max_index:
                break
            scaled.append(processed[idx])
            i += speed  # Увеличиваем шаг согласно скорости

        return scaled

    def apply_effects(self, video, effects, schedule, mode):
        output = []
        for idx, frame in enumerate(video):
            result = frame.copy()

            for start, end in schedule:
                if start <= idx < end:
                    eff_idx = idx - start
                    if eff_idx < len(effects):
                        eff, alpha = effects[eff_idx]
                        alpha = alpha[..., None]  # Добавляем ось для смешивания

                        if mode == "blend":
                            result = (eff * alpha + result * (1 - alpha)).astype(np.uint8)
                        else:
                            result = np.where(alpha > 0.1, eff, result)

            output.append(result)
        return output

    def tensor_to_numpy(self, tensor):
        # Конвертируем тензор ComfyUI (H, W, C) в numpy RGB
        return (tensor.cpu().numpy() * 255).astype(np.uint8)[..., ::-1]  # RGB -> BGR

    def numpy_to_tensor(self, arr):
        # Конвертируем BGR numpy в RGB тензор
        return torch.from_numpy(arr[..., ::-1].copy() / 255.0).float()
