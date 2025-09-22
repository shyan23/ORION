"""
AI/ML Core Components for the ORION Medical AI System
Contains MedSAM models and analyzers
"""

import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from transformers import AutoProcessor, AutoModel
from Swin_LiteMedSAM.models.mask_decoder import MaskDecoder_Prompt
from Swin_LiteMedSAM.models.prompt_encoder import PromptEncoder
from Swin_LiteMedSAM.models.swin import SwinTransformer
from Swin_LiteMedSAM.models.transformer import TwoWayTransformer

logger = logging.getLogger(__name__)


class MedSAM_Lite(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder, self.mask_decoder, self.prompt_encoder = image_encoder, mask_decoder, prompt_encoder

    def forward(self, image, points, boxes, masks, tokens=None):
        image_embedding, fs = self.image_encoder(image)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(points=points, boxes=boxes, masks=masks, tokens=tokens)
        low_res_masks, _ = self.mask_decoder(fs, image_embeddings=image_embedding, image_pe=self.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=False)
        return low_res_masks


class LocalMedSAM:
    def __init__(self, model_path="ai_models/Swin_medsam/model.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.is_ready = None, False
        self.model_path = model_path

    def load_model(self):
        try:
            logger.info(f"🔄 Starting MedSAM model loading from {self.model_path} onto {self.device}...")

            logger.info("📋 Initializing SwinTransformer image encoder...")
            image_encoder = SwinTransformer()
            logger.info("✅ SwinTransformer image encoder initialized")

            logger.info("📋 Initializing PromptEncoder...")
            prompt_encoder = PromptEncoder(embed_dim=256, image_embedding_size=(64, 64), input_image_size=(256, 256), mask_in_chans=16)
            logger.info("✅ PromptEncoder initialized")

            logger.info("📋 Initializing MaskDecoder with TwoWayTransformer...")
            mask_decoder = MaskDecoder_Prompt(num_multimask_outputs=3, transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8), transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256)
            logger.info("✅ MaskDecoder initialized")

            logger.info("📋 Assembling MedSAM_Lite model...")
            self.model = MedSAM_Lite(image_encoder, mask_decoder, prompt_encoder)
            logger.info("✅ MedSAM_Lite model assembled")

            logger.info(f"📥 Loading checkpoint from {self.model_path}...")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            logger.info("✅ Checkpoint loaded from disk")

            logger.info("📋 Loading state dict into model...")
            self.model.load_state_dict(checkpoint.get('model', checkpoint))
            logger.info("✅ State dict loaded successfully")

            logger.info(f"📋 Moving model to {self.device} and setting eval mode...")
            self.model.to(self.device).eval()
            self.is_ready = True
            logger.info("🎉 MedSAM model loaded successfully and ready for inference!")

        except Exception as e:
            logger.error(f"❌ Failed to load MedSAM model: {e}")
            import traceback
            logger.error(f"📋 Full error traceback: {traceback.format_exc()}")

    @torch.no_grad()
    def get_mask(self, image_np, point_prompt):
        if not self.is_ready:
            return None
        h, w = image_np.shape[:2]
        norm_img = (image_np - image_np.min()) / np.ptp(image_np) * 255 if np.ptp(image_np) > 0 else np.zeros_like(image_np)
        img_3c = np.repeat(norm_img[:, :, None], 3, -1) if norm_img.ndim == 2 else norm_img
        scale = 256 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img_3c, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_padded = np.pad(img_resized, ((0, 256 - new_h), (0, 256 - new_w), (0, 0)))
        img_tensor = torch.as_tensor(img_padded).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        points = torch.as_tensor([[[point_prompt[0] * scale, point_prompt[1] * scale]]], dtype=torch.float, device=self.device)
        labels = torch.as_tensor([[1]], dtype=torch.int, device=self.device)
        masks = self.model(img_tensor, (points, labels), None, torch.zeros((1, 1, 256, 256), dtype=torch.float).to(self.device))
        masks = F.interpolate(F.interpolate(masks, (256, 256), mode="bilinear", align_corners=False)[:, :, :new_h, :new_w], (h, w), mode="bilinear", align_corners=False)
        return (torch.sigmoid(masks[0,0]) > 0.5).cpu().numpy().astype(np.uint8)


class LocalROIAnalyzer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.processor, self.is_ready = None, None, False
        self.candidate_labels = ["normal soft tissue", "fat", "fluid or cyst", "bone or calcification", "air or gas", "a high-density lesion", "a low-density lesion", "a blood vessel", "lung tissue"]
        self.text_prompts = [f"a medical image of {label}" for label in self.candidate_labels]

    def _monitor_download_progress(self, cache_dir="/root/.cache/huggingface", expected_size_mb=3354, stop_event=None):
        """Monitor HuggingFace model download progress"""
        start_time = time.time()
        prev_size = 0

        while not stop_event.is_set():
            try:
                # Get current cache size
                result = os.popen(f"du -sm {cache_dir} 2>/dev/null || echo '0'").read().strip()
                current_size = int(result.split()[0]) if result and result.split()[0].isdigit() else 0

                # Calculate progress
                progress = min(current_size / expected_size_mb * 100, 100)
                speed = (current_size - prev_size) / 2 if prev_size > 0 else 0  # MB/s (2 second intervals)
                elapsed = time.time() - start_time

                # Create progress bar
                bar_length = 30
                filled_length = int(bar_length * progress / 100)
                bar = "█" * filled_length + "░" * (bar_length - filled_length)

                logger.info(f"📥 Downloading: {bar} {progress:.1f}% ({current_size}/{expected_size_mb} MB) Speed: {speed:.1f} MB/s Elapsed: {elapsed:.0f}s")

                if progress >= 100:
                    break

                prev_size = current_size
                time.sleep(2)

            except Exception as e:
                logger.warning(f"Progress monitoring error: {e}")
                break

    def load_model(self):
        try:
            import threading
            logger.info("🔄 Loading multimodal analysis model...")

            # Start download progress monitoring in background
            stop_event = threading.Event()
            progress_thread = threading.Thread(target=self._monitor_download_progress, args=("/root/.cache/huggingface", 3354, stop_event))
            progress_thread.daemon = True
            progress_thread.start()

            logger.info("📥 Downloading microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 model...")
            self.processor = AutoProcessor.from_pretrained("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", trust_remote_code=True)
            self.model = AutoModel.from_pretrained("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", trust_remote_code=True).to(self.device)

            # Stop progress monitoring
            stop_event.set()
            progress_thread.join(timeout=1)

            self.model.eval()
            self.is_ready = True
            logger.info("✅ Multimodal analysis model loaded and ready!")

        except Exception as e:
            logger.error(f"❌ Failed to load multimodal analysis model: {e}")
            import traceback
            logger.error(f"📋 Full error traceback: {traceback.format_exc()}")

    @torch.no_grad()
    def analyze_roi_content(self, roi_image_np):
        if not self.is_ready:
            return {"error": "Model not loaded", "confidence_scores": {}}

        try:
            from PIL import Image
            roi_pil = Image.fromarray((roi_image_np * 255).astype(np.uint8)) if roi_image_np.max() <= 1 else Image.fromarray(roi_image_np.astype(np.uint8))

            inputs = self.processor(text=self.text_prompts, images=roi_pil, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model(**inputs)

            image_features = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_features = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)

            similarity_scores = (image_features @ text_features.T).squeeze(0)
            probabilities = torch.softmax(similarity_scores, dim=0).cpu().numpy()

            confidence_scores = {label: float(prob) for label, prob in zip(self.candidate_labels, probabilities)}
            most_likely = max(confidence_scores, key=confidence_scores.get)

            return {
                "most_likely_content": most_likely,
                "confidence": confidence_scores[most_likely],
                "confidence_scores": confidence_scores
            }

        except Exception as e:
            logger.error(f"❌ ROI analysis failed: {e}")
            return {"error": str(e), "confidence_scores": {}}