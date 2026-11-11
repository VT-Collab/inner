# object detection (SAM 3), tracking (EdgeTAM) and camera geometry
import cv2
import yaml
import numpy as np
import torch
from termcolor import colored
from transformers import Sam3Model, Sam3Processor, EdgeTamVideoModel, Sam2VideoProcessor


def load_calibration(path: str, cameras: list) -> dict:
    """per camera: K (3, 3), depth_scale (m per unit), T_base_cam (4, 4)"""
    calib = yaml.safe_load(open(path))
    out = {}
    for cam in cameras:
        assert cam in calib, f"{cam} missing from {path}"
        c = calib[cam]
        assert c.get('T_base_cam') is not None, f"T_base_cam for {cam} is not set in {path}"
        out[cam] = {
            'K': np.array([[c['fx'], 0., c['cx']],
                           [0., c['fy'], c['cy']],
                           [0., 0., 1.]], dtype=np.float32),
            'depth_scale': c.get('depth_scale'),
            'T_base_cam': np.array(c['T_base_cam'], dtype=np.float32).reshape(4, 4),
        }
    return out


def mask_centroid_3d(mask: np.ndarray,
                     depth: np.ndarray,
                     K: np.ndarray,
                     depth_scale: float,
                     erode_px: int = 3,
                     min_pixels: int = 30,
                     depth_range: tuple = (0.1, 3.0)) -> np.ndarray:
    """median 3D point (camera frame, meters) of the visible surface under a mask. NaN if too few valid pixels"""
    if erode_px > 0:
        kernel = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)

    v, u = np.nonzero(mask)
    z = depth[v, u].astype(np.float32) * depth_scale
    valid = (z > depth_range[0]) & (z < depth_range[1])
    if valid.sum() < min_pixels:
        return np.full(3, np.nan, dtype=np.float32)

    u, v, z = u[valid], v[valid], z[valid]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.median(np.stack([x, y, z], axis=1), axis=0).astype(np.float32)


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """apply a 4x4 transform to (..., 3) points. NaNs pass through"""
    return (points @ T[:3, :3].T + T[:3, 3]).astype(np.float32)


class Segmenter:
    """SAM 3 for text-prompted detection and EdgeTAM for tracking. Load once, share across cameras"""

    def __init__(self,
                 sam3_model: str = 'facebook/sam3',
                 edgetam_model: str = 'yonigozlan/EdgeTAM-hf',
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.bfloat16):
        self.device = device
        self.dtype = dtype

        self.sam3 = Sam3Model.from_pretrained(sam3_model, dtype=dtype).to(device).eval()
        self.sam3_processor = Sam3Processor.from_pretrained(sam3_model)
        self.edgetam = EdgeTamVideoModel.from_pretrained(edgetam_model, dtype=dtype).to(device).eval()
        self.edgetam_processor = Sam2VideoProcessor.from_pretrained(edgetam_model)
        print(colored("[Segmenter] ", "green") + f"Loaded {sam3_model} and {edgetam_model}")

    @torch.inference_mode()
    def detect(self, image: np.ndarray, labels: list, threshold: float = 0.5):
        """highest-scoring SAM 3 instance per label. Returns masks (K, H, W) and scores (K,), zero if not found"""
        h, w = image.shape[:2]
        masks = np.zeros((len(labels), h, w), dtype=bool)
        scores = np.zeros(len(labels), dtype=np.float32)

        img_inputs = self.sam3_processor(images=image, return_tensors='pt').to(self.device, self.dtype)
        vision_embeds = self.sam3.get_vision_features(pixel_values=img_inputs.pixel_values)

        for k, label in enumerate(labels):
            text_inputs = self.sam3_processor(text=label, return_tensors='pt').to(self.device)
            outputs = self.sam3(vision_embeds=vision_embeds,
                                input_ids=text_inputs.input_ids,
                                attention_mask=text_inputs.get('attention_mask'))
            result = self.sam3_processor.post_process_instance_segmentation(
                outputs, threshold=threshold, target_sizes=[(h, w)])[0]
            if len(result['scores']) == 0:
                continue
            best = int(result['scores'].argmax())
            masks[k] = result['masks'][best].cpu().numpy().astype(bool)
            scores[k] = float(result['scores'][best])

        return masks, scores


class CameraTracker:
    """streaming tracker for one camera: SAM 3 on the first detection, then EdgeTAM"""

    def __init__(self,
                 segmenter: Segmenter,
                 labels: list,
                 det_threshold: float = 0.5,
                 history: int = 16):
        self.seg = segmenter
        self.labels = list(labels)
        self.det_threshold = det_threshold
        self.history = history
        self.reset()

    def reset(self):
        self.session = None
        self.obj_ids = []
        self.t = 0

    @torch.inference_mode()
    def step(self, image: np.ndarray):
        """image: (H, W, 3) uint8 RGB. Returns masks (K, H, W) bool and scores (K,) in [0, 1]"""
        h, w = image.shape[:2]
        masks = np.zeros((len(self.labels), h, w), dtype=bool)
        scores = np.zeros(len(self.labels), dtype=np.float32)

        proc = self.seg.edgetam_processor
        if self.session is None and not self._start(image):
            return masks, scores

        inputs = proc(images=image, device=self.seg.device, return_tensors='pt')
        outputs = self.seg.edgetam(inference_session=self.session,
                                   frame=inputs.pixel_values[0],
                                   frame_idx=self.t)
        pred = proc.post_process_masks([outputs.pred_masks], original_sizes=inputs.original_sizes, binarize=True)[0]
        obj_scores = torch.sigmoid(outputs.object_score_logits.float()).reshape(-1).cpu().numpy()

        for i, obj_id in enumerate(outputs.object_ids):
            masks[obj_id] = pred[i, 0].cpu().numpy()
            scores[obj_id] = obj_scores[i]

        self._prune()
        self.t += 1
        return masks, scores

    def _start(self, image):
        det_masks, det_scores = self.seg.detect(image, self.labels, self.det_threshold)
        found = np.flatnonzero(det_scores > 0)
        if len(found) == 0:
            return False

        proc = self.seg.edgetam_processor
        self.session = proc.init_video_session(inference_device=self.seg.device, dtype=self.seg.dtype)
        self.session.video_height, self.session.video_width = image.shape[:2]
        # object id is the label index
        self.obj_ids = [int(k) for k in found]
        proc.add_inputs_to_inference_session(self.session,
                                             frame_idx=self.t,
                                             obj_ids=self.obj_ids,
                                             input_masks=[det_masks[k] for k in found])
        found_labels = [self.labels[k] for k in found]
        print(colored("[CameraTracker] ", "green") + f"Tracking {found_labels} from frame {self.t}")
        return True

    def _prune(self):
        # drop old frames, the session otherwise runs out of GPU memory on long demos
        s = self.session
        s.processed_frames.pop(self.t, None)
        oldest = self.t - self.history
        for obj_idx, out in s.output_dict_per_obj.items():
            for t in [t for t in out['non_cond_frame_outputs'] if t < oldest]:
                del out['non_cond_frame_outputs'][t]
            for t in [t for t in s.frames_tracked_per_obj[obj_idx] if t < oldest]:
                del s.frames_tracked_per_obj[obj_idx][t]
