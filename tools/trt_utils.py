"""TensorRT 8.x helpers for HFTNet: INT8 entropy calibration and engine building from an exported ONNX model.
Engines carry the Ultralytics metadata header, so they load with RTDETR('model.engine') for evaluation and inference.
"""
import glob, json, os, time
import cv2, numpy as np, torch, yaml
from ultralytics import RTDETR
from ultralytics.data.augment import LetterBox

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')
_os = os


def split_images(data_yaml, split):
    """Image files of one split of a YOLO dataset yaml (folder or list file, relative to 'path')."""
    d = yaml.safe_load(open(data_yaml)); root = str(d.get('path', '') or '')
    v = d.get(split)
    if not v:
        raise SystemExit(f'split {split!r} is not defined in {data_yaml}')
    out = []
    for item in (v if isinstance(v, list) else [v]):
        p = item if _os.path.isabs(str(item)) else _os.path.join(root, str(item))
        if _os.path.isdir(p):
            out += [f for f in glob.glob(_os.path.join(p, '**', '*'), recursive=True) if f.lower().endswith(IMG_EXT)]
        elif _os.path.isfile(p):
            base = _os.path.dirname(p)
            for line in open(p):
                line = line.strip()
                if line:
                    out.append(line if _os.path.isabs(line) else _os.path.join(root or base, line))
    out = sorted(out)
    if not out:
        raise SystemExit(f'no images found for split {split!r} of {data_yaml}')
    return out


def preprocess(im, imgsz):
    """Exactly the RT-DETR predictor preprocessing: stretch to imgsz x imgsz, BGR->RGB, CHW, /255, float32."""
    x = LetterBox((imgsz, imgsz), auto=False, scaleFill=True)(image=im)
    x = np.ascontiguousarray(x[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return x


def onnx_metadata(onnx_path, weights, imgsz):
    """Metadata header for the engines: the Ultralytics ONNX metadata, or the same fields taken from the weights."""
    try:
        import onnx
        meta = {p.key: p.value for p in onnx.load(onnx_path, load_external_data=False).metadata_props}
        if {'stride', 'names', 'imgsz'} <= set(meta):
            meta['batch'] = '1'
            return meta
    except Exception:
        pass
    m = RTDETR(weights).model
    stride = int(max(m.stride)) if hasattr(m, 'stride') else 32
    return {'description': 'HFTNet', 'task': 'detect', 'stride': stride, 'batch': 1, 'imgsz': [imgsz, imgsz], 'names': m.names}


def make_calibrator(trt, files, cache, imgsz):
    class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            trt.IInt8EntropyCalibrator2.__init__(self)
            self.files, self.i = files, 0
            self.buf = torch.empty((1, 3, imgsz, imgsz), dtype=torch.float32, device='cuda')

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            while self.i < len(self.files):
                im = cv2.imread(self.files[self.i]); self.i += 1
                if im is None:
                    continue
                self.buf.copy_(torch.from_numpy(preprocess(im, imgsz))); torch.cuda.synchronize()
                if self.i % 100 == 0:
                    print(f'  INT8 calibration {self.i}/{len(self.files)}', flush=True)
                return [int(self.buf.data_ptr())]
            return None

        def read_calibration_cache(self):
            return open(cache, 'rb').read() if _os.path.exists(cache) else None

        def write_calibration_cache(self, c):
            with open(cache, 'wb') as f:
                f.write(c)
    return EntropyCalibrator()


def build_engine(onnx_path, out_path, precision, metadata, workspace_gb, calib_files, cache, imgsz):
    import tensorrt as trt
    if not trt.__version__.startswith('8.'):
        raise RuntimeError(f'TensorRT {trt.__version__} found; the Ultralytics 8.0 loader needs TensorRT 8.x (pip install tensorrt==8.6.1)')
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        raise RuntimeError('ONNX parse failed: ' + '; '.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb) << 30)
    if precision == 'fp32':
        config.clear_flag(trt.BuilderFlag.TF32)
    elif precision == 'tf32':
        config.set_flag(trt.BuilderFlag.TF32)
    elif precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'int8':
        config.set_flag(trt.BuilderFlag.INT8); config.set_flag(trt.BuilderFlag.FP16)
        config.int8_calibrator = make_calibrator(trt, calib_files, cache, imgsz)
    else:
        raise ValueError(precision)
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f'TensorRT build failed for {precision}')
    meta = json.dumps(metadata)
    with open(out_path, 'wb') as f:
        f.write(len(meta).to_bytes(4, byteorder='little', signed=True)); f.write(meta.encode()); f.write(serialized)
    return trt.__version__, time.time() - t0
