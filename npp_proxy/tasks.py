"""Frozen TorchVision task models and their evaluation datasets."""

from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.models import ResNet50_Weights, Swin_T_Weights, ViT_B_16_Weights, resnet50, swin_t, vit_b_16
from torchvision.models.detection import (
    FCOS_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_Weights, RetinaNet_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn, fcos_resnet50_fpn, retinanet_resnet50_fpn,
)

from .codec import read_rgb
from .training import load_state


CLASSIFIERS = {'resnet50': (resnet50, ResNet50_Weights.IMAGENET1K_V1),
               'swin-t': (swin_t, Swin_T_Weights.IMAGENET1K_V1),
               'vit-b16': (vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1)}
DETECTORS = {'fcos': (fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights.COCO_V1),
             'faster-rcnn': (fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights.COCO_V1),
             'retinanet': (retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights.COCO_V1)}


class FlickrPatches(Dataset):
    def __init__(self, root, size):
        self.paths = sorted(p for p in Path(root).rglob('*') if p.is_file()
                            and p.suffix.lower() in datasets.folder.IMG_EXTENSIONS)
        self.crop = transforms.RandomCrop(size, pad_if_needed=True, padding_mode='edge')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.crop(read_rgb(self.paths[index]))


class COCOImages(Dataset):
    def __init__(self, root, annotations):
        from pycocotools.coco import COCO

        if annotations is None:
            raise ValueError('Detection requires --annotations instances_<split>2017.json')
        self.root = Path(root)
        self.coco = COCO(str(annotations))
        self.ids = sorted(self.coco.imgs)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        rgb = read_rgb(self.root / self.coco.imgs[image_id]['file_name'])
        height, width = rgb.shape[-2:]
        boxes, labels = [], []
        for annotation in self.coco.imgToAnns[image_id]:
            if annotation.get('iscrowd'):
                continue
            x, y, w, h = annotation['bbox']
            box = [max(0., x), max(0., y), min(width, x + w), min(height, y + h)]
            if box[2] > box[0] and box[3] > box[1]:
                boxes.append(box)
                labels.append(annotation['category_id'])
        target = {'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  'labels': torch.tensor(labels, dtype=torch.int64)}
        return rgb, target, image_id

    def evaluate(self, predictions, ids):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        if predictions:
            # loadRes mutates its input dictionaries and expects an info entry in older versions.
            self.coco.dataset.setdefault('info', {})
            detected = self.coco.loadRes(predictions)
        else:
            detected = COCO()
            detected.dataset = {**self.coco.dataset, 'annotations': []}
            detected.createIndex()
        evaluator = COCOeval(self.coco, detected, iouType='bbox')
        evaluator.params.imgIds = ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        if evaluator.stats[0] < 0:
            raise ValueError('COCO mAP is undefined for this subset')
        return float(evaluator.stats[0]) * 100


class Task:
    def __init__(self, kind, backbone, device, local_weights=None):
        self.kind = kind
        registry = CLASSIFIERS if kind == 'classification' else DETECTORS
        backbone = backbone or ('resnet50' if kind == 'classification' else 'fcos')
        if backbone not in registry:
            raise ValueError(f'Backbone {backbone} does not support {kind}')
        build, weights = registry[backbone]
        if local_weights is None:
            model = build(weights=weights)
            self.weight_id = f'{weights.__class__.__name__}.{weights.name}'
        else:
            options = {} if kind == 'classification' else {'weights_backbone': None, 'num_classes': 91}
            model = build(weights=None, **options)
            model.load_state_dict(load_state(local_weights))
            self.weight_id = str(Path(local_weights).resolve())
        self.backbone = backbone
        self.model = model.to(device).eval().requires_grad_(False)
        if kind == 'classification':
            settings = weights.transforms()
            self.normalize = transforms.Normalize(settings.mean, settings.std)

    def dataset(self, root, annotations, train=False):
        if self.kind == 'detection':
            return COCOImages(root, annotations)
        geometry = transforms.RandomResizedCrop(224) if train else transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
        ])
        data = datasets.ImageFolder(root, loader=read_rgb, transform=geometry)
        if len(data.classes) != 1000 or any(len(name) != 9 or not name.startswith('n')
                                           or not name[1:].isdigit() for name in data.classes):
            raise ValueError('ImageNet requires all 1000 ILSVRC WNID directories in each split')
        return data

    def loss(self, rgb, target):
        if self.kind == 'classification':
            labels = torch.as_tensor(target, device=rgb.device).reshape(-1)
            return F.cross_entropy(self.model(self.normalize(rgb)), labels)
        targets = {key: value.to(rgb.device) for key, value in target.items()}
        # Only the top-level detector switches behavior. Its frozen modules remain in eval mode.
        self.model.training = True
        try:
            return sum(self.model(list(rgb), [targets]).values())
        finally:
            self.model.training = False

    def predict(self, rgb, example):
        if self.kind == 'classification':
            label = example[1]
            return int(self.model(self.normalize(rgb.unsqueeze(0))).argmax(dim=1).item() == label)
        _, _, image_id = example
        pred = self.model([rgb])[0]
        boxes = pred['boxes'].cpu().clone()
        boxes[:, 2:] -= boxes[:, :2]
        return [{'image_id': image_id, 'category_id': label, 'bbox': box, 'score': score}
                for box, label, score in zip(boxes.tolist(), pred['labels'].tolist(), pred['scores'].tolist(), strict=True)]
