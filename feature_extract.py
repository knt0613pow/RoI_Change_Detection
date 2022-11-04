from torchvision import models
import copy
import glob 
import torch
from data_loader.data_loaders import ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import utils
from torchvision.ops import roi_align
import numpy as np


def collate_fn(batch):
    img, target, path = tuple(zip(*batch))
    img = torch.stack(img, 0)
    for idx, tgt in enumerate(target):
        tgt["boxes"] = tgt["boxes"].to(device)
        tgt["labels"] = tgt["labels"].to(device)
    return img.to(device), target, path

def faster_rcnn(model, data, labels):
    original_image_sizes = data[0].shape[-2:]
    for label in labels:
        if label['boxes'].shape[0] == 0 : 
            return None, None
            label['boxes'] = torch.Tensor([0.0,0.,0.,0.]).to(device)
            label['labels'] = torch.Tensor([0]).to(device)
    images, labels = model.transform(data, labels)
    

    fmap_multiscale = model.backbone(images.tensors)
    proposals, _ = model.rpn(images, fmap_multiscale, labels)
    detections, _ = model.roi_heads(fmap_multiscale, proposals, images.image_sizes, labels)
    boxes = copy.deepcopy(detections)
    
    return fmap_multiscale , boxes

Weight_path = 'OD_weight/Daesan2_rcnn_45.pt'
data_path = 'data'


batch_size = 1
images = glob.glob(f'{data_path}/*.jpg')
num_image = len(images)
trsfm = transforms.Compose([transforms.ToTensor(), ])
device = torch.device("cuda")

dataset = ImageDataset(data_path, trsfm)
DL = DataLoader(dataset, batch_size= batch_size, shuffle=False,  collate_fn=  collate_fn)
model = models.detection.fasterrcnn_resnet50_fpn(num_classes = 2,pretrained=False)
model.load_state_dict(torch.load(Weight_path))
model.to(device)

in_features = model.roi_heads.box_predictor.cls_score.in_features
layers = list(model.roi_heads.children())[:2]
roi_fmap_obj = copy.deepcopy(layers[1])
roi_fmap = copy.deepcopy(layers[1])
roi_pool = copy.deepcopy(layers[0])

pooling_size = 7
stride = 16


for batch_idx, (data, target, pathes) in enumerate(DL):

    boxes = [tgt['boxes'] for tgt in target]
    fmap , _= faster_rcnn(model, data, target)
    if fmap == None: 
        path = pathes[0]
        path  = path[:-3] + 'npy'
        feature_mat = np.zeros((1,1))
        np.save(path, feature_mat) 
        continue
    
    n_boxes = [len(box) for box in boxes]

    idx_boxes = []
    idx_temp = 0
    for n_box in n_boxes:
        idx_boxes.append(idx_temp)
        idx_temp += n_box
    idx_boxes.append(idx_temp)
    ROI_feature = model.roi_heads.box_roi_pool(fmap, boxes, data.shape)
    ROI_feature = model.roi_heads.box_head(ROI_feature)
    for idx, box in enumerate(boxes):
        path = pathes[idx]
        path = path[:-3] + 'npy'
        feature_mat = ROI_feature[idx_boxes[idx]: idx_boxes[idx+1]].cpu().detach().numpy() 
        np.save(path, feature_mat) 
    








# def extract_node_edge()