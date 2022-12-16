
import os
import cv2
import torch
import time
from glob import glob
from tool.utils import load_class_names, plot_boxes_cv2
from tool.torch_utils import do_detect


def detect_cv2(imgfolder):
    imglist = glob(f'{imgfolder}/*.jpg')
    class_names = ['tabacco']

    for imgfile in imglist:
        start = time.time()
        img = cv2.imread(imgfile)
        sized = cv2.resize(img, (416, 416))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        for i in range(2):  # This 'for' loop is for speed check
                            # Because the first iteration is usually longer
            boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)
            print(boxes)
            if boxes[0]:
                plot_boxes_cv2(img, boxes[0], f'results/{os.path.basename(imgfile)}', class_names)
        

if __name__=='__main__':
    from models import Yolov4
    n_classes = 1
    weightfile = 'checkpoints/Yolov4_epoch114.pth'
    # model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)

    # pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
    # model.load_state_dict(pretrained_dict)

    use_cuda = True
    # if use_cuda:
    #     model.cuda()
    
    model = Yolov4(yolov4conv137weight=None, n_classes=1, inference=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device=device)
    pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
    model.load_state_dict(pretrained_dict)
    model.cuda()
    model.eval()
    imgfolder = 'dataset/tabacco/val'
    detect_cv2(imgfolder)