from ultralytics import YOLO, RTDETR
from swanlab.integration.ultralytics import add_swanlab_callback


if __name__ == '__main__':

    model = YOLO("./weights/best.pt")
    model.val(data="./data/M3FD_multi.yaml",batch=8, split='test')
