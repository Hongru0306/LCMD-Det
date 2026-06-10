from ultralytics import YOLO, RTDETR
from swanlab.integration.ultralytics import add_swanlab_callback


if __name__ == '__main__':
    # 训练  
    model = YOLO('./lcmd_n_hbb.yaml') 
    
    
    model.train(data='./data/M3FD_multi.yaml',
                cache=False,
                imgsz=640,
                epochs=2,
                batch=16,
                workers=8,
                # device='1',
                # optimizer='SGD', # using SGD
                # patience=0, # set 0 to close earlystop.
                amp=False, # close amp
                # fraction=0.2,
                project='runs/abl2',
                name='exp',
                )


