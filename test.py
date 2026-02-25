from utils_mm import mm

if __name__ == '__main__':

    # val
    model = mm("weights/best.pt")
    model.val(data=r"/cfg/datasets/mydata.yaml",batch=1)

