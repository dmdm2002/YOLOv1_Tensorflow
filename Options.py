class params(object):
    def __init__(self):
        self.EPOCHS = 50
        self.lr = 1e-3
        self.bathSZ = 1
        self.root = 'D:/Datasets/car_detection/data'
        self.ckp_path = 'D:/ckp/Yolo_v1/1'
        self.outPath = ''
        self.original_H = 380
        self.original_W = 676
        self.H = 224
        self.W = 224
        self.train_cnt = 367
        self.test_cnt = 0
        self.num_classes = 12
        self.cnt = 560