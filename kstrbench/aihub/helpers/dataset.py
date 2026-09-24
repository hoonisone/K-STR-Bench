from abc import *
from PIL import Image
import json
from pathlib import Path

class Dataset_Loader(metaclass=ABCMeta):
    def __init__(self):
        pass
        
    @abstractmethod    
    def get_x(self, index):
        raise NotImplementedError("Subclasses must override this method")
    
    @abstractmethod
    def get_y(self, index):
        raise NotImplementedError("Subclasses must override this method")
    
    def show_x(self, index):
        self._show_x(self.get_x(index))
    
    def show_y(self, index):
        self._show_y(self.get_y(index))
        
    def show_x_y(self, index):
        self._show_x_y(self.get_x(index), self.get_y(index))
        
    def __getitem__(self, index):
        return self.get_x(index), self.get_y(index)
    
    @abstractmethod    
    def _show_x(self, x):
        raise NotImplementedError("Subclasses must override this method")
    
    @abstractmethod
    def _show_y(self, index):
        raise NotImplementedError("Subclasses must override this method")
    
    @abstractmethod
    def _show_x_y(self, x, y):
        raise NotImplementedError("Subclasses must override this method")
    
    @abstractmethod
    def __len__(self):
        raise NotImplementedError("Subclasses must override this method")

    @staticmethod
    def xywh_to_polygon(x, y, w, h):
        return [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]

    @staticmethod
    def centered_xywh_to_polygon(cx, cy, w, h):
        x = cx-w/2
        y = cy-h/2
        return Dataset_Loader.xywh_to_polygon(x, y, w, h)
    
    @staticmethod
    def polygon_to_xywh(polygon):
        x = [point[0] for point in polygon]
        y = [point[1] for point in polygon]
        min_x, max_x = min(x), max(x)
        min_y, max_y = min(y), max(y)
        x, y, w, h = min_x, min_y, max_x-min_x, max_y-min_y
        return [x, y, w, h]

    @staticmethod
    def polygon_to_centered_xywh(polygon):
        x = [point[0] for point in polygon]
        y = [point[1] for point in polygon]
        min_x, max_x = min(x), max(x)
        min_y, max_y = min(y), max(y)
        x, y, w, h = min_x, min_y, max_x-min_x, max_y-min_y
        return [x+w/2, y+h/2, w, h]
    
    @staticmethod
    def load_image(path):
        return Image.open(path)
    
# class PPOCR_STD_Dataset_Loader(Dataset_Loader):
#     def __init__(self, path, label_file_name):
#         label_path = Path(path)/label_file_name
#         with open(label_path) as f:
#             lines = [line.rstrip("\n") for line in f.readlines()]
#             self.x_path_list = [Path(path)/line.split("\t")[0] for line in lines]
#             self.y_list = [json.loads(line.split("\t")[1]) for line in lines]        

#     def get_x(self, index):
#         return {"image":self.load_image(self.x_path_list[index])}
        
#     def get_y(self, index):
#         y = self.y_list[index]
#         return [{"text":v["transcription"], "polygon":v["points"]} for v in y]
    
#     def __len__(self):
#         return len(self.y_list)
    
class Dataset_Converter(metaclass=ABCMeta):
    # 데이터 셋 로더의 wrapper로 데이터 형태를 변환하여 반환한다.
    # 데이터 셋 도메인이 바뀌어 레이블 형태를 바꿔야 하는 경우 등에 사용 가능
    # x, y에 대한 transtorm 메서드만 override해주면 된다.
    
    def __init__(self, dataset_loader):
        self.dataset_loader = dataset_loader
        
    def __len__(self):
        return len(self.dataset_loader)
    
    def get_x(self, index):
        return self.transform_x(self.dataset_loader.get_x(index))

    def get_y(self, index):
        return self.transform_y(self.dataset_loader.get_y(index))
    
    def __getitem__(self, index):
        return self.get_x(index), self.get_y(index)
    
    def show_x(self, index):    
        self.dataset_loader._show_x(self.get_x(index))
    
    def show_y(self, index):
        self.dataset_loader._show_y(self.get_y(index))
        
    def show_x_y(self, index):
        self._show_x_y(self.get_x(index), self.get_y(index))
        
    @abstractmethod
    def transform_x(self, x):
        raise NotImplementedError("Subclasses must override this method")
    
    @abstractmethod
    def transform_y(self, y):
        raise NotImplementedError("Subclasses must override this method")