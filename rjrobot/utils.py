import sys
import base64
import cv2
import numpy as np
def build_obj_from_dict(info_dict, parent=None, default_args=None):
    r'''从字典中建立示例对象

        Args:
            info_dict : dict
                必须有type键，值是需要实例化的类的名

            parent : str = None
                需要实例化的类的属于那个包名

            default_args : dict =None
                实例化的类的默认参数

        Returns:
            instanse: 初始化之后的类的实例
    '''
    assert isinstance(info_dict, dict) and 'type' in info_dict,"'info_dict' must be a dict and contain the key 'type'"
    assert isinstance(default_args, dict) or default_args is None,"'default_args' must be a dict or None"
    args = info_dict.copy()
    obj_type = args.pop('type')
    if 'parent' in info_dict:
        parent = args.pop('type')
    if isinstance(obj_type, str):
        if parent is not None:
            obj_type = getattr(parent, obj_type)
        else:
            obj_type = sys.modules[obj_type]
    elif not isinstance(obj_type, type):
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_type(**args)

    
# NOTE: 进程安全,进程独立
class Singleton(type):
    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cls.__instance = None

    def __call__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__call__(*args, **kwargs)
            return cls.__instance
        else:
            if kwargs and hasattr(cls.__instance, "_reinit"):
                cls.__instance._reinit(**kwargs)
            return cls.__instance

def img_to_base64(img,contain_header=False):
    """Convert image to base64 string.
    
    Args:
        img (numpy.ndarray or str): The input image as either a numpy array or a path to a local image.
        
    Returns:
        str: The base64 encoded string of the image.
    """
    # Check if img is a string (file path)
    if isinstance(img, str):
        # Read image from file
        img = cv2.imread(img)
        if img is None:
            raise ValueError(f"Failed to load image from path: {img}")
    
    # Check if img is a numpy array
    if not isinstance(img, np.ndarray):
        raise TypeError("Input must be a numpy array or a valid file path")
    
    if img.shape[0]==3 and len(img.shape)==3:
        img = np.transpose(img,(1,2,0))
    
    # Encode image to jpg format
    success, buffer = cv2.imencode('.jpg', img)
    if not success:
        raise ValueError("Failed to encode image")
    
    # Convert to base64 and then to string
    base64_str = base64.b64encode(buffer).decode('utf-8')
    if contain_header:
        base64_str = f"data:image/jpg;base64,{base64_str}"
    
    return base64_str

def base64_to_img(base64_str):
    """Convert base64 string back to numpy array image.
    
    Args:
        base64_str (str): The base64 encoded string.
        
    Returns:
        numpy.ndarray: The decoded image as a numpy array.
    """
    # Decode base64 string to bytes
    img_bytes = base64.b64decode(base64_str)
    
    # Convert bytes to numpy array
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    
    # Decode the numpy array to image
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    return img