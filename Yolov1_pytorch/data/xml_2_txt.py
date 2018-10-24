import xml.etree.ElementTree as ET
import os
from config import opt

def parse_rec(filename):
    """
    Parse a PASCAL VOC xml file
    解析一个 PASCAL VOC xml file
    将数据集从xml解析为txt  用于生成voc2007test.txt等
    """
    tree = ET.parse(filename)
    objects = []
    # 遍历一张图中的所有物体
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        #obj_struct['pose'] = obj.find('pose').text
        #obj_struct['truncated'] = int(obj.find('truncated').text)
        #obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        # 从原图左上角开始为原点，向右为x轴，向下为y轴。左上角（xmin，ymin）和右下角(xmax,ymax)
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text)),
                              int(float(bbox.find('ymax').text))]
        objects.append(obj_struct)

    return objects

# 新建一个名为voc2012train的txt文件，准备写入数据
txt_file = open('data/voc2012train.txt','w')
Annotations = opt.train_Annotations
xml_files = os.listdir(Annotations)

# 遍历所有的xml
for xml_file in xml_files:
    image_path = xml_file.split('.')[0] + '.jpg'
    # txt 写入图像名字   非完整路径
    txt_file.write(image_path+' ')
    results = parse_rec(Annotations + xml_file)
    num_obj = len(results)
    # txt写入  一张图中的物体总数
    txt_file.write(str(num_obj)+' ')
    # 遍历一张图片中的所有物体
    for result in results:
        class_name = result['name']
        bbox = result['bbox']
        class_name = opt.VOC_CLASSES.index(class_name)
        # txt写入  bbox的坐标 以及  每个物体对应的类的序号
        txt_file.write(str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+' '+str(class_name)+' ')
    txt_file.write('\n')
    #最后格式:图像名（1个值）   物体总数（1个值）    bbox坐标（4个值）   物体对应的类的序号（1个值）

txt_file.close()