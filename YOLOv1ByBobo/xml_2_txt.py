import xml.etree.ElementTree as ET
import os

VOC_CLASSES = (    # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        #obj_struct['pose'] = obj.find('pose').text
        #obj_struct['truncated'] = int(obj.find('truncated').text)
        #obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text)),
                              int(float(bbox.find('ymax').text))]
        objects.append(obj_struct)

    return objects

txt_file = open('voc2012.txt','w')
#bobo   change dir
Annotations = '/home/zhuhui/data/VOCdevkit/VOC2012/Annotations/'
xml_files = os.listdir(Annotations)

count = 0
for xml_file in xml_files:
    count += 1
    image_path = xml_file.split('.')[0] + '.jpg'
    txt_file.write(image_path+' ')
    results = parse_rec(Annotations + xml_file)
    num_obj = len(results)
    txt_file.write(str(num_obj)+' ')
    for result in results:
        class_name = result['name']
        bbox = result['bbox']
        class_name = VOC_CLASSES.index(class_name)
        txt_file.write(str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+' '+str(class_name)+' ')
    txt_file.write('\n')
    #if count == 10:
    #    break
txt_file.close()