

def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    '''
    解析yolo-v3层配置文件并返回模块定义
    返回结果 为  每部分写为一行
    path： yolov3.cfg的路径
    '''
    file = open(path, 'r')
    # 按行读取，存为list
    lines = file.read().split('\n')
    # 过滤掉 "#"开头的内容，即注释信息
    lines = [x for x in lines if x and not x.startswith('#')]
    # lstrip去掉左边的(头部)，rstrip去掉右边的(尾部)  默认删除字符串头和尾的空白字符(包括\n，\r，\t这些)
    lines = [x.rstrip().lstrip() for x in lines] # 去除边缘空白，即去掉左右两侧的空格等字符
    module_defs = []
    for line in lines:
        # 检查字符串是否是以指定子字符串 [ 开头，如果是则返回 True，否则返回 False
        if line.startswith('['): # This marks the start of a new block  标志着一个新区块的开始
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

def parse_data_config(path):
    """Parses the dataloader configuration file"""
    '''
    解析dataloader配置文件
    '''
    options = dict()
    # 默认GPU有4个
    options['gpus'] = '0,1,2,3'
    # 数据集加载器加载数据时使用线程数
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options
