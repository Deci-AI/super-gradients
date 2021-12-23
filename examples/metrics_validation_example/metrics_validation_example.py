import xml.etree.ElementTree as ET
from tqdm import tqdm
from pathlib import Path
from zipfile import ZipFile
import torch
import os

NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def download(url, dir='.', unzip=True, delete=True, curl=False, threads=1):
    # Multi-threaded file download and unzip function, used in data.yaml for autodownload
    def download_one(url, dir):
        # Download 1 file
        f = dir / Path(url).name  # filename
        if Path(url).is_file():  # exists in current path
            Path(url).rename(f)  # move to dir
        elif not f.exists():
            print(f'Downloading {url} to {f}...')
            if curl:
                os.system(f"curl -L '{url}' -o '{f}' --retry 9 -C -")  # curl download, retry and resume on fail
            else:
                torch.hub.download_url_to_file(url, f, progress=True)  # torch download
        if unzip and f.suffix in ('.zip', '.gz'):
            print(f'Unzipping {f}...')
            if f.suffix == '.zip':
                ZipFile(f).extractall(path=dir)  # unzip
            elif f.suffix == '.gz':
                os.system(f'tar xfz {f} --directory {f.parent}')  # unzip
            if delete:
                f.unlink()  # remove zip

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    for u in [url] if isinstance(url, (str, Path)) else url:
        download_one(u, dir)

def convert_label(path, lb_path, year, image_id):
    def convert_box(size, box):
        dw, dh = 1. / size[0], 1. / size[1]
        x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
        return x * dw, y * dh, w * dw, h * dh

    in_file = open(path / f'VOC{year}/Annotations/{image_id}.xml')
    out_file = open(lb_path, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls in NAMES and not int(obj.find('difficult').text) == 1:
            xmlbox = obj.find('bndbox')
            bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
            cls_id = NAMES.index(cls)  # class id
            out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')


# Download
dir = "/data/pascal_unified" # dataset root dir
url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/'
urls = [url + 'VOCtrainval_06-Nov-2007.zip',  # 446MB, 5012 images
        url + 'VOCtest_06-Nov-2007.zip',  # 438MB, 4953 images
        url + 'VOCtrainval_11-May-2012.zip']  # 1.95GB, 17126 images
download(urls, dir=dir / 'images', delete=False)
# Convert
path = dir / f'images/VOCdevkit'
for year, image_set in ('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test'):
    imgs_path = dir / 'images' / f'{image_set}{year}'
    lbs_path = dir / 'labels' / f'{image_set}{year}'
    imgs_path.mkdir(exist_ok=True, parents=True)
    lbs_path.mkdir(exist_ok=True, parents=True)
    image_ids = open(path / f'VOC{year}/ImageSets/Main/{image_set}.txt').read().strip().split()
    for id in tqdm(image_ids, desc=f'{image_set}{year}'):
        f = path / f'VOC{year}/JPEGImages/{id}.jpg'  # old img path
        lb_path = (lbs_path / f.name).with_suffix('.txt')  # new label path
        f.rename(imgs_path / f.name)  # move image
        convert_label(path, lb_path, year, id)  # convert labels to YOLO format