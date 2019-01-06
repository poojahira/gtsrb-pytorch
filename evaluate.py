from __future__ import print_function
import argparse
from tqdm import tqdm
import os
import PIL.Image as Image

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as datasets
import numpy as np
from data import initialize_data # data.py in the same folder
from model import Net
import torchvision


parser = argparse.ArgumentParser(description='PyTorch GTSRB evaluation script')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='pred.csv', metavar='D',
                    help="name of the output csv file")

args = parser.parse_args()

state_dict = torch.load(args.model)
model = Net()
model.load_state_dict(state_dict)
model.eval()

from data import data_jitter_hue,data_jitter_brightness,data_jitter_saturation,data_jitter_contrast,data_rotate,data_hvflip,data_shear,data_translate,data_center,data_grayscale


test_dir = args.data + '/test_images'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

transforms = [data_transforms,data_jitter_hue,data_jitter_brightness,data_jitter_saturation,data_jitter_contrast,data_rotate,data_hvflip,data_shear,data_translate,data_center]
output_file = open(args.outfile, "w")
output_file.write("Filename,ClassId\n")

for f in tqdm(os.listdir(test_dir)):
    if 'ppm' in f:
        output = torch.zeros([1, 43], dtype=torch.float32)
        with torch.no_grad():
            for i in range(0,len(transforms)):
                data = transforms[i](pil_loader(test_dir + '/' + f))
                data = data.view(1, data.size(0), data.size(1), data.size(2))
                data = Variable(data)
                output = output.add(model(data))
            pred = output.data.max(1, keepdim=True)[1]
            file_id = f[0:5]
            output_file.write("%s,%d\n" % (file_id, pred))

            

output_file.close()

print("Succesfully wrote " + args.outfile + ', you can upload this file to the kaggle '
      'competition at https://www.kaggle.com/c/nyu-cv-fall-2017/')
        


