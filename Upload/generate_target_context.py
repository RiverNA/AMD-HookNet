import glob
import os
import PIL
import numpy as np3
from config import cfg
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

ti = './Sliding_window/Train/Transform_images'
mi = './Sliding_window/Train/Transform_masks'

images = sorted(glob.glob(os.path.join(ti, '*.png')))
masks = sorted(glob.glob(os.path.join(mi, '*.png')))

save_path_target = './Sliding_window/Train/Training/target_images'
save_path_context = './Sliding_window/Train/Training/context_images'
save_path_target_mask = './Sliding_window/Train/Training/target_masks'
save_path_context_mask = './Sliding_window/Train/Training/context_masks'

size = 288

target = transforms.Compose([
    transforms.CenterCrop((size, size)),
    transforms.ToTensor(),
])

context_images = transforms.Compose([
    transforms.CenterCrop((size * 2, size * 2)),
    transforms.Resize((size, size), interpolation=PIL.Image.BICUBIC),
    transforms.ToTensor(),
])

context_masks = transforms.Compose([
    transforms.CenterCrop((size * 2, size * 2)),
    transforms.Resize((size, size), interpolation=PIL.Image.NEAREST),
    transforms.ToTensor(),
])

if not os.path.exists(save_path_target):
    os.makedirs(save_path_target)
if not os.path.exists(save_path_context):
    os.makedirs(save_path_context)
if not os.path.exists(save_path_target_mask):
    os.makedirs(save_path_target_mask)
if not os.path.exists(save_path_context_mask):
    os.makedirs(save_path_context_mask)

for i in range(len(images)):
    image = Image.open(images[i])
    suffix = images[i].split('/')[-1].split('.')[0]

    target_image = target(image)
    context_image = context_images(image)
    save_image(target_image, os.path.join(save_path_target, suffix + '.png'))
    save_image(context_image, os.path.join(save_path_context, suffix + '.png'))

for i in range(len(masks)):
    mask = Image.open(masks[i])
    suffix = masks[i].split('/')[-1].split('.')[0]

    target_mask = target(mask)
    context_mask = context_masks(mask)
    save_image(target_mask, os.path.join(save_path_target_mask, suffix + '.png'))
    save_image(context_mask, os.path.join(save_path_context_mask, suffix + '.png'))

ti = './Sliding_window/Valid/Transform_images'
mi = './Sliding_window/Valid/Transform_masks'

images = sorted(glob.glob(os.path.join(ti, '*.png')))
masks = sorted(glob.glob(os.path.join(mi, '*.png')))

save_path_target = './Sliding_window/Valid/Validation/target_images'
save_path_context = './Sliding_window/Valid/Validation/context_images'
save_path_target_mask = './Sliding_window/Valid/Validation/target_masks'
save_path_context_mask = './Sliding_window/Valid/Validation/context_masks'

target = transforms.Compose([
    transforms.CenterCrop((size, size)),
    transforms.ToTensor(),
])

context_images = transforms.Compose([
    transforms.CenterCrop((size * 2, size * 2)),
    transforms.Resize((size, size), interpolation=PIL.Image.BICUBIC),
    transforms.ToTensor(),
])

context_masks = transforms.Compose([
    transforms.CenterCrop((size * 2, size * 2)),
    transforms.Resize((size, size), interpolation=PIL.Image.NEAREST),
    transforms.ToTensor(),
])

if not os.path.exists(save_path_target):
    os.makedirs(save_path_target)
if not os.path.exists(save_path_context):
    os.makedirs(save_path_context)
if not os.path.exists(save_path_target_mask):
    os.makedirs(save_path_target_mask)
if not os.path.exists(save_path_context_mask):
    os.makedirs(save_path_context_mask)

for i in range(len(images)):
    image = Image.open(images[i])
    suffix = images[i].split('/')[-1].split('.')[0]

    target_image = target(image)
    context_image = context_images(image)
    save_image(target_image, os.path.join(save_path_target, suffix + '.png'))
    save_image(context_image, os.path.join(save_path_context, suffix + '.png'))

for i in range(len(masks)):
    mask = Image.open(masks[i])
    suffix = masks[i].split('/')[-1].split('.')[0]

    target_mask = target(mask)
    context_mask = context_masks(mask)
    save_image(target_mask, os.path.join(save_path_target_mask, suffix + '.png'))
    save_image(context_mask, os.path.join(save_path_context_mask, suffix + '.png'))

ti = './Sliding_window/Test/Transform_images'
mi = './Sliding_window/Test/Transform_masks'

images = sorted(glob.glob(os.path.join(ti, '*.png')))
masks = sorted(glob.glob(os.path.join(mi, '*.png')))

save_path_target = './Sliding_window/Test/Testing/target_images'
save_path_context = './Sliding_window/Test/Testing/context_images'
save_path_target_mask = './Sliding_window/Test/Testing/target_masks'
save_path_context_mask = './Sliding_window/Test/Testing/context_masks'

target = transforms.Compose([
    transforms.CenterCrop((size, size)),
    transforms.ToTensor(),
])

context_images = transforms.Compose([
    transforms.CenterCrop((size * 2, size * 2)),
    transforms.Resize((size, size), interpolation=PIL.Image.BICUBIC),
    transforms.ToTensor(),
])

context_masks = transforms.Compose([
    transforms.CenterCrop((size * 2, size * 2)),
    transforms.Resize((size, size), interpolation=PIL.Image.NEAREST),
    transforms.ToTensor(),
])

if not os.path.exists(save_path_target):
    os.makedirs(save_path_target)
if not os.path.exists(save_path_context):
    os.makedirs(save_path_context)
if not os.path.exists(save_path_target_mask):
    os.makedirs(save_path_target_mask)
if not os.path.exists(save_path_context_mask):
    os.makedirs(save_path_context_mask)

for i in range(len(images)):
    image = Image.open(images[i])
    suffix = images[i].split('/')[-1].split('.')[0]

    target_image = target(image)
    context_image = context_images(image)
    save_image(target_image, os.path.join(save_path_target, suffix + '.png'))
    save_image(context_image, os.path.join(save_path_context, suffix + '.png'))

for i in range(len(masks)):
    mask = Image.open(masks[i])
    suffix = masks[i].split('/')[-1].split('.')[0]

    target_mask = target(mask)
    context_mask = context_masks(mask)
    save_image(target_mask, os.path.join(save_path_target_mask, suffix + '.png'))
    save_image(context_mask, os.path.join(save_path_context_mask, suffix + '.png'))
