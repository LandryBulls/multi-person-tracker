import os
import cv2
import subprocess
import os.path as osp
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

def video_to_images(vid_file, img_folder=None, return_info=False):
    if img_folder is None:
        img_folder = osp.join('/tmp', osp.basename(vid_file).replace('.', '_'))

    os.makedirs(img_folder, exist_ok=True)

    command = ['ffmpeg',
               '-i', vid_file,
               '-f', 'image2',
               '-v', 'error',
               f'{img_folder}/%06d.png']
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

    print(f'Images saved to \"{img_folder}\"')

    img_shape = cv2.imread(osp.join(img_folder, '000001.png')).shape

    if return_info:
        return img_folder, len(os.listdir(img_folder)), img_shape
    else:
        return img_folder


def images_to_video(img_folder, output_vid_file):
    os.makedirs(img_folder, exist_ok=True)

    command = [
        'ffmpeg', '-y', '-threads', '16', '-i', f'{img_folder}/%06d.png', '-profile:v', 'baseline',
        '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error', output_vid_file,
    ]

    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)


def frame2array(frame_no, video_opened):
    # returns a video from as a numpy array in uint8
    video_opened.set(cv2.CAP_PROP_POS_FRAMES,frame_no)
    ret, frame = video_opened.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #cv2.destroyAllWindows()
    return frame

class ImageFolder(Dataset):
    def __init__(self, image_folder):
        self.image_file_names = [
            osp.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')
        ]
        self.image_file_names = sorted(self.image_file_names)

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_file_names[idx]), cv2.COLOR_BGR2RGB)
        return to_tensor(img)

class VideoFile(Dataset):
    def __init__(self, video_path):
        self.video_cv2 = cv2.VideoCapture(video_path)

    def __len__(self):
        return int(self.video_cv2.get(cv2.CAP_PROP_FRAME_COUNT))

    def __getitem__(self, idx):
        img = frame2array(idx, self.video_cv2)
        return to_tensor(img)