import cv2
import os
import argparse

from cv2 import data


def video_to_frames(video, output):
    # create output dir if it does not exist
    if not os.path.exists(output):
        os.mkdir(output)

    print('Video: ', video)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Total frame: {total_frames} | fps: {fps}')
    print('Extracting frame ...', end='')
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            count += 1
            video = video.replace('+', '_')
            filename = os.path.join(output, data_path.rsplit('/', 1)[-1] + '_' + video.split(
                '/')[-1].split('.')[0] + '_{}'.format(count) + '.jpg')
            cv2.imwrite(filename, frame)
        else:
            cap.release()
            break

    assert total_frames == count, 'Frames not extracted properly'
    print(f'Done | {count} frames extracted')
    print('Saved frames are in ', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_paths', help='DryEye Video path', nargs='+',
                        default=['./dataset/Gold_Standard', './dataset/Gold_Standard_II'])
    parser.add_argument('--output', help='output directory',
                        default='./dataset/images')

    args = parser.parse_args()
    data_paths = args.data_paths
    output = args.output

    for data_path in data_paths:
        if os.path.isdir(data_path):
            for i, filename in enumerate(os.listdir(data_path)):
                print(f'\n### {i+1}/{len(os.listdir(data_path))}')
                video_to_frames(os.path.join(data_path, filename), output)
        else:  # is a video file
            video_to_frames(data_path, output)
