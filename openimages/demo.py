import chainer
import shufflenet_v2
import chainertools
import cv2
import numpy as np
import time


def main(args):
    with chainer.using_config('train', False):
        with chainer.using_config('enable_backprop', False):
            snapshot_file = "shufflenet-v2-snapshots/x1/snapshot_iter_335305"
            label_encoder = chainertools.openimages.openimages_label_encoder(
                ".")
            net = shufflenet_v2.ShuffleNetV2(1, label_encoder.num_classes())
            chainer.serializers.load_npz(
                snapshot_file, net, "updater/model:main/predictor/")
            if args.gpu >= 0:
                net.to_gpu(args.gpu)

            camera_id = -1
            camera = cv2.VideoCapture(camera_id)
            dt_filtered = 0.
            alpha = 0.1
            while True:
                success, frame = camera.read()
                if not success:
                    raise RuntimeError("could not read frame from camera")

                t0 = time.time()
                frame_small_orig = cv2.resize(frame, (224, 224))
                frame_small = cv2.cvtColor(frame_small_orig, cv2.COLOR_BGR2RGB)
                frame_small = np.transpose(frame_small, (2, 0, 1))
                input = net.xp.asarray([frame_small], dtype=np.float32)
                # print(input.shape, input)
                output = net(input)
                output = chainer.functions.sigmoid(output)
                output = chainer.cuda.to_cpu(output.data[0])
                t1 = time.time()

                labels_idx = np.where(output > 0.5)[0]
                readable_labels = [label_encoder.readable_label_of_encoded_label(
                    lab) for lab in labels_idx]

                dt = t1 - t0
                dt_filtered = alpha * dt + (1 - alpha) * dt_filtered
                fps = 1. / dt_filtered
                print("{:.2f} fps".format(fps))
                print(list(zip(readable_labels, output[labels_idx])))
                cv2.imshow("x1", frame)
                cv2.waitKey(1)


def parse_command_line():
    # TODO: add command line args for snapshot
    parser = argparse.ArgumentParser(
        description="Demonstration of multilabel classification with Shufflenet v2.")
    parser.add_argument('--gpu', help='Run on gpu (integer id starting at 0) or cpu (-1)', type=int, default=-1)
    return parser.parse_args()

if __name__ == '__main__':
    import argparse
    args = parse_command_line()
    main(args)
