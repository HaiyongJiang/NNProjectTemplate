#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : visual_tf.py
# Author            : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
# Date              : 25.05.2018
# Last Modified Date: 08.10.2018
# Last Modified By  : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
import os,sys
import argparse


def show_tensorboard(folder, idx=-1, bprint=False, prefix=""):
    files = [folder+fname for fname in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, fname)) \
            and fname.startswith(prefix)]
    if len(files) == 0:
        print("ERROR: no file found for prefix: " + prefix)
    else:
        print("%d files found."%(len(files)))
    print("files: ")
    print("\n".join([str(ii)+":"+f for ii,f in enumerate(sorted(files))]))
    if not bprint:
        file_last = sorted(files)[idx]
        cmd = "tensorboard --logdir=" + file_last
        #  cmd = "tensorboard --logdir=" + file_last
        print("commands: ")
        print(cmd)
        os.system(cmd)


def main():
    print(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", dest="folder", type=str, default="../outputs/meshes/",
            help="logging folder")
    parser.add_argument("-n", dest="split_name", type=str, default="train",
            choices=["train", "eval"], help="Different split name(train/eval)")
    parser.add_argument("-i", dest="index", type=int, default=-1,
            help="index of the log file/folder (furture)")
    parser.add_argument("-p", dest="bprint", action='store_true',
            help="index of the log file/folder (furture)")
    parser.add_argument("--prefix", type=str, default="", help="file prefix")
    args = parser.parse_args()

    show_tensorboard(args.folder, args.index, args.bprint, args.prefix)
    pass


if __name__ == "__main__":
    main()
