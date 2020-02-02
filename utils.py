from glob import glob
import cv2


def get_imgs(path):
    """
    - returns  a dictionary of all files
    having key => value as  objectname => image path

    - returns total number of files.

    """
    imlist = {}
    for each in glob(path + "*"):
        word = each.split("/")[-1]
        imlist[word] = []
        for imagefile in glob(path+word+"/*"):
            im = cv2.imread(imagefile, 0)
            imlist[word].append(im)

    return imlist
