import random
import numpy as np
from skimage import color, io
from imageio import imwrite
import cv2
import subprocess
import glob
import os
import argparse

## Selection Sort 
def select_sort(x):
    swaps = list()
    for i in range(len(x)):
        mini = x[i]
        mini_ix = i
        for j in range(i, len(x)):
            if x[j] < mini:
                mini = x[j]
                mini_ix = j
        x[mini_ix], x[i] = x[i], x[mini_ix] 
        swaps.append([mini_ix, i])
    return x, swaps

## Insertion Sort
def insert_sort(x):
    swaps = list()
    for i in range(1, len(x)):
        for j in range(i, 0, -1):
            if x[j] < x[j-1]:
                x[j], x[j-1] = x[j-1], x[j]
                swaps.append([j, j-1])
            else:
                break
    return x, swaps

## Merge Sort

def merge_sort(x):

    def merge(a, b):
        output = list()
        while len(a) != 0 and len(b) != 0:
            if a[0] < b[0]:
                output.append(a[0])
                a.pop(0)
            else:
                output.append(b[0])
                b.pop(0)
        
        for x in a:
            output.append(x)
        for x in b:
            output.append(x)

        return output
        
    if len(x) <= 1:
        return x
    
    left = list()
    right = list()
    for i, element in enumerate(x):
        if i < (len(x)/2):
            left.append(element)
        else:
            right.append(element)

    left = merge_sort(left)
    right = merge_sort(right)

    return merge(left, right)

## Counting Sort

def count_sort(x):
    maxi = max(x)
    count = [0]*(maxi+1) #Arrays start at 0
    for val in x:
        count[val] += 1
    output = list()
    for index, element in enumerate(count):
        output += [index]*element
    return output

## Heap Sort

def heap_sort(x):

    def heapify(A, i, limit):
        left_idx = 2*i + 1
        right_idx = 2*i + 2
        parent_idx = i

        if left_idx < limit and A[left_idx] > A[parent_idx]:
            parent_idx = left_idx

        if right_idx < limit and A[right_idx] > A[parent_idx]:
            parent_idx = right_idx
        
        if parent_idx != i:
            A[i], A[parent_idx] = A[parent_idx], A[i]
            heapify(A, parent_idx, limit)

    def build_heap(x):
        for i in range(len(x)//2 - 1, -1, -1):
            heapify(x, i, len(x))
        return x

    heap = build_heap(x)    

    for i in range(len(x) - 1, 0, -1):
        heapify(heap, 0, i+1)
        heap[i], heap[0] = heap[0], heap[i]
    return(heap)

# Don't want no arbitrary code execution here!
function_mappings = {
    "Selection" : select_sort,
    "Insertion" : insert_sort,
}


def make_swap(img, row, move):
    placeholder = img[row,move[0],:].copy()
    img[row, move[0],:] = img[row, move[1],:]
    img[row, move[1],:] = placeholder
    return img

def visualize(sort, SIZE, FRAMERATE, VIDEOLENGTH, OUTPUT=""):

    sort_algorithm = sort.__name__
    if OUTPUT == "":
        OUTPUT = f"{sort_algorithm}-visualized.mp4"

    print("Generating video with following settings:")
    print(f"Size: {SIZE}\nFramerate: {FRAMERATE}\nVideolength: {VIDEOLENGTH}\nAlgorithm: {sort_algorithm}\nOutput file: {OUTPUT}\n")

    print("Setting up shuffled image..")
    ## Init Shuffled Image
    

    img = np.zeros((SIZE, SIZE, 3), dtype='float32')

    for i in range(img.shape[1]):
        img[:,i,:] = i / img.shape[1], 1.0, 1.0

    for i in range(img.shape[0]):
        np.random.shuffle(img[i,:,:])

    ## --

    print("Sorting image..")

    moves = list()
    maxMoves = 0

    for i in range(img.shape[0]):
        _, newMoves = sort(list(img[i,:,0]))
        moves.append(newMoves)
        if len(newMoves) > maxMoves:
            maxMoves = len(newMoves)


    # x second movie with y frames, x*y images.
    image_step_length = maxMoves // (FRAMERATE * VIDEOLENGTH)
    image_current_step = 0

    print("Recreating sort and saving imageframes...")

    for i in range(maxMoves):
        for j in range(img.shape[0]):
            if i < len(moves[j])-1:
                img = make_swap(img, j, moves[j][i])

        if i % image_step_length == 0:
            name = f"{sort_algorithm}-{image_current_step:05}.png"
            in_rgb = (color.convert_colorspace(img, 'HSV', 'RGB')*255)
            cv2.imwrite(name, in_rgb)
            image_current_step += 1

    print("Creating video...")
    if os.path.isfile(OUTPUT):
        while True:
            choice = input(f"File with name {OUTPUT} already exists. Do you want to proceed and overwrite previous file? [y/N]: ").capitalize()
            if choice == "Y":
                os.remove(f"{OUTPUT}")
                print("Overwriting..")
                break
            elif choice == "N" or choice == "":
                print("Aborting...")
                files = glob.glob(f"{sort_algorithm}-*.png")
                for png in files:
                    os.remove(png)
                exit()
                
        
    subprocess.run(["ffmpeg", "-framerate", str(FRAMERATE), "-i", f"{sort_algorithm}-%05d.png", OUTPUT], capture_output=True)

    print("Cleaning up frames..")
    files = glob.glob(f"{sort_algorithm}-*.png")
    for png in files:
        os.remove(png)
    print("Done!")

def init_parser():
    parser = argparse.ArgumentParser(description="Visualize various sorting algorithms")

    parser.add_argument("-s", "--size", default=300, action="store", type=int, dest="SIZE", help="The size of the output video in pixels, frames made up of NxN pixels. Defaults to 300x300")
    parser.add_argument("-f", "--framerate", default=24, action="store", type=int, dest="FRAMERATE", help="The framerate of the video, amount of frames per second. Defaults to 24")
    parser.add_argument("-l", "--length", default=5, action="store", type=int, dest="VIDEOLENGTH", help="The length of the output video in seconds. Defaults to 5.")
    parser.add_argument("SORT", action="store", help="The sorting algorithm to visualize. Currently available are: Insertion, Selection")
    parser.add_argument("-o", "--output", default="", action="store", type=str, dest="OUTPUT", help="Name of output video file. Defaults to {sort_algorithm}-visualized.mp4")

    return parser

if __name__ == "__main__":  
    parser = init_parser()
    results = parser.parse_args()
    try:
        sortingmethod = function_mappings[results.SORT]
    except:
        print("See -h for usage.")
        exit()
    visualize(sortingmethod, results.SIZE, results.FRAMERATE, results.VIDEOLENGTH)

