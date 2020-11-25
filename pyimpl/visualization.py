import random
import numpy as np
from skimage import color, io
from imageio import imwrite
import cv2
import subprocess
import glob
import os
import argparse
from math import log10, floor
import struct

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
    swaps = list()

    def _merge_sort(x, offset=0):
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

        left = _merge_sort(left, offset)
        right = _merge_sort(right, offset+len(right))

        merged = merge(left, right)
        
        swaps.append([offset, merged.copy()])

        return merged

    return _merge_sort(x), swaps

## Radix Sort - Float version

def radix_sort(x):

    swaps = list()

    ## Floats

    ## https://stackoverflow.com/a/16445458
    def float2int(value):
        return sum(b << 8*i for i,b in enumerate(struct.pack('f', value)))

    keyvalue_mapping = dict()
    floats = list()


    for i in range(len(x)):
        floats.append(float2int(float(x[i])))
        keyvalue_mapping[floats[i]] = x[i]

    def counting_sort(values, keys): # Assumes keys are integers and keys[i] is the key for the value values[i]
        output = [0] * len(values)
        freq = [0] * (max(keys) + 1)

        for element in keys:
            freq[element] += 1
        
        for i in range(1,len(freq)):
            freq[i] += freq[i-1]
        

        for i in range(len(values)-1, -1, -1):
            output[freq[keys[i]]-1] = values[i] #-1 bcs arrays start at 0
            freq[keys[i]] -= 1

        swaps.append([0, [keyvalue_mapping[i] for i in values]])
        return output

    maxi = max(floats)

    if maxi > 0:
        maxi = floor(log10(maxi) + 1)
    else:
        maxi = 1

    for radix in range(0, maxi):
        keys = [(element // 10**radix % 10) for element in floats]
        floats = counting_sort(floats, keys)
    
    output = list()
    for el in floats:
        output.append(keyvalue_mapping[el])
        
    return output, swaps

## Heap Sort

def heap_sort(x):

    swaps = list()
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
            swaps.append([i, parent_idx])
            heapify(A, parent_idx, limit)

    def build_heap(x):
        for i in range(len(x)//2 - 1, -1, -1):
            heapify(x, i, len(x))
        return x

    heap = build_heap(x)    

    for i in range(len(x) - 1, 0, -1):
        heapify(heap, 0, i+1)
        heap[i], heap[0] = heap[0], heap[i]
        swaps.append([i, 0])
    return heap,swaps

# Don't want no arbitrary code execution here!
function_mappings = {
    "Selection" : select_sort,
    "Insertion" : insert_sort,
    "Merge" : merge_sort,
    "Heap" : heap_sort,
    "Radix" : radix_sort
}

type_mappings = {

    "select_sort" : "swap",
    "insert_sort" : "swap",
    "merge_sort" : "replace",
    "heap_sort" : "swap",
    "radix_sort" : "replace",
}

def make_swap(img, row, move):
    placeholder = img[row,move[0],:].copy()
    img[row, move[0],:] = img[row, move[1],:]
    img[row, move[1],:] = placeholder
    return img

def make_replace(img, row, move):
    for i in range(len(move[1])):
        img[row, i+move[0],0] = move[1][i]
    return img

def visualize(sort, SIZE, FRAMERATE, VIDEOLENGTH, OUTPUT=""):

    sort_algorithm = sort.__name__
    if OUTPUT == "":
        OUTPUT = f"{sort_algorithm}-visualized.mp4"

    if OUTPUT[-4:] != ".mp4":
        OUTPUT += ".mp4"

    COLUMN = SIZE # X-Axis
    ROW = SIZE # Y-Axis, needs to be <=Column for proper effect.
    TYPE = type_mappings[sort_algorithm]

    print("Generating video with following settings:")
    print(f"Size: {ROW}x{COLUMN}\nFramerate: {FRAMERATE}\nVideolength: {VIDEOLENGTH}\nAlgorithm: {sort_algorithm}\nOutput file: {OUTPUT}\nSorting type: {TYPE}\n")

    ## Init Shuffled Image
    print("Setting up shuffled image..")

    img = np.zeros((ROW, COLUMN, 3), dtype='float32') # Initialize array SIZExSIZE where each element is a pixel (RGB).

    for i in range(COLUMN):
        img[:,i] = i / COLUMN, 1.0, 1.0 # Change every innermost element (Pixel), creating a rainbow effect if thought of as (HSV) pixels (HSV = [0,1], RGB = [0,255]) HSV = Hue, Saturation, Value(Brightness). Changes Hue (Color) Only. 

    # RGB_Rainbow = (color.convert_colorspace(img, 'HSV', 'RGB'))
    # cv2.imshow("Generated Rainbow", RGB_Rainbow)
    # cv2.waitKey()

    for i in range(ROW):
        np.random.shuffle(img[i]) # Shuffles pixels around (Only in the same row)

    ## --
    
    print("Sorting image..")

    moves = list()
    maxMoves = 0

    for i in range(ROW):
        _, newMoves = sort(list(img[i,:,0])) # Sorts each row and saves the work done.
        moves.append(newMoves)
        if len(newMoves) > maxMoves:
            maxMoves = len(newMoves)


    # x second movie with y frames, x*y images.
    image_step_length = maxMoves // (FRAMERATE * VIDEOLENGTH)
    assert image_step_length > 0, f"Requires {(FRAMERATE * VIDEOLENGTH)} frames, only {maxMoves} frames available!"
    image_current_step = 0

    print("Recreating sort and saving imageframes...")

    # Algorithm sorts each row individually, we simulate but one move/row 
    for i in range(maxMoves): # For each move
        for j in range(ROW): # Per Row
            if i < len(moves[j]): # If that move was done that row
                if TYPE == "swap":
                    img = make_swap(img, j, moves[j][i]) # Make it
                elif TYPE == "replace":
                    img = make_replace(img, j, moves[j][i])

        if i % image_step_length == 0:
            name = f"{sort_algorithm}-{image_current_step:05}.png"
            in_rgb = (color.convert_colorspace(img, 'HSV', 'RGB')*255)
            cv2.imwrite(name, in_rgb)
            image_current_step += 1
    # RGB_Rainbow = (color.convert_colorspace(img, 'HSV', 'RGB'))
    # cv2.imshow("Generated Rainbow", RGB_Rainbow)
    # cv2.waitKey()
    # exit()
    
    ## Always capture last frame
    name = f"{sort_algorithm}-{image_current_step:05}.png"
    in_rgb = (color.convert_colorspace(img, 'HSV', 'RGB'))
    cv2.imwrite(name, in_rgb)
    # cv2.imshow("last frame", in_rgb)
    # cv2.waitKey()

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
                
        
    subprocess.run(["ffmpeg", "-framerate", str(FRAMERATE), "-i", f"{sort_algorithm}-%05d.png", f"{OUTPUT}"], capture_output=True)

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
    parser.add_argument("SORT", action="store", help="The sorting algorithm to visualize. Currently available are: Insertion, Selection, Merge, Heap, Radix")
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
    visualize(sortingmethod, results.SIZE, results.FRAMERATE, results.VIDEOLENGTH, results.OUTPUT)

