from math import log10, floor

## Selection Sort
def select_sort(x):
    for i in range(len(x)):
        mini = x[i]
        mini_ix = i
        for j in range(i, len(x)):
            if x[j] < mini:
                mini = x[j]
                mini_ix = j
        x[mini_ix], x[i] = x[i], x[mini_ix] 
    return x

## Insertion Sort
def insert_sort(x):
    for i in range(1, len(x)):
        for j in range(i, 0, -1):
            if x[j] < x[j-1]:
                x[j], x[j-1] = x[j-1], x[j]
            else:
                break
    return x

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

## Radix sort

def radix_sort(x):

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

        return output

    maxi = max(x)

    if maxi > 0:
        maxi = floor(log10(maxi) + 1)
    else:
        maxi = 1

    for radix in range(0, maxi):
        keys = [(element // 10**radix % 10) for element in x]
        x = counting_sort(x, keys)
    
    return x


if __name__ == "__main__":
    a = [7, 2, 9, 8, 9, 4, 4, 4, 1, 11, 1, 0, 10, 3, 5, 6]
    b = [i+.03 for i in a]
    c = [4, 2, 3, 1]
    d = [1, 3, 1]
    e = [7, 2, 9, 8, 9, 4, 4, 4, 1, 1, 0, 3, 5, 6]
    print(radix_sort(a))