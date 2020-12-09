# oskhen-sorting
A collection of sorting algorithms

## Python

### Tests
Navigate to the `pyimpl` folder and either run the test with `pytest` or simply run the `sort_test.py` file.

### Visualization
For visualization, run `visualization.py -h` for directions.

To install all requirements, run ```python3 -m pip install -r requirements.txt``` or similar command. 
Requires `ffmpeg` as a command-line utilty, which most likely means that it doesn't support windows (at least not natively)

### Timing
```
Function <function heap_sort at 0x7f322d915550> ran in 0.6996315710002818 seconds with n = 100000
Function <function insert_sort at 0x7f322d915430> ran in 534.652702931 seconds with n = 100000
Function <function merge_sort at 0x7f322d9154c0> ran in 1.9888448769997922 seconds with n = 100000
Function <function radix_sort_float at 0x7f322d915670> ran in 0.8535405550001087 seconds with n = 100000
Function <function radix_sort_int at 0x7f322d9155e0> ran in 0.2957499089998237 seconds with n = 100000
Function <function select_sort at 0x7f322d9151f0> ran in 295.99610172099983 seconds with n = 100000
```