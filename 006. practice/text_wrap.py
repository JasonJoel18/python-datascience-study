"""
You are given a stringS and width w.
Your task is to wrap the string into a paragraph of width w.
Function Description
Complete the wrap function in the editor below.
wrap has the following parameters:
• string string: a long string
• int max_width: the width to wrap to
Returns
• string: a single string with newline characters (\\In') where the breaks should be

Input Format
The first line contains a string, string.
The second line contains the width, max width.

Constraints
• 0 < len (string) < 1000
• O < max width < len(string)

Sample Input O
ABCDEFGHIJKLIMNOQRSTUVWXYZ
4
"""
import textwrap

def wrap(string, max_width):
    wrapped_lines = textwrap.wrap(string, max_width)
    return "\n".join(wrapped_lines)

if __name__ == '__main__':
    # string, max_width = input(), int(input())
    string = "ABCDEFGHIJKLIMNOQRSTUVWXYZ" 
    max_width = 4 
    result = wrap(string, max_width)
    print(result)