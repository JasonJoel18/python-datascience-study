"""Consider a list (list = []). You can perform the following commands:

    insert i e: Insert integer 

at position
.
print: Print the list.
remove e: Delete the first occurrence of integer
.
append e: Insert integer

    at the end of the list.
    sort: Sort the list.
    pop: Pop the last element from the list.
    reverse: Reverse the list.

Initialize your list and read in the value of
followed by lines of commands where each command will be of the

types listed above. Iterate through each command in order and perform the corresponding operation on your list.

Example





: Append to the list,
.
: Append to the list,
.
: Insert at index ,
.

    : Print the array.
    Output:

[1, 3, 2]

Input Format

The first line contains an integer,
, denoting the number of commands.
Each line of the

subsequent lines contains one of the commands described above.

Constraints

    The elements added to the list must be integers.

Output Format

For each command of type print, print the list on a new line.

Sample Input 0

12
insert 0 5
insert 1 10
insert 0 6
print
remove 6
append 9
append 1
sort
print
pop
reverse
print

Sample Output 0

[6, 5, 10]
[1, 5, 9, 10]
[9, 5, 1]

"""
if __name__ == '__main__':
    N = int(input())
    no_list = []

    def list_append(x):
        no_list.append(x)
        
    def list_insert(i,x):
        no_list.insert(i,x)
    
    def list_print():
        print(no_list)
    
    def list_remove(x):
        no_list.remove(x)
    
    def list_sort():
        no_list.sort()
    
    def list_pop():
        no_list.pop()
    
    def list_reverse():
        no_list.reverse()

    for _ in range(N):
        # my_str = input(f'Enter the query no {_+1}/{N}:')
        my_str = input()
        my_list = my_str.split()
        # print(my_list)
        
        if my_list[0] == 'insert':
            list_insert(int(my_list[1]),int(my_list[2]))
        elif my_list[0] == 'print':
            list_print()
        elif my_list[0] == 'remove':
            list_remove(int(my_list[1]))
        elif my_list[0] == 'append':
            list_append(int(my_list[1]))
        elif my_list[0] == 'sort':
            list_sort()
        elif my_list[0] == 'pop':
            list_pop()
        elif my_list[0] == 'reverse':
            list_reverse()
        else:
            print("invlid input")
            