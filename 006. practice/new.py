if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    
    marks_list = student_marks.get('Malika')
    average_marks = float(sum(marks_list))/(len(marks_list))
    print("{:.2f}".format(round(average_marks, 2)))