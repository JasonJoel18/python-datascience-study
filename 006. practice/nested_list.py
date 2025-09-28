if __name__ == '__main__':
    student_list = []
    unique_marks = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        student_marks = [name,score]
        student_list.append(student_marks)       
        
        if score not in unique_marks:
            unique_marks.append(score)
        
    unique_marks.sort(reverse=True)
    run_off_score = unique_marks[1]
    print(run_off_score)
    run_off_list = []
        
    c = int(1)
    for x in student_list:

        print(c)
        if x[1] == run_off_score:
            student_name = x[0]
            run_off_list.append(student_name)
        c=c+1
            
    print(run_off_list)
    print(student_list)
    print(unique_marks) 