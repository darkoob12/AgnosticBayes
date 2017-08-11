def label_first(file_name):
    import csv
    from collections import deque
    total = []
    with open(file_name) as f:
        rd = csv.reader(f)
        for row in rd:
            d = deque(row)
            d.rotate(1)
            total.append(list(d))
    

    with open(file_name+'.csv','w',newline='') as f:
        wr = csv.writer(f)
        for ex in total:
            wr.writerow(ex)



if __name__ == '__main__':
    label_first('lymphoma_cancer')
##    print('hello, world!')
##    import numpy as np
##    import csv
##    from collections import deque
##    total = []
##    with open('breast_cancer_ts.csv') as f:
##        rd = csv.reader(f)
##        for row in rd:
##            d = deque(row)
##            d.rotate(1)
##            total.append(list(d))
##
##    with open('breast_cancer_tr.csv') as f:
##        rd = csv.reader(f)
##        for row in rd:
##            d = deque(row)
##            d.rotate(1)
##            total.append(list(d))
##    
##
##    with open('breast_cancer','w',newline='') as f:
##        wr = csv.writer(f)
##        for ex in total:
##            wr.writerow(ex)
