import csv
p = './train/tr_img_01165.txt'

with open(p, 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        label = line[-1]
        # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
        line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]

        x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
        print(x1, y1, x2, y2, x3, y3, x4, y4)