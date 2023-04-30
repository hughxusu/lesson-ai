import random

balls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
boxes = [[], [], []]


for one in balls:
    index = random.randint(0, 2)
    box = boxes[index]
    box.append(one)

for row in boxes:
    for ball in row:
        print(ball, end='\t')
    print()