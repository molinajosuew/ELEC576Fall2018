def get_mice_and_cat_stats():
    raw_data = open('./data/stats_file.txt', 'r').readlines()
    data = []
    for i in range(0, len(raw_data)):
        data.append(list(map(int, raw_data[i].replace('\n', '')[:-1].split(','))))
    return data


def average_numbers(numbers):
    s = 0
    for number in numbers:
        s += number
    return s / len(numbers)


mice_and_cats = get_mice_and_cat_stats()

print('[', end = '')
for i in range(0, len(mice_and_cats), 2):
    for j in range(0, len(mice_and_cats[i])):
        print(mice_and_cats[i][j], end = '')
        if j + 1 < len(mice_and_cats[i]):
            print(',', end = '')
    if i + 2 < len(mice_and_cats):
        print(';', end = '')
print(']')

print('[', end = '')
for i in range(0, len(mice_and_cats), 2):
    print(average_numbers(mice_and_cats[i]), end = '')
    if i + 2 < len(mice_and_cats[i]):
        print(',', end = '')
print(']')