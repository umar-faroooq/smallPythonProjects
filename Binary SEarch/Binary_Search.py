def binary_search(list, element):
    list.sort()
    middle = 0
    start = 0
    end = len(list)
    steps = 0

    while(start<=end):
        print("Step: ", steps, ":", str(list[start:end+1]))

        steps += 1

        middle = (start + end) // 2

        if element == list[middle]:
            return middle
        if element < list[middle]:
            end = middle-1
        else:
            start = middle + 1
    return -1


my_list = [6, 7, 1,33,45,6,6]
target=2

binary_search(my_list, target)




