yearly_salary = [72000, 48000, 54000, 61000, 1000, 58000, 52000, 79000, 83000, 67000]

x_min = min(yearly_salary)
x_max = max(yearly_salary)

normalized_salary = []
for x_i in yearly_salary:
    x_new = (x_i - x_min) / (x_max - x_min)
    normalized_salary.append(x_new)


for every in normalized_salary:
    print(every)

##################################################


# import matplotlib.pyplot as plt
# from math import sqrt

# data_x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
# data_y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
# data_label = [True, True, False, True, False, False, False, False, True, False, False, False, False]

# color_list = []
# for label in data_label:
#     if label == True:
#         color_list.append("blue")
#     else:
#         color_list.append("red")

# new_data = (10, 92)

# distances = []
# for x, y in zip(data_x, data_y):
#     distance = sqrt((new_data[0] - x) ** 2 + (new_data[1] - y) ** 2)
#     distances.append(distance)

# zipped_dataset = zip(distances, data_x, data_y, data_label, color_list)
# sorted_dataset = sorted(zipped_dataset)
# print(sorted_dataset)

# k = 1
# votes = []
# for data in sorted_dataset[:k]:
#     if data[3] == True:
#         votes.append(True)
#     else:
#         votes.append(False)
# print(votes)

# count_true = 0
# count_false = 0
# for vote in votes:
#     if vote == True:
#         count_true = count_true + 1
#     else:
#         count_false = count_false + 1

# print(count_true)
# print(count_false)

# if count_true > count_false:
#     color = "cyan"
# elif count_true == count_false:
#     print("your voting ended up with equality, democracy loses, peace wins")
#     color = "purple"
# else:
#     color = "orange"

# plt.axes().set_aspect('equal', 'datalim')
# plt.scatter(data_x, data_y, c = color_list)
# plt.scatter(new_data[0], new_data[1], c = color)
# plt.show()