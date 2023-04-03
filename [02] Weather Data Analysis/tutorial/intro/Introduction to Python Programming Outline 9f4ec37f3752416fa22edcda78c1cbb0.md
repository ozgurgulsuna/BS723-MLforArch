# Introduction to Python Programming Outline

## Before you start:

<aside>
💡 Please check the best practices whenever you don't feel confident about how to code:

</aside>

[PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

<aside>
💡 Don't stick only to what is available below, use various resources to understand a concept:

</aside>

[Python Variables](https://www.w3schools.com/python/python_variables.asp)

## Fundamentals:

1. **Syntax**
    - **Variables** `int` `float` `str`
        
        ```python
        an_integer = 5
        a_float = 3.2
        a_string = "hello machine"
        a_boolean = True
        ```
        
    - **Operators** `+` `-` `*` `/` `%` `**`
        
        ```python
        result = 10 + 30
        result = 40 - 10
        result = 50 * 5
        result = 16 / 4
        result = 25 % 2
        result = 5 ** 3
        ```
        
        ```python
        a_summation = 5 + 3.2
        a_subtraction = an_integer - 3.2
        a_multiplication = a_subtraction * 5
        a_division = a_subtraction / a_summation
        a_modulo = 12 % 5
        print(a_modulo)
        an_exponent = a_modulo ** an_integer
        ```
        
    - **Comments** `#` `"""`
        
        ```python
        # This is a comment, it is only for humans to read, machines have no power here
        now_machines_have_control = True
        
        """
        This is another way of commenting
        that allows you to comment out
        multiple lines
        """
        ```
        
2. **Control Flow**
    - **Comparison `==` `!=`** `<` `>`
        
        ```python
        a = 2
        b = 3
        a < b  # evaluates to True
        a > b  # evaluates to False
        a >= b # evaluates to False
        a <= b # evaluates to True
        a <= a # evaluates to True
        a == 2 # evaluates to True
        a != 3 # evaluates to True
        ```
        
    - **Logic** `and` `or` `not`
        
        ```python
        True and True     # Evaluates to True
        True and False    # Evaluates to False
        False and False   # Evaluates to False
        1 == 1 and 1 < 2  # Evaluates to True
        1 < 2 and 3 < 1   # Evaluates to False
        "Yes" and 100     # Evaluates to 100
        "Yes" and 0       # Evaluates to 0
        
        True or True      # Evaluates to True
        True or False     # Evaluates to True
        False or False    # Evaluates to False
        1 < 2 or 3 < 1    # Evaluates to True
        3 < 1 or 1 > 6    # Evaluates to False
        1 == 1 or 1 < 2   # Evaluates to True
        
        not True     # Evaluates to False
        not False    # Evaluates to True
        1 > 2        # Evaluates to False
        not 1 > 2    # Evaluates to True
        1 == 1       # Evaluates to True
        not 1 == 1   # Evaluates to False
        ```
        
    - **Conditions** `if` `elif` `else`
        
        ```python
        # if Statement
         
        test_value = 100
         
        if test_value > 1:
          # Expression evaluates to True
          print("This code is executed!")
         
        if test_value > 1000:
          # Expression evaluates to False
          print("This code is NOT executed!")
         
        print("Program continues at this point.")
        ```
        
        ```python
        # else Statement
         
        test_value = 50
         
        if test_value < 1:
          print("Value is < 1")
        else:
          print("Value is >= 1")
         
        test_string = "VALID"
         
        if test_string == "NOT_VALID":
          print("String equals NOT_VALID")
        else:
          print("String equals something else!")
        ```
        
        ```python
        pet_type = "fish"
         
        if pet_type == "dog":
          print("You have a dog.")
        elif pet_type == "cat":
          print("You have a cat.")
        elif pet_type == "fish":
          # this is performed
          print("You have a fish")
        else:
          print("Not sure!")
        ```
        
3. **Lists**
    - **List Structure** `[]` `[[]]`
        
        ```python
        primes = [2, 3, 5, 7, 11]
        print(primes)
         
        empty_list = []
        ```
        
        ```python
        numbers = [1, 2, 3, 4, 10]
        names = ['Jenny', 'Sam', 'Alexis']
        mixed = ['Jenny', 1, 2]
        list_of_lists = [['a', 1], ['b', 2]]
        ```
        
    - **List Indexing** `[i]` `[i][j]` `[-i]` `[:i]` `[i:]`
        
        ```python
        berries = ["blueberry", "cranberry", "raspberry"]
         
        berries[0]   # "blueberry"
        berries[2]   # "raspberry"
        ```
        
        ```python
        berries = ["blueberry", "cranberry", "raspberry"]
         
        berries[0]   # "blueberry"
        berries[2]   # "raspberry"
        ```
        
        ```python
        soups = ['minestrone', 'lentil', 'pho', 'laksa']
        soups[-1]   # 'laksa'
        soups[-3:]  # 'lentil', 'pho', 'laksa'
        soups[:-2]  # 'minestrone', 'lentil'x
        ```
        
    - **List Methods** `.append()` `.remove()` `.count()` `.pop()`
        
        ```python
        orders = ['daisies', 'periwinkle']
        orders.append('tulips')
        print(orders)
        # Result: ['daisies', 'periwinkle', 'tulips']
        ```
        
        ```python
        # Create a list
        shopping_line = ["Cole", "Kip", "Chris", "Sylvana", "Chris"]
         
        # Removes the first occurance of "Chris"
        shopping_line.remove("Chris")
        print(shopping_line)
         
        # Output
        # ["Cole", "Kip", "Sylvana", "Chris"]
        ```
        
        ```python
        backpack = ['pencil', 'pen', 'notebook', 'textbook', 'pen', 'highlighter', 'pen']
        numPen = backpack.count('pen')
        print(numPen)
        # Output: 3
        ```
        
        ```python
        # A list of CS topics which needs to be corrected
        cs_topics = ["Python", "Data Structures", "Balloon Making", "Algorithms", "Clowns 101"]
         
        # Pop the last element
        removed_element = cs_topics.pop()
        print(cs_topics)
        print(removed_element)
         
        # Output
        # ['Python', 'Data Structures', 'Balloon Making', 'Algorithms']
        # 'Clowns 101'
         
        # Pop the element "Baloon Making"
        cs_topics.pop(2)
        print(cs_topics)
         
        # Output
        # ['Python', 'Data Structures', 'Algorithms']
        ```
        
4. **Loops**
    - **For Loop** `for i in list:`
        
        ```python
        for <temporary variable> in <list variable>:
          <action statement>
          <action statement>
         
        #each num in nums will be printed below
        nums = [1,2,3,4,5]
        for num in nums: 
          print(num)
        ```
        
    1. **While Loop** `while true:`
    2. **Nested Loops** `for list in lists: for l in list:`
5. **Functions**
    1. **Returning** `return x`
    2. **Parameters** `def func(p1, p2, p3):`
    3. **Variables** `global` `local` `environment`
    4. **Arguments** `args` `kwargs`
6. **Dictionaries**
    1. **Data Structure** `{1: '1', 2: 2, 3: [3]}` 
    2. **Keys and Values** `dict[key] = value`
    3. **Dictionary Methods** `.update()` `.get()` `.pop()`
7. **Classes**
    1. **Instantiating** `class Class:`
    2. **Polymorphism** `AClass()` `BClass()`
    3. **Inheritance** `ParentClass()` `ChildClass()`
8. **Programming**
    1. **Sources** `github` `gitlab`
    2. **Libraries** `numpy` `pandas` `scikit-learn` 
    3. **Frameworks** `tensorflow` `gym`
- **Data:**
    
    ![Introduction%20to%20Python%20Programming%20Outline%209f4ec37f3752416fa22edcda78c1cbb0/Untitled.png](Introduction%20to%20Python%20Programming%20Outline%209f4ec37f3752416fa22edcda78c1cbb0/Untitled.png)
    
- **Challenge:**
    
    ![Introduction%20to%20Python%20Programming%20Outline%209f4ec37f3752416fa22edcda78c1cbb0/Untitled%201.png](Introduction%20to%20Python%20Programming%20Outline%209f4ec37f3752416fa22edcda78c1cbb0/Untitled%201.png)
    

```python
yearly_salary = [72000, 48000, 54000, 61000, 1000, 58000, 52000, 79000, 83000, 67000]

x_min = min(yearly_salary)
x_max = max(yearly_salary)

normalized_salary = []
for x_i in yearly_salary:
    x_new = (x_i - x_min) / (x_max - x_min)
    normalized_salary.append(x_new)

print(normalized_salary)
```

```python
import matplotlib.pyplot as plt
from math import sqrt

data_x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
data_y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
data_label = [True, True, False, True, False, False, False, False, True, False, False, False, False]

color_list = []
for label in data_label:
    if label == True:
        color_list.append("blue")
    else:
        color_list.append("red")

new_data = (10, 92)

distances = []
for x, y in zip(data_x, data_y):
    distance = sqrt((new_data[0] - x) ** 2 + (new_data[1] - y) ** 2)
    distances.append(distance)

zipped_dataset = zip(distances, data_x, data_y, data_label, color_list)
sorted_dataset = sorted(zipped_dataset)
print(sorted_dataset)

k = 1
votes = []
for data in sorted_dataset[:k]:
    if data[3] == True:
        votes.append(True)
    else:
        votes.append(False)
print(votes)

count_true = 0
count_false = 0
for vote in votes:
    if vote == True:
        count_true = count_true + 1
    else:
        count_false = count_false + 1

print(count_true)
print(count_false)

if count_true > count_false:
    color = "cyan"
elif count_true == count_false:
    print("your voting ended up with equality, democracy loses, peace wins")
    color = "purple"
else:
    color = "orange"

plt.axes().set_aspect('equal', 'datalim')
plt.scatter(data_x, data_y, c = color_list)
plt.scatter(new_data[0], new_data[1], c = color)
plt.show()
```

## Documentation to Start Using Python in an Interactive Development Environment:

### Install Python and Anaconda:

[Python Releases for Windows](https://www.python.org/downloads/windows/)

[Python Releases for Mac OS X](https://www.python.org/downloads/mac-osx/)

[Anaconda | Individual Edition](https://www.anaconda.com/products/individual)

### Install and Open Visual Studio Code (or any IDE of your choice):

[Download Visual Studio Code - Mac, Linux, Windows](https://code.visualstudio.com/download)

### Open an Empty File for Your Project and Open this File in VSCode:

![Introduction%20to%20Python%20Programming%20Outline%209f4ec37f3752416fa22edcda78c1cbb0/Untitled%202.png](Introduction%20to%20Python%20Programming%20Outline%209f4ec37f3752416fa22edcda78c1cbb0/Untitled%202.png)

### Configure Python Interpreter:

Simply hit `Ctrl + Shift + P` and search for `Python: Select Interpreter` 

Select the `Python` version that you have installed

If you get stuck, read the official documentation below:

[Using Python Environments in Visual Studio Code](https://code.visualstudio.com/docs/python/environments)