def add(a, b):
    answer = a+b
    print(str(a) + " + " + str(b) + " = " + str(answer) + "\n")

def sub(a, b):
    answer = a-b
    print(str(a) + " - " + str(b) + " = " + str(answer) + "\n")

def mul(a, b):
    answer = a*b
    print(str(a) + " * " + str(b) + " = " + str(answer) + "\n")

def div(a, b):
    answer = a/b
    print(str(a) + " / " + str(b) + " = " + str(answer) + "\n")

while True:
    print("A, Addition")
    print("B, Subtraction")
    print("C, Multiplication")
    print("D, Division" + "\n")

    choice = input("Input Your Choice:  " + "\n")

    if choice == "A" or choice == "a":
        print("Addition")
        a = int(input("Input First Number:  "))
        b = int(input("Input Second Number:  "))
        add(a, b)
    elif choice == "B" or choice == "b":
        print("Subtraction")
        a = int(input("Input First Number:  "))
        b = int(input("Input Second Number:  "))
        sub(a, b)

    elif choice == "C" or choice == "c":
        print("Multiplication")
        a = int(input("Input First Number:  "))
        b = int(input("Input Second Number:  "))
        mul(a, b)

    elif choice == "D" or choice == "d":
        print("Division")
        a = int(input("Input First Number:  "))
        b = int(input("Input Second Number:  "))
        div(a, b)

    elif choice == 'e' or choice =="E":
        print("Program Ended")
        quit()
    
    else:
        print(" ------ ")
        print(" Invalid ")
        print(" ------ "  + "\n")
        print("invalid Choice")
        print("Please Choose Valid Character from Below mentioned")

