target = 6
array = [1,2,3,4,5,6]
# print(array)
# n=(len(array))
# temp = []
# for i in range(n):
#   i= i+1
#   print(array[-i])
#   temp.append(array[-i])
# array = temp
# print(temp)
def summ(array, target, par = []):

  s = sum(par)
  print(s)
  if s == target:
    yield par
  if s >= target:
    return
  for i in range(len(array)):
    a = array[i]
    rem = array[i+1:]
    yield summ(rem, target, par + [n])
if __name__ == "__main__":
  print(summ([1,2,3,4,5,6,7,8], 6))
