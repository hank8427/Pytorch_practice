matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]]
target = 3

def getValuebyIndex(index, col):
    r = index//col
    c = index%col
    return r, c
def searchMatrix(matrix, target):
    row = len(matrix)
    col = len(matrix[0])

    left = 0
    right = row*col-1

    while left <= right:
        print(left, right)
        mid = (right+left)//2
        if target == matrix[getValuebyIndex(mid, col)[0]][getValuebyIndex(mid, col)[1]]:
            return True
        elif target > matrix[getValuebyIndex(mid, col)[0]][getValuebyIndex(mid, col)[1]]:
            left = mid + 1
        elif target < matrix[getValuebyIndex(mid, col)[0]][getValuebyIndex(mid, col)[1]]:
            right = mid - 1
    return False

print(searchMatrix(matrix, target))




