nums = [-1, 0, 3, 5, 9, 12]
target = 9
def search(nums,target): 
    left = 0
    right = len(nums)-1
    while left <= right:
        mid = round((right+left)/2)
        # print(mid)
        if nums[mid] == target:
            return mid
        elif target > nums[mid]:
            left = mid+1
        else:
            right = mid
    return -1
print(search(nums, target))
        