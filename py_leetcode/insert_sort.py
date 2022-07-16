def insert_sort(nums):
    for i in range(1,len(nums)):
        j = i-1
        key = nums[i]
        while j >=0 and nums[j] > key:
            nums[j+1] = nums[j]
            j = j-1
        nums[j+1] = key
    return nums
nums = [2,1]
print(insert_sort(nums))