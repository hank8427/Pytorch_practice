nums = [5,1,1,2,0,0]

def quicksort(nums):
    if len(nums) <= 1:
        return nums
    else:
        pivot = nums.pop()

    low = []
    high = []
    for i in nums:
        if i <= pivot:
            low.append(i)
        elif i > pivot:
            high.append(i)
    return quicksort(low) + [pivot] + quicksort(high)

print(quicksort(nums))
