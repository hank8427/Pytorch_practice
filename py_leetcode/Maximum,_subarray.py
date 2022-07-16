import numpy as np
import time

start = time.time()
# nums = np.random.randint(-100,100,(1000,1))
nums = [-2,1,1,-3,4,1,-1]
curr_sum = [nums[0]]

for i in range(1,len(nums)):
    value = nums[i]
    if curr_sum[i-1] > 0:
        curr_sum.append(curr_sum[i-1] + value)
    else:
        curr_sum.append(value)
result = np.max(curr_sum)
print(result)

end = time.time()
# print(end - start)