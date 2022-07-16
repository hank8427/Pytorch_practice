class Solution:
    def twoSum(self, nums, target):
        for i in range(len(nums)):
            for j in range (i+1,len(nums)):
                result = nums[i] + nums[j]
                if result == target:
                    return i, j

    # def twoSum(self, nums: List[int], target: int) -> List[int]:
    #     dict = {}
    #     for i in range(len(nums)):   
    #         if target-nums[i] not in dict:
    #             print("gggg")
    #             dict[nums[i]] = i
    #         else:    
    #             return(dict[target-nums[i]], i)
