# -*- coding:utf-8 -*-
def intersection(nums1, nums2):
    lookup = {}
    result = set()
    for i in nums1:
        if i not in lookup:
            lookup[i] = 1
        else:
            lookup[i] += 1

    for j in nums2:
        if j not in lookup:
            #return False
            pass

        else:
            result.add(j)
            # lookup[j]-=1

    # for k in lookup:
    #     if lookup[k] !=0:
    #         return False

    return list(result)

nums1 = [4,9,5]
nums2 = [9,4,9,8,4]
result = intersection(nums1, nums2)
print(result)