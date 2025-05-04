#include <iostream>
#include <vector>

class Solution {
public:
    int removeElement(std::vector<int>& nums, int val) {
        int n = nums.size();
        int l = 0;

        for (int r = 0; r < n; ++r) {
            if (nums[r] != val) {
                nums[l] = nums[r];
                l++;
            }
        }

        return l;  // Return the new length of the array
    }
};

int main() {
    Solution solution;
    std::vector<int> nums = {3, 2, 2, 3};
    int val = 3;

    int new_length = solution.removeElement(nums, val);
    
    std::cout << "New length: " << new_length << std::endl;
    std::cout << "Modified array: ";
    for (int i = 0; i < new_length; ++i) {
        std::cout << nums[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
// g++ -std=c++17 Leetcode_0027_Remove_Element.cpp -o test