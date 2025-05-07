#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    bool search(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (nums[mid] == target) {
                return true;  // Target found
            }

            // Left sorted portion
            if (nums[left] < nums[mid]) {
                if (nums[left] <= target && target < nums[mid]) {
                    right = mid - 1;  // Search in the left portion
                } else {
                    left = mid + 1;   // Search in the right portion
                }
            }
            // Right sorted portion
            else if (nums[left] > nums[mid]) {
                if (nums[mid] < target && target <= nums[right]) {
                    left = mid + 1;   // Search in the right portion
                } else {
                    right = mid - 1;  // Search in the left portion
                }
            }
            // Duplicates, we cannot determine which part is sorted, so move the left pointer
            else {
                left++;  // Skip the duplicate element
            }
        }
        
        return false;  // Target not found
    }
};

int main() {
    Solution solution;
    vector<int> nums = {2, 5, 6, 0, 0, 1, 2};
    int target = 0;
    
    bool result = solution.search(nums, target);
    cout << "Target found: " << (result ? "True" : "False") << endl;  // Expected output: True

    return 0;
}

// g++ -std=c++17 Leetcode_0081_Search_in_Rotated_Sorted_Array_II.cpp -o test
