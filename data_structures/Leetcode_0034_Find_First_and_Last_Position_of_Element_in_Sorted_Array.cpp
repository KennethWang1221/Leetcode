#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        int left = binSearch(nums, target, true);
        int right = binSearch(nums, target, false);
        return {left, right};
    }

private:
    int binSearch(vector<int>& nums, int target, bool leftBias) {
        int l = 0;
        int r = nums.size() - 1;
        int result = -1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (target > nums[mid]) {
                l = mid + 1;
            } else if (target < nums[mid]) {
                r = mid - 1;
            } else {
                result = mid;
                if (leftBias) {
                    r = mid - 1;
                } else {
                    l = mid + 1;
                }
            }
        }
        return result;
    }
};

int main() {
    Solution solution;
    
    vector<int> nums1 = {5,7,7,8,8,10};
    int target1 = 8;
    vector<int> result1 = solution.searchRange(nums1, target1);
    cout << "Test case 1: [" << result1[0] << ", " << result1[1] << "]" << endl; // Expected: [3, 4]
    
    vector<int> nums2 = {5,7,7,8,8,10};
    int target2 = 6;
    vector<int> result2 = solution.searchRange(nums2, target2);
    cout << "Test case 2: [" << result2[0] << ", " << result2[1] << "]" << endl; // Expected: [-1, -1]
    
    vector<int> nums3 = {};
    int target3 = 0;
    vector<int> result3 = solution.searchRange(nums3, target3);
    cout << "Test case 3: [" << result3[0] << ", " << result3[1] << "]" << endl; // Expected: [-1, -1]
    
    vector<int> nums4 = {1};
    int target4 = 1;
    vector<int> result4 = solution.searchRange(nums4, target4);
    cout << "Test case 4: [" << result4[0] << ", " << result4[1] << "]" << endl; // Expected: [0, 0]
    
    vector<int> nums5 = {2, 2};
    int target5 = 2;
    vector<int> result5 = solution.searchRange(nums5, target5);
    cout << "Test case 5: [" << result5[0] << ", " << result5[1] << "]" << endl; // Expected: [0, 1]
    
    return 0;
}
// g++ -std=c++17 Leetcode_0034_Find_First_and_Last_Position_of_Element_in_Sorted_Array.cpp -o test
