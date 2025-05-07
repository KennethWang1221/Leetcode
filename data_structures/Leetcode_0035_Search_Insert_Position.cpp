#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int start = 0, end = nums.size() - 1;

        while (start <= end) {
            int middle = (start + end) / 2;
            if (nums[middle] < target) {
                start = middle + 1;
            } else if (nums[middle] > target) {
                end = middle - 1;
            } else {
                return middle;  // Found the target, return the index
            }
        }

        return start;  // Target not found, return the insert position
    }
};

int main() {
    Solution solution;
    vector<int> nums = {1, 3, 5, 6};
    int target = 5;

    int result = solution.searchInsert(nums, target);
    cout << "Insert position: " << result << endl;  // Expected output: 2

    return 0;
}

// g++ -std=c++17 Leetcode_0035_Search_Insert_Position.cpp -o test