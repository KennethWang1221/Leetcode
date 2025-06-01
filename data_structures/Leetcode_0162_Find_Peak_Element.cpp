#include <iostream>
#include <vector>

using namespace std;
class Solution {
public:

    int findPeakElement(vector<int>& nums) {
        int l = 0;
        int r = nums.size() - 1;
        int mid = 0;
        while (l <= r) {
            mid = l + (r - l) / 2;
            if (mid < nums.size() - 1 && nums[mid] < nums[mid + 1]) {
                l = mid + 1;
            } else if (mid > 0 && nums[mid] < nums[mid - 1]) {
                r = mid - 1;
            } else {
                break;
            }
        }
        return mid;
    }
};

int main() {
    vector<int> nums = {1, 2, 3, 1};
    Solution sol;
    int result = sol.findPeakElement(nums);
    cout << "Peak element index: " << result << endl; // Expected output: 2
    return 0;
}

// g++ -std=c++17 Leetcode_0162_Find_Peak_Element.cpp -o test
