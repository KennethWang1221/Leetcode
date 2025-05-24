#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int l = 0, r = 0, n = nums.size();

        while (r < n) {
            int count = 1;

            // Count duplicates of the current element
            while (r + 1 < n && nums[r] == nums[r + 1]) {
                r++;
                count++;
            }

            // Place at most two duplicates of the current element
            for (int i = 0; i < min(2, count); ++i) {
                nums[l] = nums[r];
                l++;
            }

            // Move to the next different element
            r++;
        }

        return l;  // The new length of the modified array
    }
};

int main() {
    Solution solution;

    // Test case
    vector<int> nums = {0, 0, 1, 1, 1, 1, 2, 3, 3};
    int newLength = solution.removeDuplicates(nums);

    // Print the modified array and its new length
    cout << "New Length: " << newLength << endl;
    cout << "Modified Array: ";
    for (int i = 0; i < newLength; ++i) {
        cout << nums[i] << " ";
    }
    cout << endl;

    return 0;
}

//g++ -std=c++11 Leetcode_0080_Remove_Duplicates_Sorted_Array_II.cpp -o test && ./test