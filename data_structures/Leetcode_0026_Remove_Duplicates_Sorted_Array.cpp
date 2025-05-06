#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int L = 1;  // Pointer for the position to place the next unique element
        int n = nums.size();
        
        for (int R = 1; R < n; R++) {
            if (nums[R] != nums[R - 1]) {
                nums[L] = nums[R];
                L++;
            }
        }
        
        return L;  // Return the length of the array without duplicates
    }
};

int main() {
    Solution solution;
    vector<int> nums = {0, 0, 1, 1, 1, 2, 2, 3, 3, 4};  // Example array
    int result = solution.removeDuplicates(nums);

    // Output the result
    cout << "Length after removing duplicates: " << result << endl;
    cout << "Modified array: ";
    for (int i = 0; i < result; i++) {
        cout << nums[i] << " ";
    }
    cout << endl;

    return 0;
}
// g++ -std=c++17 Leetcode_0026_Remove_Duplicates_Sorted_Array.cpp -o test
