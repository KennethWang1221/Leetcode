#include <iostream>
#include <vector>
#include <unordered_set>  // for unordered_set
using namespace std;

class Solution {
public:
    bool containsNearbyDuplicate(vector<int>& nums, int k) {
        unordered_set<int> window;
        int L = 0;
        int n = nums.size();
        
        for (int R = 0; R < n; R++) {
            // If the window size exceeds k, remove the element at L
            while (R - L > k) {
                window.erase(nums[L]);
                L++;
            }
            
            // Check if the current element is in the window
            if (window.find(nums[R]) != window.end()) {
                /* 
                If it finds it, it returns an iterator pointing to that value .
                If it doesn't find it, it returns window.end() — which is like a "not found" signal.
                */
                return true;
            }

            /* Or can use this way to check if duplicates
            Since sets do not allow duplicates , it will never return more than 1 .
            Returns 1 if the value exists in the set.
            Returns 0 if it does not exist.
            if (window.count(nums[r])) {
                return true;
            }
            */
            
            // Add the current element to the window
            window.insert(nums[R]);
        }

        return false;
    }
};

int main() {
    Solution solution;
    vector<int> nums = {1, 2, 3, 1};
    int k = 3;

    bool result = solution.containsNearbyDuplicate(nums, k);
    cout << (result ? "True" : "False") << endl;

    // Test with another example
    nums = {1, 2, 3, 1, 2, 3};
    k = 2;
    result = solution.containsNearbyDuplicate(nums, k);
    cout << (result ? "True" : "False") << endl;

    return 0;
}
// g++ -std=c++17 Leetcode_0219_Contains_Duplicate_II.cpp -o test