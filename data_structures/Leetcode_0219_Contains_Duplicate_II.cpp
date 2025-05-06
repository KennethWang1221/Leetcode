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
                return true;
            }
            
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