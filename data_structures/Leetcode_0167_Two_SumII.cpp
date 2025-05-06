#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        int l = 0, r = numbers.size() - 1;

        while (l < r) {
            int diff = numbers[l] + numbers[r];

            if (diff > target) {
                r--;
            } else if (diff < target) {
                l++;
            } else {
                return {l + 1, r + 1};  // Return 1-based indices
            }
        }

        return {};  // If no solution is found
    }
};

int main() {
    Solution solution;
    vector<int> numbers = {2, 7, 11, 15};
    int target = 9;
    
    vector<int> result = solution.twoSum(numbers, target);

    if (!result.empty()) {
        cout << "Indices: " << result[0] << ", " << result[1] << endl;
    } else {
        cout << "No solution found" << endl;
    }

    return 0;
}



// g++ -std=c++17 Leetcode_0167_Two_SumII.cpp -o test
