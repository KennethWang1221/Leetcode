#include <iostream>
#include <vector>
#include <set>
using namespace std;

class Solution {
public:
    bool mergeTriplets(vector<vector<int>>& triplets, vector<int>& target) {
        set<int> good;  // To store the indices of valid target values

        // Iterate through each triplet in the list
        for (const auto& t : triplets) {
            // Skip the triplet if it exceeds the target in any of the positions
            if (t[0] > target[0] || t[1] > target[1] || t[2] > target[2]) {
                continue;
            }
            // Check if the values of the triplet match the target
            for (int i = 0; i < 3; ++i) {
                if (t[i] == target[i]) {
                    good.insert(i);
                }
            }
        }

        // If we found all the target values, return true
        return good.size() == 3;
    }
};

int main() {
    Solution solution;

    // Example usage
    vector<vector<int>> triplets = {{2, 5, 3}, {1, 8, 4}, {1, 7, 5}};
    vector<int> target = {2, 7, 5};

    bool result = solution.mergeTriplets(triplets, target);
    cout << (result ? "True" : "False") << endl;  // Expected output: True

    return 0;
}
// g++ -std=c++11 Leetcode_1899_Merge_Triplets_to_Form_Target_Triplet.cpp -o test && ./test