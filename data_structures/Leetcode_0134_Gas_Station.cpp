#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int curSum = 0;     // Current remaining fuel
        int totalSum = 0;   // Total remaining fuel
        int start = 0;      // Starting position

        for (int i = 0; i < gas.size(); ++i) {
            int diff = gas[i] - cost[i];
            curSum += diff;
            totalSum += diff;

            if (curSum < 0) {
                start = i + 1;
                curSum = 0;
            }
        }

        return (totalSum >= 0) ? start : -1;
    }
};

// Test Case
int main() {
    Solution sol;
    vector<int> gas = {1, 2, 3, 4, 5};
    vector<int> cost = {3, 4, 5, 1, 2};

    int res = sol.canCompleteCircuit(gas, cost);
    cout << "Starting Index: " << res << endl; // Output: 3

    return 0;
}
// g++ -std=c++11 Leetcode_0134_Gas_Station.cpp -o test && ./test