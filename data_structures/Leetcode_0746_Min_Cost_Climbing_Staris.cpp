#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int minCostClimbingStairs(vector<int>& cost) {
        int n = cost.size();
        cost.push_back(0);  // Add the 0 for the top of the stairs (the top is considered as a step with 0 cost)
        
        // Starting from the second-to-last element, calculate the minimum cost to reach each step
        for (int i = n - 3; i >= 0; --i) {
            cost[i] += min(cost[i + 1], cost[i + 2]);
        }

        // The result is the minimum of starting from the first or the second step
        return min(cost[0], cost[1]);
    }
};

int main() {
    Solution sol;
    vector<int> cost = {1, 100, 1, 1, 1, 100, 1, 1, 100, 1};
    int result = sol.minCostClimbingStairs(cost);
    cout << result << endl;  // Expected output: 6
    
    return 0;
}


// g++ -std=c++17 Leetcode_0746_Min_Cost_Climbing_Staris.cpp -o test