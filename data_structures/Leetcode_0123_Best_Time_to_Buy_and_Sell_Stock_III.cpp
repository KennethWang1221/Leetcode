#include <vector>
#include <algorithm>
#include <iostream>
using namespace std;

class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if (prices.empty()) 
            return 0;
        int n = prices.size();
        vector<int> prev(n, 0);
        
        for (int k = 0; k < 2; k++) {
            vector<int> curr(n, 0);
            int max_diff = -prices[0];
            for (int i = 1; i < n; i++) {
                curr[i] = max(curr[i-1], prices[i] + max_diff);
                max_diff = max(max_diff, prev[i] - prices[i]);
            }
            prev = curr;
        }
        
        return prev[n-1];
    }
};

int main() {
    Solution sol;
    vector<int> prices = {3,3,5,0,0,3,1,4};
    int result = sol.maxProfit(prices);
    cout << "Test case 1: " << result << endl; // Expected: 6

    prices = {1,2,3,4,5};
    result = sol.maxProfit(prices);
    cout << "Test case 2: " << result << endl; // Expected: 4

    prices = {7,6,4,3,1};
    result = sol.maxProfit(prices);
    cout << "Test case 3: " << result << endl; // Expected: 0

    prices = {1};
    result = sol.maxProfit(prices);
    cout << "Test case 4: " << result << endl; // Expected: 0

    return 0;
}

// g++ -std=c++17 Leetcode_0123_Best_Time_to_Buy_and_Sell_Stock_III.cpp -o test