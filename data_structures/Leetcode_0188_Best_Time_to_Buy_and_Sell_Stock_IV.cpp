#include <vector>
#include <algorithm>
#include <iostream>
using namespace std;

class Solution {
public:
    int maxProfit(int k, vector<int>& prices) {
        int n = prices.size();
        if (n == 0) 
            return 0;
        
        if (k >= n / 2) {
            int profit = 0;
            for (int i = 1; i < n; i++) {
                if (prices[i] > prices[i-1]) {
                    profit += prices[i] - prices[i-1];
                }
            }
            return profit;
        }
        
        vector<int> prev(n, 0);
        for (int i = 1; i <= k; i++) {
            vector<int> curr(n, 0);
            int max_diff = -prices[0];
            for (int j = 1; j < n; j++) {
                curr[j] = max(curr[j-1], prices[j] + max_diff);
                max_diff = max(max_diff, prev[j] - prices[j]);
            }
            prev = curr;
        }
        
        return prev[n-1];
    }
};

int main() {
    Solution sol;
    vector<int> prices;
    int k, result;

    k = 2;
    prices = {2,4,1};
    result = sol.maxProfit(k, prices);
    cout << "Test case 1: " << result << endl; // Expected: 2

    k = 2;
    prices = {3,3,5,0,0,3,1,4};
    result = sol.maxProfit(k, prices);
    cout << "Test case 2: " << result << endl; // Expected: 6

    k = 2;
    prices = {1,2,4,2,5,7,2,4,9,0};
    result = sol.maxProfit(k, prices);
    cout << "Test case 3: " << result << endl; // Expected: 13

    return 0;
}

// g++ -std=c++17 Leetcode_0188_Best_Time_to_Buy_and_Sell_Stock_IV.cpp -o test