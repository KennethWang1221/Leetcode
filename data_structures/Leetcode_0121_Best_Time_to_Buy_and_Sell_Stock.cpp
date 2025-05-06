// Greedy 

#include <iostream>
#include <vector>
#include <algorithm>  // for min and max
using namespace std;

class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        int res = 0;
        int low = INT_MAX;  // Initialize to a very large number

        for (int i = 0; i < n; i++) {
            low = min(low, prices[i]);  // Update the lowest price
            int profit = prices[i] - low;  // Calculate potential profit
            res = max(res, profit);  // Update the result with the maximum profit
        }

        return res;
    }

    int maxProfit_sliding_window(vector<int>& prices) {
        int l = 0, r = 1, n = prices.size();
        int maxP = 0;

        while (r < n) {
            if (prices[l] < prices[r]) {
                int profit = prices[r] - prices[l];
                maxP = max(maxP, profit);
            } else {
                l = r;  // Update l to r if prices[l] is greater than or equal to prices[r]
            }
            r++;  // Move the right pointer forward
        }

        return maxP;
    }
};

int main() {
    Solution solution;
    vector<int> prices = {7, 1, 5, 3, 6, 4};
    
    int result = solution.maxProfit(prices);
    cout << "Maximum profit: " << result << endl;

    int result2 = solution.maxProfit_sliding_window(prices);
    cout << "Maximum profit: " << result2 << endl;

    return 0;
}
// g++ -std=c++17 Leetcode_0121_Best_Time_to_Buy_and_Sell_Stock.cpp -o test