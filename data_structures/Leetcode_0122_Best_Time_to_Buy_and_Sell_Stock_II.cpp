#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int profit = 0;
        int n = prices.size();

        // Loop through the prices to calculate profit
        for (int i = 1; i < n; ++i) {
            if (prices[i] > prices[i - 1]) {
                profit += (prices[i] - prices[i - 1]);
            }
        }

        return profit;
    }
};

int main() {
    Solution solution;

    vector<int> prices = {7, 1, 5, 3, 6, 4};
    int res = solution.maxProfit(prices);
    cout << res << endl;  // Expected output: 7 (Buy at 1, sell at 5, buy at 3, sell at 6)

    return 0;
}


// g++ -std=c++17 Leetcode_0122_Best_Time_to_Buy_and_Sell_Stock_II.cpp -o test