#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    int candy(vector<int>& ratings) {
        int n = ratings.size();
        vector<int> candies(n, 1); // Every child starts with at least one candy

        // Left to right: if current rating > left neighbor
        for (int i = 1; i < n; ++i) {
            if (ratings[i] > ratings[i - 1]) {
                candies[i] = candies[i - 1] + 1;
            }
        }

        // Right to left: if current rating > right neighbor
        for (int i = n - 2; i >= 0; --i) {
            if (ratings[i] > ratings[i + 1]) {
                candies[i] = max(candies[i], candies[i + 1] + 1);
            }
        }

        // Sum up all candies
        int totalCandies = 0;
        for (int c : candies) {
            totalCandies += c;
        }

        return totalCandies;
    }
};

// Test Case
int main() {
    Solution sol;
    vector<int> ratings1 = {1, 0, 2};
    cout << sol.candy(ratings1) << endl; // Output: 5

    vector<int> ratings2 = {5, 4, 3, 5, 6, 2};
    cout << sol.candy(ratings2) << endl; // Output: 14

    return 0;
}
// g++ -std=c++11 Leetcode_0135_Candy.cpp -o test && ./test