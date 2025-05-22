#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    bool lemonadeChange(vector<int>& bills) {
        int five = 0, ten = 0; // To track the number of 5 and 10 dollar bills

        for (int bill : bills) {
            // Case 1: Received a 5 dollar bill
            if (bill == 5) {
                five++;
            }
            // Case 2: Received a 10 dollar bill
            else if (bill == 10) {
                ten++;
                // We need to give 5 dollars as change
                if (five > 0) {
                    five--;
                } else {
                    return false; // Not enough 5 dollar bills for change
                }
            }
            // Case 3: Received a 20 dollar bill
            else if (bill == 20) {
                // We need to give 15 dollars as change
                if (ten > 0 && five > 0) {
                    ten--;
                    five--;
                } else if (five >= 3) {
                    five -= 3;
                } else {
                    return false; // Not enough change to give
                }
            }
        }

        return true; // Successfully gave change to all customers
    }
};

int main() {
    Solution solution;

    vector<int> bills = {5, 5, 5, 10, 20};
    bool result = solution.lemonadeChange(bills);

    cout << (result ? "True" : "False") << endl; // Expected output: True

    return 0;
}

// g++ -std=c++17 Leetcode_0860_Lemonade_Change.cpp -o test 