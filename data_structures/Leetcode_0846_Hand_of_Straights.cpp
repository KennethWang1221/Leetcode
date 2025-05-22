#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <algorithm>
using namespace std;

class Solution {
public:
    bool isNStraightHand(vector<int>& hand, int groupSize) {
        int n = hand.size();
        if (n % groupSize != 0) {
            return false;  // If total number of cards can't be divided by group size, return false
        }

        unordered_map<int, int> count;
        for (int n : hand) {
            count[n]++;  // Count the frequency of each number
        }

        // Min-heap to get the smallest number in the hand
        priority_queue<int, vector<int>, greater<int>> minH;
        for (auto& p : count) {
            minH.push(p.first);
        }

        while (!minH.empty()) {
            int first = minH.top();  // Get the smallest number
            for (int i = 0; i < groupSize; ++i) {
                if (count[first + i] == 0) {
                    return false;  // If the next number in sequence is not available, return false
                }
                count[first + i]--;  // Decrease the count of that number
                if (count[first + i] == 0) {
                    if (first + i != minH.top()) {
                        return false;  // If the number we processed is not the expected one in the heap, return false
                    }
                    minH.pop();  // Pop the number from the heap if it's exhausted
                }
            }
        }

        return true;  // Return true if we successfully formed all groups
    }
};

int main() {
    Solution solution;
    vector<int> hand = {1, 2, 3, 6, 2, 3, 4, 7, 8};
    int groupSize = 3;

    bool result = solution.isNStraightHand(hand, groupSize);
    cout << (result ? "True" : "False") << endl;  // Expected output: True

    return 0;
}

// g++ -std=c++11 Leetcode_0846_Hand_of_Straights.cpp -o test && ./test