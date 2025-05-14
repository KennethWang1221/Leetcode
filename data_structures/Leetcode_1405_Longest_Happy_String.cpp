#include <iostream>
#include <vector>
#include <queue>
#include <tuple>
using namespace std;

class Solution {
public:
    string longestDiverseString(int a, int b, int c) {
        string res = "";
        priority_queue<pair<int, char>> maxHeap;  // Max-heap to store (count, char)

        // Push all non-zero counts of 'a', 'b', and 'c' into the heap
        if (a > 0) maxHeap.push({a, 'a'});
        if (b > 0) maxHeap.push({b, 'b'});
        if (c > 0) maxHeap.push({c, 'c'});

        while (!maxHeap.empty()) {
            auto [count, char_] = maxHeap.top();
            maxHeap.pop();

            // Check if we have two consecutive characters of the same kind in the result
            if (res.size() > 1 && res[res.size() - 1] == res[res.size() - 2] && res[res.size() - 1] == char_) {
                if (maxHeap.empty()) {
                    break;  // We cannot add any more valid characters, break out
                }

                // Pop the next most frequent character and add it to the result
                auto [count2, char2] = maxHeap.top();
                maxHeap.pop();
                res += char2;
                count2--;  // Decrease the count

                // If the character still has a remaining count, push it back into the heap
                if (count2 > 0) {
                    maxHeap.push({count2, char2});
                }

                // After adding the second character, push the current character back into the heap
                maxHeap.push({count, char_});
            } else {
                // If we don't have two consecutive same characters, simply add the current character
                res += char_;
                count--;  // Decrease the count

                if (count > 0) {
                    maxHeap.push({count, char_});
                }
            }
        }

        return res;
    }
};

int main() {
    Solution solution;
    
    // Test cases
    cout << solution.longestDiverseString(1, 1, 7) << endl;  // Output should be a valid string, such as "ccaccbcc"
    cout << solution.longestDiverseString(7, 1, 0) << endl;  // Output should be "aabaa"

    return 0;
}
// g++ -std=c++17 Leetcode_1405_Longest_Happy_String.cpp -o test