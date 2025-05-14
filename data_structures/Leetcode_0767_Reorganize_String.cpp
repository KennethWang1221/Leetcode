#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <string>
using namespace std;

class Solution {
public:
    string reorganizeString(string s) {
        unordered_map<char, int> count_map;
        
        // Step 1: Count frequency of each character
        for (char c : s) {
            count_map[c]++;
        }

        // Step 2: Max-heap to store characters by their frequency
        priority_queue<pair<int, char>> maxheap;

        // Push characters into the max-heap with frequency and character
        for (auto& entry : count_map) {
            maxheap.push({entry.second, entry.first});
        }

        string result = "";
        pair<int, char> prev = {-1, '#'};  // To store the previous character and its remaining frequency

        while (!maxheap.empty()) {
            // Get the character with the highest frequency
            auto current = maxheap.top();
            maxheap.pop();

            result += current.second;  // Add the character to the result string

            // Decrease its frequency as it's being used
            current.first--;

            // If the previous character exists and still has frequency left, push it back to the heap
            if (prev.first > 0) {
                maxheap.push(prev);
            }

            // If the current character still has frequency, push it back to the heap
            if (current.first > 0) {
                prev = current;
            } else {
                prev = {-1, '#'};  // No need to save if current character's frequency is 0
            }
        }

        // Step 3: Check if the string's length matches the original string
        if (result.length() == s.length()) {
            return result;  // Return the reorganized string if it's valid
        } else {
            return "";  // Return an empty string if not possible to reorganize
        }
    }
};

int main() {
    Solution solution;
    string s = "aab";
    cout << solution.reorganizeString(s) << endl;  // Output should be "aba"

    return 0;
}



// g++ -std=c++17 Leetcode_0767_Reorganize_String.cpp -o test