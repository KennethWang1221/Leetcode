#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
using namespace std;

class Solution {
public:
    bool isAlienSorted(vector<string>& words, string order) {
        // Create a map for each character's order in the alien alphabet
        unordered_map<char, int> order_map;
        for (int i = 0; i < order.size(); ++i) {
            order_map[order[i]] = i;
        }

        // Convert each word to its respective "alien" order values
        vector<vector<int>> new_words;
        for (const string& word : words) {
            vector<int> temp;
            for (char c : word) {
                temp.push_back(order_map[c]);
            }
            new_words.push_back(temp);
        }

        // Compare adjacent words
        for (int i = 1; i < new_words.size(); ++i) {
            if (new_words[i - 1] > new_words[i]) {
                return false;
            }
        }

        return true;
    }
};

int main() {
    Solution solution;

    // Test case 1
    vector<string> words1 = {"word", "world", "row"};
    string order1 = "worldabcefghijkmnpqstuvxyz";
    cout << (solution.isAlienSorted(words1, order1) ? "True" : "False") << endl;  // Expected: False

    // Test case 2
    vector<string> words2 = {"hello", "leetcode"};
    string order2 = "hlabcdefgijkmnopqrstuvwxyz";
    cout << (solution.isAlienSorted(words2, order2) ? "True" : "False") << endl;  // Expected: True

    return 0;
}
// g++ -std=c++17 Leetcode_0953_Verifying_an_Alien_Dictionary.cpp -o test