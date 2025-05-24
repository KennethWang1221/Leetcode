#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>

using namespace std;

class Solution {
public:
    vector<int> findSubstring(string s, vector<string>& words) {
        vector<int> res;
        if (s.empty() || words.empty()) return res;

        int n = s.size();
        int word_len = words[0].size();
        int word_count = words.size();
        int total_len = word_len * word_count;

        unordered_map<string, int> word_freq;
        for (const string& word : words)
            word_freq[word]++;

        // Try all possible start offsets within one word length
        for (int start = 0; start < word_len; ++start) {
            int left = start;
            int matched = 0;
            unordered_map<string, int> curr_freq;

            for (int right = start; right <= (int)s.size() - word_len; right += word_len) {
                string word = s.substr(right, word_len);

                if (word_freq.find(word) == word_freq.end()) {
                    curr_freq.clear();
                    left = right + word_len;
                    matched = 0;
                    continue;
                }

                curr_freq[word]++;
                if (curr_freq[word] == word_freq[word])
                    matched++;

                // Shrink window if we have more than needed
                while (curr_freq[word] > word_freq[word]) {
                    string left_word = s.substr(left, word_len);
                    if (curr_freq[left_word] == word_freq[left_word])
                        matched--;
                    curr_freq[left_word]--;
                    left += word_len;
                }

                // If all matched, record position
                if (matched == word_freq.size())
                    res.push_back(left);
            }
        }

        return res;
    }
};

// Test Case
int main() {
    Solution sol;
    string s = "bcabbcaabbccacacbabccacaababcbb";
    vector<string> words = {"c", "b", "a", "c", "a", "a", "a", "b", "c"};

    vector<int> result = sol.findSubstring(s, words);

    cout << "[ ";
    for (int idx : result) {
        cout << idx << " ";
    }
    cout << "]" << endl;

    return 0;
}
// g++ -std=c++17 Leetcode_0030_Substring_with_Concatenation_of_All_Words.cpp -o test