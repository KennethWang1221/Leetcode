#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <set>
#include <functional>

using namespace std;

class Solution {
    struct TrieNode {
        unordered_map<char, TrieNode*> children;
        bool isWord;
        int refs;

        TrieNode() : isWord(false), refs(0) {}
    };

public:
    void addWord(TrieNode* root, const string& word) {
        TrieNode* node = root;
        node->refs++;
        for (char c : word) {
            if (node->children.find(c) == node->children.end()) {
                node->children[c] = new TrieNode();
            }
            node = node->children[c];
            node->refs++;
        }
        node->isWord = true;
    }

    void removeWord(TrieNode* root, const string& word) {
        TrieNode* node = root;
        node->refs--;
        for (char c : word) {
            node = node->children[c];
            node->refs--;
        }
    }

    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
        TrieNode* root = new TrieNode();
        for (const string& word : words) {
            addWord(root, word);
        }

        int ROWS = board.size();
        int COLS = board[0].size();
        vector<vector<bool>> visited(ROWS, vector<bool>(COLS, false));
        set<string> result;

        function<void(int, int, TrieNode*, string)> dfs =
            [&](int r, int c, TrieNode* node, string word) {
                if (r < 0 || r >= ROWS || c < 0 || c >= COLS || visited[r][c]) {
                    return;
                }

                char ch = board[r][c];
                auto it = node->children.find(ch);
                if (it == node->children.end() || it->second->refs <= 0) {
                    return;
                }

                TrieNode* nextNode = it->second;
                visited[r][c] = true;
                word += ch;

                if (nextNode->isWord) {
                    result.insert(word);
                    nextNode->isWord = false;
                    removeWord(root, word);
                }

                dfs(r + 1, c, nextNode, word);
                dfs(r - 1, c, nextNode, word);
                dfs(r, c + 1, nextNode, word);
                dfs(r, c - 1, nextNode, word);

                visited[r][c] = false;
            };

        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLS; ++c) {
                dfs(r, c, root, "");
            }
        }

        return vector<string>(result.begin(), result.end());
    }
};

int main() {
    vector<vector<char>> board = {
        {'o', 'a', 'a', 'n'},
        {'e', 't', 'a', 'e'},
        {'i', 'h', 'k', 'r'},
        {'i', 'f', 'l', 'v'}
    };

    vector<string> words = {"oath", "pea", "eat", "rain"};

    Solution sol;
    vector<string> result = sol.findWords(board, words);

    cout << "Found words: ";
    for (const auto& word : result) {
        cout << word << " ";
    }
    cout << endl;

    return 0;
}


// g++ -std=c++17 Leetcode_0212_Word_Search_II.cpp -o test