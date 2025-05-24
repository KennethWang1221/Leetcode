#include <iostream>
#include <sstream>
#include <vector>
using namespace std;

class Solution {
public:
    int lengthOfLastWord(string s) {
        // Remove leading and trailing spaces
        int n = s.length();
        int right = n - 1;

        // Skip trailing spaces
        while (right >= 0 && s[right] == ' ') {
            right--;
        }

        // Find the length of the last word
        int length = 0;
        while (right >= 0 && s[right] != ' ') {
            length++;
            right--;
        }

        return length;
    }
};

int main() {
    Solution solution;

    // Test case
    string s = "   fly me   to   the moon  ";
    int result = solution.lengthOfLastWord(s);

    // Print the result
    cout << "Length of Last Word: " << result << endl;  // Expected output: 4 (for "moon")

    return 0;
}

// g++ -std=c++17 Leetcode_0058_Length_of_Last_Word.cpp -o test
