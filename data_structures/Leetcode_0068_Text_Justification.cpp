#include <iostream>
#include <vector>
#include <string>

using namespace std;

class Solution {
public:
    vector<string> fullJustify(vector<string>& words, int maxWidth) {
        vector<string> res;
        vector<string> line;
        int lineLength = 0;
        int i = 0;

        while (i < words.size()) {
            if (lineLength + (int)line.size() + (int)words[i].size() > maxWidth) {
                // Format current line
                int extraSpaces = maxWidth - lineLength;
                int gaps = line.size() - 1;

                if (gaps == 0) {
                    // Only one word: add all spaces at end
                    string lineStr = line[0] + string(extraSpaces, ' ');
                    res.push_back(lineStr);
                } else {
                    int spacePerGap = extraSpaces / gaps;
                    int remainder = extraSpaces % gaps;

                    string lineStr = line[0];
                    for (int j = 1; j < line.size(); ++j) {
                        lineStr += string(spacePerGap, ' ');
                        if (remainder-- > 0) {
                            lineStr += ' ';
                        }
                        lineStr += line[j];
                    }
                    res.push_back(lineStr);
                }

                // Reset line
                line.clear();
                lineLength = 0;
            }

            // Add current word to the line
            line.push_back(words[i]);
            lineLength += words[i].size();
            i++;
        }

        // Handle last line (left justified)
        string lastLine = "";
        for (int j = 0; j < line.size(); ++j) {
            if (j > 0) lastLine += " ";
            lastLine += line[j];
        }

        // Fill remaining space with padding
        lastLine += string(maxWidth - lastLine.size(), ' ');
        res.push_back(lastLine);

        return res;
    }
};

// Test Case
int main() {
    Solution sol;
    vector<string> words = {"What", "must", "be", "acknowledgment", "shall", "be"};
    int maxWidth = 16;

    vector<string> result = sol.fullJustify(words, maxWidth);

    cout << "[";
    for (const string& line : result) {
        cout << "\"" << line << "\", ";
    }
    cout << "]" << endl;

    return 0;
}
// g++ -std=c++17 Leetcode_0068_Text_Justification.cpp -o test